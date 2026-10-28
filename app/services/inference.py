"""The app's side of the inference worker (``labs/inference/worker.py``): one process, under the
experiment Python, started on the first request and kept, so a model loaded once answers at
once afterwards. Requests carry an id; answers arrive as signals (``partial`` while it works,
then ``finished`` or ``failed``). The process never runs in the window's own process, so CUDA
and large models stay out of it.

A request sent while the worker starts waits for its "ready"; if the worker dies, the requests
in flight fail with the reason and the next request starts a new one."""

from __future__ import annotations

import json
import logging
import os
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

from PySide6.QtCore import QObject, QProcess, QProcessEnvironment, Signal

from core.common import AppConfig, experiment_python
from core.common.paths import AppPaths

__all__ = ["InferenceService"]

log = logging.getLogger("tedo.inference")
Callback = Callable[[dict[str, Any]], None]


class InferenceService(QObject):
    partial = Signal(str, dict)     # request id, data
    finished = Signal(str, dict)    # request id, result
    failed = Signal(str, str)       # request id, message
    state_changed = Signal(str)     # "starting" | "ready" | "stopped"

    def __init__(self, paths: AppPaths, config: AppConfig, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.paths, self.config = paths, config
        self._process: QProcess | None = None
        self._buffer = b""
        self._waiting: list[bytes] = []
        self._pending: dict[str, tuple[Callback | None, Callback | None, Callable[[str], None] | None]] = {}
        self.info: dict[str, Any] = {}
        self.state = "stopped"

    # Requests ------------------------------------------------------------------------------
    def request(self, task: str, args: dict[str, Any], *, on_partial: Callback | None = None,
                on_done: Callback | None = None, on_error: Callable[[str], None] | None = None) -> str:
        """Send ``task`` with ``args``; returns the request id. The callbacks are optional
        shortcuts for the signals, called for this request only."""
        rid = uuid.uuid4().hex[:12]
        self._pending[rid] = (on_partial, on_done, on_error)
        line = (json.dumps({"id": rid, "task": task, "args": args}) + "\n").encode("utf-8")
        if self.state == "ready" and self._process is not None:
            self._process.write(line)
        else:
            self._waiting.append(line)
            self._start()
        return rid

    def cancel(self, rid: str) -> None:
        if self._process is not None and rid in self._pending:
            self._process.write((json.dumps({"cancel": rid}) + "\n").encode("utf-8"))

    def busy(self) -> bool:
        return bool(self._pending)

    # Process -------------------------------------------------------------------------------
    def _start(self) -> None:
        if self._process is not None:
            return
        python = experiment_python(self.config)
        if not python:
            self._fail_all("No experiment Python is set up yet: Settings → Experiment Python.")
            return
        process = QProcess()
        env = QProcessEnvironment.systemEnvironment()
        code_root = str(self.paths.code_root)
        env.insert("PYTHONPATH", os.pathsep.join(filter(None, [code_root, os.environ.get("PYTHONPATH", "")])))
        env.insert("PYTHONUNBUFFERED", "1")
        env.insert("PYTHONIOENCODING", "utf-8")
        env.insert("HF_HOME", str(self.paths.models / "huggingface-cache"))
        env.insert("TORCH_HOME", str(self.paths.models / "torch-hub"))
        process.setProcessEnvironment(env)
        process.setWorkingDirectory(code_root)
        process.setProgram(python)
        process.setArguments(["-m", "labs.inference.worker", "--workspace", str(self.paths.workspace)])
        process.setProcessChannelMode(QProcess.ProcessChannelMode.SeparateChannels)
        process.readyReadStandardOutput.connect(lambda: self._read(process))
        process.readyReadStandardError.connect(lambda: self._stderr(process))
        process.finished.connect(lambda code, status: self._ended(process, code))
        process.errorOccurred.connect(lambda error: self._error(process, error))
        self._process, self._buffer = process, b""
        self._set_state("starting")
        process.start()

    def _read(self, process: QProcess) -> None:
        if process is not self._process:
            return
        self._buffer += bytes(process.readAllStandardOutput().data())
        while b"\n" in self._buffer:
            line, self._buffer = self._buffer.split(b"\n", 1)
            try:
                message = json.loads(line.decode("utf-8", errors="replace"))
            except ValueError:
                continue
            self._handle(message)

    def _handle(self, message: dict[str, Any]) -> None:
        event, rid = message.get("event"), str(message.get("id", ""))
        if event == "ready":
            self.info = dict(message.get("data") or {})
            self._set_state("ready")
            for line in self._waiting:
                self._process.write(line)  # type: ignore[union-attr]
            self._waiting.clear()
            return
        partial, done, error = self._pending.get(rid, (None, None, None))
        if event == "partial":
            data = dict(message.get("data") or {})
            self.partial.emit(rid, data)
            if partial is not None:
                partial(data)
        elif event == "result":
            self._pending.pop(rid, None)
            data = dict(message.get("data") or {})
            self.finished.emit(rid, data)
            if done is not None:
                done(data)
        elif event == "error":
            self._pending.pop(rid, None)
            text = str(message.get("message", "failed"))
            log.info("Inference request failed: %s", text)
            self.failed.emit(rid, text)
            if error is not None:
                error(text)

    def _stderr(self, process: QProcess) -> None:
        text = bytes(process.readAllStandardError().data()).decode("utf-8", errors="replace").strip()
        if text:
            log.debug("inference worker: %s", text[-2000:])

    def _error(self, process: QProcess, error: QProcess.ProcessError) -> None:
        if error == QProcess.ProcessError.FailedToStart and process is self._process:
            self._process = None
            self._set_state("stopped")
            self._fail_all(f"The inference worker could not start ({process.program()}).")

    def _ended(self, process: QProcess, code: int) -> None:
        if process is not self._process:
            return
        self._process = None
        self._set_state("stopped")
        if self._pending:
            self._fail_all(f"The inference worker stopped (exit code {code}); the next request starts it "
                           "again.")

    def _fail_all(self, text: str) -> None:
        pending, self._pending = self._pending, {}
        self._waiting.clear()
        for rid, (_partial, _done, error) in pending.items():
            self.failed.emit(rid, text)
            if error is not None:
                error(text)

    def _set_state(self, state: str) -> None:
        self.state = state
        self.state_changed.emit(state)

    def stop(self) -> None:
        """Ask the worker to quit (app exit or a change of experiment Python); kill it if it lingers."""
        process = self._process
        if process is None:
            return
        self._process = None
        self._fail_all("stopped")
        process.write(b'{"quit": true}\n')
        process.closeWriteChannel()
        if not process.waitForFinished(3000):
            process.kill()
            process.waitForFinished(2000)
        self._set_state("stopped")

    def workspace_file(self, name: str) -> Path:
        """A scratch file for inputs made in the window (drawings, recordings)."""
        folder = self.paths.workspace / ".tedo" / "inputs"
        folder.mkdir(parents=True, exist_ok=True)
        return folder / name
