"""The machine, live: a one-off hardware probe in its own process, and a 1 s sampler.

The probe imports PyTorch and asks about CUDA, which must not happen in the UI
process (it would pin a CUDA context and its GPU memory for the app's life), so
it runs as ``python -m core.hardware.info`` and reports JSON. Sampling is psutil
and NVML only, a few hundred microseconds per tick, so it runs on the UI timer.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from PySide6.QtCore import QObject, QProcess, QProcessEnvironment, QTimer, Signal

from core.hardware.info import HardwareInfo
from core.hardware.sampler import ResourceSample, ResourceSampler, SampleHistory

__all__ = ["HardwareService"]

log = logging.getLogger("tedo.hardware")

PROBE_TIMEOUT_MS = 180_000  # the first PyTorch import on a cold disk can take a minute


class HardwareService(QObject):
    info_changed = Signal(object)  # HardwareInfo, or None when the probe failed
    sampled = Signal(object)       # ResourceSample

    def __init__(self, *, python: str, code_root: Path, interval_ms: int = 1000,
                 history: int = 120, probe: bool = True, sampler: ResourceSampler | None = None,
                 parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._python = python
        self._code_root = code_root
        self._probe_enabled = probe
        self._sampler = sampler
        self.history = SampleHistory(history)
        self.info: HardwareInfo | None = None
        self.info_error: str | None = None
        self._probe: QProcess | None = None
        self._finished_probes: list[QProcess] = []
        self._timer = QTimer(self)
        self._timer.setInterval(interval_ms)
        self._timer.timeout.connect(self.sample_now)
        self._started = False
        self._timed_out = False
        self._stopping = False

    # Lifecycle ----------------------------------------------------------------
    def start(self) -> None:
        if self._started:
            return
        self._started = True
        if self._sampler is None:
            self._sampler = ResourceSampler()
        self.sample_now()
        self._timer.start()
        if self._probe_enabled:
            self.reprobe()

    def stop(self) -> None:
        self._stopping = True  # a probe killed on the way out is not a failure
        self._timer.stop()
        if self._probe is not None and self._probe.state() != QProcess.ProcessState.NotRunning:
            self._probe.kill()
            self._probe.waitForFinished(2_000)
        if self._sampler is not None:
            self._sampler.close()
        self._started = False

    @property
    def probing(self) -> bool:
        return self._probe is not None

    @property
    def gpu_monitoring(self) -> bool:
        return bool(self._sampler and self._sampler.gpu_monitoring)

    @property
    def gpu_monitoring_note(self) -> str | None:
        return self._sampler.nvml_error if self._sampler else None

    # Sampling -----------------------------------------------------------------
    def sample_now(self) -> ResourceSample | None:
        if self._sampler is None:
            return None
        sample = self._sampler.sample()
        self.history.add(sample)
        self.sampled.emit(sample)
        return sample

    # Probe ----------------------------------------------------------------------
    def reprobe(self) -> None:
        """Run the hardware probe again (e.g. after installing a CUDA build of PyTorch)."""
        if self._probe is not None:
            return
        process = QProcess()  # Python-owned; released after its signals finish (see jobs.py)
        environment = QProcessEnvironment.systemEnvironment()
        pythonpath = os.pathsep.join(filter(None, [str(self._code_root),
                                                   os.environ.get("PYTHONPATH", "")]))
        environment.insert("PYTHONPATH", pythonpath)
        process.setProcessEnvironment(environment)
        process.setWorkingDirectory(str(self._code_root))
        process.finished.connect(lambda code, _status: self._probe_finished(code))
        process.errorOccurred.connect(self._probe_error)
        self._probe = process
        self._timed_out = False
        QTimer.singleShot(PROBE_TIMEOUT_MS, self._probe_timed_out)
        process.start(self._python, ["-m", "core.hardware.info"])

    def _probe_finished(self, exit_code: int) -> None:
        process = self._release_probe()
        if process is None or self._stopping:
            return
        output = bytes(process.readAllStandardOutput().data()).decode("utf-8", errors="replace")
        lines = [line for line in output.splitlines() if line.startswith("{")]
        try:
            if self._timed_out:
                raise ValueError(f"no answer within {PROBE_TIMEOUT_MS // 1000} s")
            if exit_code != 0 or not lines:
                raise ValueError(f"probe exited with {exit_code}")
            self.info, self.info_error = HardwareInfo.from_json(lines[-1]), None
        except (ValueError, TypeError, KeyError) as exc:
            errors = bytes(process.readAllStandardError().data()).decode("utf-8", errors="replace")
            self.info_error = f"Hardware probe failed: {exc}"
            log.warning("%s\n%s", self.info_error, errors[-2000:])
        self.info_changed.emit(self.info)

    def _probe_error(self, error: QProcess.ProcessError) -> None:
        if error == QProcess.ProcessError.FailedToStart and self._release_probe() is not None:
            self.info_error = f"Could not start {self._python} to probe the hardware."
            self.info_changed.emit(None)

    def _probe_timed_out(self) -> None:
        if self._probe is not None and self._probe.state() != QProcess.ProcessState.NotRunning:
            self._timed_out = True
            self._probe.kill()

    def _release_probe(self) -> QProcess | None:
        process, self._probe = self._probe, None
        if process is not None:
            self._finished_probes.append(process)
            QTimer.singleShot(0, self._drop_finished)
        return process

    def _drop_finished(self) -> None:
        self._finished_probes = [p for p in self._finished_probes
                                 if p.state() != QProcess.ProcessState.NotRunning]
