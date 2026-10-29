"""The inference worker: ``python -m labs.inference.worker --workspace <folder>``.

It reads one JSON request per line on standard input and answers with JSON lines on standard
output; anything a library prints goes to standard error instead, so the answers cannot be
corrupted. Requests are handled one at a time, in order; a ``cancel`` message stops the one
running (between tokens, windows or batches).

    → {"id": "r1", "task": "generate", "args": {...}}      ← {"id": "r1", "event": "partial", "data": {...}}
    → {"cancel": "r1"}                                       ← {"id": "r1", "event": "result", "data": {...}}
    → {"quit": true}                                         ← {"id": "r1", "event": "error",
                                                                "message": "..."}

At start it says ``{"event": "ready", "data": {"device": ..., "torch": ..., "transformers": ...}}``.
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import sys
import threading
import traceback
from collections.abc import Callable
from pathlib import Path
from typing import Any

from core.common.masking import mask_text
from core.common.stdio import detach_stdin

__all__ = ["TASKS", "main"]

Handler = Callable[[dict[str, Any], Callable[[dict[str, Any]], None], Callable[[], bool], Path], dict[str,
                                                                                                      Any]]


def _tasks() -> dict[str, Handler]:
    from . import runs, speech, text, vision, visualize

    return {"generate": text.generate, "embed": text.embed, "tokenize": text.tokenize,
            "transcribe": speech.transcribe, "zero_shot": vision.zero_shot,
            "image_search": vision.image_search, "caption": vision.caption, "detect": vision.detect,
            "predict": runs.predict, "sweep": runs.sweep, "visualize": visualize.visualize}


TASKS = ("generate", "embed", "tokenize", "transcribe", "zero_shot", "image_search", "caption", "detect",
         "predict", "sweep", "visualize")


class Channel:
    """Writes answers to the real standard output, whatever else the process prints."""

    def __init__(self) -> None:
        self._out = os.fdopen(os.dup(sys.stdout.fileno()), "w", encoding="utf-8", buffering=1)
        sys.stdout = sys.stderr
        self._lock = threading.Lock()

    def send(self, message: dict[str, Any]) -> None:
        line = json.dumps(message, default=str, separators=(",", ":"))
        with self._lock:
            self._out.write(line + "\n")
            self._out.flush()


def _describe() -> dict[str, Any]:
    info: dict[str, Any] = {"python": sys.version.split()[0]}
    try:
        import torch

        info.update(torch=torch.__version__, device="cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
    except ImportError:
        info.update(torch=None, device="cpu")
    try:
        from importlib.metadata import version

        info["transformers"] = version("transformers")
    except Exception:  # not installed: language and vision models cannot run
        info["transformers"] = None
    return info


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="labs.inference.worker")
    parser.add_argument("--workspace", type=Path, required=True)
    args = parser.parse_args(argv)
    os.environ["TEDO_LAB_WORKSPACE"] = str(args.workspace)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")  # models load from the workspace only
    os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
    channel = Channel()
    stream = detach_stdin()  # before anything can start a child process (Windows: see core.common.stdio)
    requests: queue.Queue[dict[str, Any] | None] = queue.Queue()
    cancelled: set[str] = set()

    def read() -> None:
        for line in stream:
            try:
                message = json.loads(line)
            except ValueError:
                continue
            if not isinstance(message, dict):
                continue
            if "cancel" in message:
                cancelled.add(str(message["cancel"]))
            elif message.get("quit"):
                break
            else:
                requests.put(message)
        requests.put(None)

    threading.Thread(target=read, daemon=True).start()
    channel.send({"event": "ready", "data": _describe()})
    handlers = _tasks()
    while True:
        message = requests.get()
        if message is None:
            return 0
        rid = str(message.get("id", ""))
        task = str(message.get("task", ""))
        if rid in cancelled:
            channel.send({"id": rid, "event": "error", "message": "stopped"})
            continue
        handler = handlers.get(task)
        if handler is None:
            channel.send({"id": rid, "event": "error", "message": f"unknown task {task!r}"})
            continue
        try:
            result = handler(dict(message.get("args") or {}),
                             lambda data, rid=rid: channel.send({"id": rid, "event": "partial",
                                                                 "data": data}),
                             lambda rid=rid: rid in cancelled, args.workspace)
            channel.send({"id": rid, "event": "result", "data": result})
        except InterruptedError:
            channel.send({"id": rid, "event": "error", "message": "stopped"})
        except Exception as exc:  # every failure is reported, the worker keeps serving
            traceback.print_exc()
            text = f"{type(exc).__name__}: {exc}" if str(exc) else type(exc).__name__
            channel.send({"id": rid, "event": "error", "message": mask_text(text)})
        finally:
            cancelled.discard(rid)


if __name__ == "__main__":
    sys.exit(main())
