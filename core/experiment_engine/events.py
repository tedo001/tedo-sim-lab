"""The worker's progress protocol: one JSON object per line on stdout.

Anything else the worker process prints (a library writing to stdout from C,
a stray ``print``) is not lost and does not break the stream: the reader turns
lines that are not events into ``log`` events.
"""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, TextIO

from core.common.masking import mask_text

from .runner import RunCallbacks, RunContext, RunResult

__all__ = ["EventType", "JsonLinesCallbacks", "ProgressEvent", "parse_line"]

EventType = Literal["run_start", "epoch_start", "step", "epoch_end", "log", "artifact", "run_end"]
_TYPES = {"run_start", "epoch_start", "step", "epoch_end", "log", "artifact", "run_end"}


@dataclass(frozen=True)
class ProgressEvent:
    run_id: str
    type: EventType
    payload: dict[str, Any] = field(default_factory=dict)
    ts: float = field(default_factory=time.time)

    def to_json(self) -> str:
        return json.dumps({"run_id": self.run_id, "type": self.type, "ts": round(self.ts, 3),
                           "payload": self.payload}, default=str, separators=(",", ":"))

    @classmethod
    def from_json(cls, line: str) -> ProgressEvent:
        data = json.loads(line)
        if not isinstance(data, dict) or data.get("type") not in _TYPES:
            raise ValueError("not a progress event")
        return cls(str(data.get("run_id", "")), data["type"], dict(data.get("payload") or {}),
                   float(data.get("ts", time.time())))


def parse_line(line: str, run_id: str = "", *, stream: str = "stdout") -> ProgressEvent:
    """An event from one line of worker output; non-event lines become ``log`` events."""
    text = line.rstrip("\r\n")
    if text.startswith("{"):
        try:
            return ProgressEvent.from_json(text)
        except (ValueError, TypeError):
            pass
    return ProgressEvent(run_id, "log", {"line": mask_text(text), "stream": stream})


class JsonLinesCallbacks(RunCallbacks):
    """Writes every callback as a :class:`ProgressEvent` line to ``stream`` (thread-safe)."""

    def __init__(self, stream: TextIO, run_id: str) -> None:
        self.stream = stream
        self.run_id = run_id
        self._lock = threading.Lock()

    def emit(self, type: EventType, **payload: Any) -> None:
        line = ProgressEvent(self.run_id, type, payload).to_json()
        with self._lock:
            self.stream.write(line + "\n")
            self.stream.flush()

    def on_run_start(self, ctx: RunContext) -> None:
        self.emit("run_start", device=ctx.device, run_dir=str(ctx.run_dir))

    def on_epoch_start(self, epoch: int, total: int) -> None:
        self.emit("epoch_start", epoch=epoch, total=total)

    def on_step(self, step: int, total: int, metrics: Mapping[str, float]) -> None:
        self.emit("step", step=step, total=total, metrics=dict(metrics))

    def on_epoch_end(self, epoch: int, metrics: Mapping[str, float]) -> None:
        self.emit("epoch_end", epoch=epoch, metrics=dict(metrics))

    def on_log(self, line: str) -> None:
        self.emit("log", line=mask_text(line), stream="runner")

    def on_artifact(self, path: Path, kind: str) -> None:
        self.emit("artifact", path=str(path), kind=kind)

    def on_run_end(self, result: RunResult) -> None:
        self.emit("run_end", status=result.status, metrics=result.metrics,
                  best_checkpoint=str(result.best_checkpoint) if result.best_checkpoint else None,
                  duration_s=round(result.duration_s, 3), error=result.error)
