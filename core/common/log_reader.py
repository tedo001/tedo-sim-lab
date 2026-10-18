"""Reading the lab's log files: records parsed from the text format the lab writes
(``LOG_FORMAT``: time, level, logger, message; the lines after a record, such as a traceback,
belong to it), filtered by level, logger and text, and followed while the file grows. Rotation
and truncation start the file again. The file is opened only while reading, so the logging
handler can always rotate it (Windows refuses to rename a file someone holds open)."""

from __future__ import annotations

import logging
import os
import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

__all__ = ["LEVELS", "LogFollower", "LogRecord", "matches", "parse_lines"]

#: Level name → number; ``""`` is a line the lab did not format (a library's own output).
LEVELS = {"": logging.INFO, "DEBUG": logging.DEBUG, "INFO": logging.INFO, "WARNING": logging.WARNING,
          "ERROR": logging.ERROR, "CRITICAL": logging.CRITICAL}
_HEAD = re.compile(r"^(?P<time>\d{4}-\d\d-\d\d \d\d:\d\d:\d\d(?:,\d{3})?) "
                   r"(?P<level>DEBUG|INFO|WARNING|ERROR|CRITICAL) +(?P<logger>\S+?): (?P<message>.*)$")


@dataclass
class LogRecord:
    time: str
    level: str
    logger: str
    message: str
    #: The lines that followed the head line (a traceback, a multi-line message).
    details: str = ""

    @property
    def levelno(self) -> int:
        return LEVELS.get(self.level, logging.INFO)

    def text(self) -> str:
        head = f"{self.time} {self.level} {self.logger}: {self.message}" if self.level else self.message
        return head + (f"\n{self.details}" if self.details else "")


def parse_lines(lines: Iterable[str], previous: LogRecord | None = None) -> list[LogRecord]:
    """Records from log lines. Lines before the first head line join ``previous`` (the last record
    of an earlier read) or, without one, become records of their own."""
    records: list[LogRecord] = []
    current = previous
    for raw in lines:
        line = raw.rstrip("\r\n")
        match = _HEAD.match(line)
        if match:
            current = LogRecord(match["time"], match["level"], match["logger"], match["message"])
            records.append(current)
        elif current is not None:
            current.details = f"{current.details}\n{line}" if current.details else line
        elif line.strip():
            records.append(LogRecord("", "", "", line))
    return records


def matches(record: LogRecord, *, level: int = 0, logger: str = "", text: str = "") -> bool:
    """At least ``level``; from ``logger`` or a logger below it; containing ``text`` (any case)."""
    if record.levelno < level:
        return False
    if logger and record.logger != logger and not record.logger.startswith(logger + "."):
        return False
    return not text or text.lower() in record.text().lower()


class LogFollower:
    """New lines of a file since the last :meth:`read`; the first read returns the last
    ``tail_bytes`` of it."""

    def __init__(self, path: Path, *, tail_bytes: int = 1_000_000) -> None:
        self.path = Path(path)
        self.tail_bytes = tail_bytes
        self._offset: int | None = None
        self._identity: tuple[int, int] | None = None
        self._partial = b""

    def read(self) -> list[str]:
        try:
            stat = os.stat(self.path)
        except OSError:
            return []
        identity = (stat.st_dev, stat.st_ino)
        if self._offset is None or identity != self._identity or stat.st_size < self._offset:
            started = self._offset is not None
            self._offset = 0 if started or stat.st_size <= self.tail_bytes else stat.st_size - self.tail_bytes
            self._identity, self._partial = identity, b""
            skip_first = not started and self._offset > 0  # the tail starts mid-line
        else:
            skip_first = False
        if stat.st_size == self._offset:
            return []
        try:
            with open(self.path, "rb") as file:
                file.seek(self._offset)
                data = file.read(stat.st_size - self._offset)
        except OSError:
            return []
        self._offset += len(data)
        data = self._partial + data
        lines = data.split(b"\n")
        self._partial = lines.pop()
        if skip_first and lines:
            lines.pop(0)
        return [line.rstrip(b"\r").decode("utf-8", errors="replace") for line in lines]  # Windows: \r\n
