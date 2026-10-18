"""The log files the lab writes (app.log, errors.log, masked), reading them back as records,
filtering, following a growing file, and comparing release versions."""

from __future__ import annotations

import logging

import pytest

from core.common import Secret
from core.common.log_reader import LogFollower, LogRecord, matches, parse_lines
from core.common.logging_setup import set_level, setup_logging
from core.common.updates import is_newer, version_key

LINES = [
    "2026-10-18 09:00:01,100 INFO    tedo.jobs: Run 1 started",
    "2026-10-18 09:00:02,200 WARNING tedo.git: push refused",
    "2026-10-18 09:00:03,300 ERROR   tedo.ui.pages: Page failed",
    "Traceback (most recent call last):",
    '  File "/lab/app/page.py", line 12, in build',
    "ValueError: bad",
    "2026-10-18 09:00:04,400 DEBUG   mlflow.store: engine ready",
]


def test_records_keep_their_tracebacks() -> None:
    records = parse_lines(["a library's own output", *LINES])
    assert [r.level for r in records] == ["", "INFO", "WARNING", "ERROR", "DEBUG"]
    error = records[3]
    assert error.logger == "tedo.ui.pages" and error.message == "Page failed"
    assert error.details.splitlines() == LINES[3:6] and error.levelno == logging.ERROR
    assert error.text().startswith("2026-10-18 09:00:03,300 ERROR tedo.ui.pages: Page failed\nTraceback")
    previous = LogRecord("t", "ERROR", "tedo", "first part")
    assert parse_lines(["second part", LINES[0]], previous)[0].logger == "tedo.jobs"
    assert previous.details == "second part"  # a record cut by a read is joined up


def test_filters() -> None:
    records = parse_lines(LINES)
    assert [r.level for r in records if matches(r, level=logging.WARNING)] == ["WARNING", "ERROR"]
    assert [r.logger for r in records if matches(r, logger="tedo.ui")] == ["tedo.ui.pages"]
    below = ["tedo.jobs", "tedo.git", "tedo.ui.pages"]
    assert [r.logger for r in records if matches(r, logger="tedo")] == below
    assert not any(matches(r, logger="tedo.u") for r in records)  # a logger name, not a prefix of letters
    assert [r.level for r in records if matches(r, text="VALUEERROR")] == ["ERROR"]  # details count, any case


def test_a_growing_file_is_followed_through_rotation(tmp_path) -> None:
    log = tmp_path / "app.log"
    log.write_text("x" * 50 + "\n" + "\n".join(LINES[:2]) + "\n", encoding="utf-8")
    follower = LogFollower(log, tail_bytes=len(LINES[1]) + 11)  # starts 9 characters before line 0 ends
    assert follower.read() == [LINES[1]]  # the tail, without the line it starts in the middle of
    assert follower.read() == []
    with log.open("a", encoding="utf-8") as file:
        file.write(LINES[2] + "\npartial")
    assert follower.read() == [LINES[2]]  # a line still being written waits for its end
    with log.open("a", encoding="utf-8") as file:
        file.write(" line\n")
    assert follower.read() == ["partial line"]
    log.rename(tmp_path / "app.log.1")  # what RotatingFileHandler does
    log.write_text(LINES[6] + "\n", encoding="utf-8")
    assert follower.read() == [LINES[6]]
    log.write_text("", encoding="utf-8")  # truncated
    assert follower.read() == []
    assert LogFollower(tmp_path / "missing.log").read() == []


def test_errors_log_keeps_only_problems_and_both_are_masked(tmp_path) -> None:
    root = logging.getLogger()
    before = list(root.handlers)
    try:
        log = setup_logging(tmp_path, "INFO", console=False)
        token = Secret("ghp_ABCDEFGHIJKLMNOPQRSTUVWX0123456789").reveal()
        logging.getLogger("tedo.test").info("all is well")
        logging.getLogger("tedo.test").warning("token %s was refused", token)
        set_level("WARNING")
        logging.getLogger("tedo.test").info("not recorded any more")
        assert log.level == logging.WARNING
    finally:
        setup_logging(None, console=False)  # close the files
        for handler in [h for h in root.handlers if h not in before]:
            root.removeHandler(handler)
    app_log = (tmp_path / "app.log").read_text(encoding="utf-8")
    errors_log = (tmp_path / "errors.log").read_text(encoding="utf-8")
    assert "all is well" in app_log and "not recorded" not in app_log
    assert "all is well" not in errors_log and "was refused" in errors_log
    assert "ghp_ABCDEF" not in app_log + errors_log


@pytest.mark.parametrize(("tag", "current", "newer"), [
    ("v0.2.0", "0.1.0", True), ("v0.1.0", "0.1.0", False), ("0.10.0", "0.9.2", True),
    ("v1.0", "1.0.0", False), ("v1.0.1", "1.0", True), ("nightly", "0.1.0", False),
    ("v0.1.0", "0.2.0", False),
])
def test_release_versions_compare_as_numbers(tag: str, current: str, newer: bool) -> None:
    assert is_newer(tag, current) is newer
    assert version_key("v0.2.1-beta") == (0, 2, 1) and version_key("none") == ()
