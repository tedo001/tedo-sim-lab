"""The Logs page (reading, filtering, following, opening a traceback's file, the log level), the
problem count in the top bar, settings changed in the app, and the update check."""

from __future__ import annotations

import json
import logging
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
import yaml
from PySide6.QtCore import Qt

from app.services.updates import UpdateCheck
from app.ui.pages.logs import LogsPage
from core.common.config import ConfigError

LINES = [
    "2026-10-18 09:00:01,100 INFO    tedo.jobs: Run 1 started",
    "2026-10-18 09:00:02,200 WARNING tedo.git: push refused",
    "2026-10-18 09:00:03,300 ERROR   tedo.ui: Page failed",
    "Traceback (most recent call last):",
    '  File "{path}", line 2, in build',
    "ValueError: bad",
]


@pytest.fixture
def logged(ctx, tmp_path):
    source = tmp_path / "page.py"
    source.write_text("x = 1\nraise ValueError('bad')\n")
    ctx.paths.logs.mkdir(parents=True, exist_ok=True)
    ctx.logs.app_log.write_text("\n".join(LINES).format(path=source) + "\n", encoding="utf-8")
    return source


def test_the_logs_page_reads_filters_and_follows(ctx, qtbot, logged) -> None:
    opened = []
    ctx.open_file = lambda path, line=1: opened.append((path, line))
    page = LogsPage(ctx)
    qtbot.addWidget(page)
    page.show()
    assert page.model.rowCount() == 3 and "3 of 3 records" in page.summary.text()
    page.level_filter.setCurrentIndex(page.level_filter.findData(logging.WARNING))
    assert [page.model.record(r).level for r in range(page.model.rowCount())] == ["WARNING", "ERROR"]
    page.logger_filter.setCurrentIndex(page.logger_filter.findData("tedo.ui"))
    assert page.model.rowCount() == 1
    page.table.selectRow(0)
    assert "ValueError: bad" in page.details.toPlainText()
    page.level_filter.setCurrentIndex(0)
    page.logger_filter.setCurrentIndex(0)
    page.search.setText("PUSH")
    assert page.model.rowCount() == 1 and "push refused" in page.copy_shown()
    page.search.clear()
    with ctx.logs.app_log.open("a", encoding="utf-8") as file:
        file.write("2026-10-18 09:00:05,000 INFO    tedo.sim: Saved a simulation\n")
    qtbot.waitUntil(lambda: page.model.rowCount() == 4, timeout=5_000)  # followed while shown
    assert page.logger_filter.findData("tedo.sim") >= 0
    page.show_problems()
    assert page.model.rowCount() == 2
    # Double-click a traceback line: the file opens at that line.
    page.details.setPlainText(page.model.record(1).text())
    block = page.details.document().findBlockByNumber(2)
    cursor = page.details.textCursor()
    cursor.setPosition(block.position())
    rect = page.details.cursorRect(cursor)
    qtbot.mouseDClick(page.details.viewport(), Qt.MouseButton.LeftButton, pos=rect.center())
    assert opened == [(logged, 2)]


def test_sources_include_problems_older_logs_and_runs(ctx, qtbot, logged) -> None:
    ctx.logs.errors_log.write_text(LINES[1] + "\n", encoding="utf-8")
    (ctx.paths.logs / "app.log.1").write_text(LINES[0] + "\n", encoding="utf-8")
    experiment = ctx.store.create_experiment("mnist cnn", "image_classification", "{}", "hash")
    run_dir = ctx.paths.experiments / "mnist-cnn" / "run1"
    run_dir.mkdir(parents=True)
    (run_dir / "run.log").write_text(LINES[0] + "\n", encoding="utf-8")
    run_id = ctx.store.create_run(experiment, run_dir, "labs.x:Runner")
    ctx.store.create_run(experiment, ctx.paths.experiments / "never-started", "labs.x:Runner")  # no log
    page = LogsPage(ctx)
    qtbot.addWidget(page)
    titles = [page.source.itemText(i) for i in range(page.source.count())]
    assert titles[:2] == ["App log · app.log", "Warnings and errors · errors.log"]
    assert "Older · app.log.1" in titles and f"Run · mnist cnn · {run_id[:8]}" in titles
    assert sum(title.startswith("Run · ") for title in titles) == 1
    page.open_source(ctx.logs.errors_log)
    assert page.model.rowCount() == 1 and page.model.record(0).level == "WARNING"


def test_the_log_level_changes_at_once_and_is_saved(ctx, qtbot, logged) -> None:
    page = LogsPage(ctx)
    qtbot.addWidget(page)
    lab = logging.getLogger("tedo")
    before = lab.level
    try:
        page.set_log_level("DEBUG")
        assert lab.level == logging.DEBUG and ctx.config.log_level == "DEBUG"
        assert yaml.safe_load(ctx.paths.settings_file.read_text())["log_level"] == "DEBUG"
    finally:
        lab.setLevel(before)


def test_errors_are_counted_in_the_top_bar(window, ctx, qtbot) -> None:
    assert not window.top_bar.problems_button.isVisible()
    logging.getLogger("tedo.test").error("something broke")
    qtbot.waitUntil(lambda: window.top_bar.problems_button.isVisible(), timeout=5_000)
    assert window.top_bar.problems_button.text() == "1 error"
    assert "something broke" in window.top_bar.problems_button.toolTip()
    window.top_bar.problems_button.click()
    assert window.current_page_id() == "logs" and ctx.logs.problems == 0
    assert not window.top_bar.problems_button.isVisible()


def test_settings_change_live_and_wrong_values_change_nothing(ctx, qtbot) -> None:
    seen = []
    ctx.settings.changed.connect(seen.append)
    changed = ctx.settings.update(max_concurrent_runs=3, mlflow_tracking=False)
    assert changed == {"max_concurrent_runs", "mlflow_tracking"} and seen == [changed]
    assert ctx.jobs.max_concurrent_runs == 3
    saved = yaml.safe_load(ctx.paths.settings_file.read_text())
    assert saved == {"max_concurrent_runs": 3, "mlflow_tracking": False}
    with pytest.raises(ConfigError, match="max_concurrent_runs"):
        ctx.settings.update(max_concurrent_runs=99, log_level="DEBUG")
    assert ctx.config.max_concurrent_runs == 3 and ctx.config.log_level == "INFO"
    assert ctx.settings.update(max_concurrent_runs=3) == set()


def test_the_settings_page_saves_and_explains(window, ctx, qtbot) -> None:
    window.navigate("settings")
    page = window.page_widget("settings")
    form = page.form
    form.concurrent.setValue(2)
    form.device.setCurrentIndex(form.device.findData("cpu"))
    assert form.save() == {"max_concurrent_runs", "default_device"}
    assert form.status.text() == ("Saved. Concurrent runs: in effect now; Default device is what the builder "
                                  "starts experiments on, unless one was chosen there.")
    builder = window.page_widget("experiment_builder")
    assert builder.form.torch.device.currentData() == "cpu"  # a default not overridden follows
    assert form.save() == set() and form.status.text() == "Nothing changed."
    page.edit_file()
    assert window.current_page_id() == "code"
    assert window.page_widget("code").tabs.current_path() == ctx.paths.settings_file.resolve()


class FakeReleases(BaseHTTPRequestHandler):
    tag = "v9.9.0"

    def log_message(self, *args) -> None:
        pass

    def do_GET(self) -> None:  # noqa: N802
        if self.path != "/repos/tedo001/tedo-sim-lab/releases/latest" or not self.tag:
            self.send_response(404)
            self.end_headers()
            return
        body = json.dumps({"tag_name": self.tag, "name": f"Lab {self.tag}",
                           "html_url": f"https://github.com/tedo001/tedo-sim-lab/releases/{self.tag}",
                           "published_at": "2026-10-01T00:00:00Z"}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


@pytest.fixture
def releases(monkeypatch):
    for name in ("http_proxy", "HTTP_PROXY", "https_proxy", "HTTPS_PROXY", "all_proxy", "ALL_PROXY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("no_proxy", "*")
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), FakeReleases)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()
    FakeReleases.tag = "v9.9.0"


def test_the_update_check(window, ctx, qtbot, releases) -> None:
    ctx.updates.api = releases
    window.navigate("settings")
    page = window.page_widget("settings")
    page.check_updates()
    qtbot.waitUntil(lambda: page.check_button.isEnabled(), timeout=10_000)
    assert "Lab v9.9.0 is available" in page.update_text.text() and not page.release_button.isHidden()
    assert window.top_bar.update_button.text() == "v9.9.0 available"
    FakeReleases.tag = "v0.0.1"
    page.check_updates()
    qtbot.waitUntil(lambda: page.check_button.isEnabled(), timeout=10_000)
    assert "latest version" in page.update_text.text() and page.release_button.isHidden()
    FakeReleases.tag = ""
    page.check_updates()
    qtbot.waitUntil(lambda: page.check_button.isEnabled(), timeout=10_000)
    assert page.update_text.text() == "No release is published yet."
    page.at_start.setChecked(True)
    assert ctx.config.check_updates and yaml.safe_load(ctx.paths.settings_file.read_text())["check_updates"]


def test_an_unreachable_update_server_is_reported(ctx, qtbot) -> None:
    check = UpdateCheck(ctx.jobs, api="http://127.0.0.1:9")
    results = []
    check.finished.connect(results.append)
    check.check()
    qtbot.waitUntil(lambda: bool(results), timeout=20_000)
    assert results[0].summary().startswith("Could not check for updates")
