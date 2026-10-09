"""Phase 9 pages: Deep Learning builder, Plugin Store."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from app.ui.pages.deep_learning import DeepLearningPage
from tests.fixtures.fake_vision import write_fake_mnist
from tests.plugins.test_sources import KAGGLE_KEY, KAGGLE_USER, FakeApis, server, zipped  # noqa: F401
from tests.ui.test_experiment_service import wait_for


def test_deep_learning_page_edits_and_checks_the_stack(ctx, qtbot) -> None:
    page = DeepLearningPage(ctx)
    qtbot.addWidget(page)
    page.dataset.setCurrentIndex(page.dataset.findData("mnist"))
    assert page.shape_text.text() == "1×28×28 → 10"
    assert page.table.rowCount() == len(page.steps) == 13 and "class LayerStack" in page.code.toPlainText()
    assert not page.run_button.isEnabled() and page.open_button.isEnabled()  # not downloaded yet
    editor = page.editor
    editor.list.setCurrentRow(len(editor.layers) - 1)
    editor.kind.setCurrentIndex(editor.kind.findData("maxpool"))
    for _ in range(3):
        editor.add_selected_kind()  # pooling a vector cannot work
    assert not page.steps and "needs image maps" in page.problem.text() and not page.open_button.isEnabled()
    for _ in range(3):
        editor.remove_selected()
    assert page.steps and not page.problem.isVisibleTo(page)
    editor.list.setCurrentRow(0)
    editor.form_holder.findChild(type(page.epochs), "setting_channels").setValue(8)
    assert editor.layers[0]["channels"] == 8 and page.steps[0].shape == (8, 28, 28)
    editor.move(1)
    assert editor.layers[1]["type"] == "conv" and editor.list.currentRow() == 1
    editor.set_layers([])
    assert [s.kind for s in page.steps] == ["flatten", "linear"]


def test_deep_learning_page_trains_its_network(ctx, qtbot) -> None:
    pytest.importorskip("torch")
    write_fake_mnist(ctx.paths.datasets)
    page = DeepLearningPage(ctx)
    qtbot.addWidget(page)
    page.dataset.setCurrentIndex(page.dataset.findData("mnist"))
    page.refresh()
    assert page.run_button.isEnabled()
    spec = page.spec()
    assert spec.model.model == "layer_stack" and spec.model.params["layers"] == page.editor.layers
    page.epochs.setValue(1)
    run_id = page.train()
    assert run_id and "Queued" in page.train_text.text()
    view = wait_for(qtbot, ctx, run_id, ("completed", "failed"))
    assert view.status == "completed", ctx.experiments.log_lines(run_id, limit=40)


def test_plugin_store_connects_and_downloads(ctx, qtbot, server, monkeypatch) -> None:  # noqa: F811
    import plugins.kaggle.plugin as kaggle_module
    from app.ui.pages.plugin_store import PluginStorePage

    monkeypatch.setattr(kaggle_module, "API", f"{server}/api/v1")
    rows = [{"ref": "ada/iris-flowers", "title": "Iris flowers", "licenseName": "CC0: Public Domain"}]
    FakeApis.routes = {"/api/v1/datasets/list?search=iris&page=1": (200, json.dumps(rows).encode()),
                       "/api/v1/datasets/list?page=1&search=mnist": (200, b"[]"),
                       "/api/v1/datasets/download/ada/iris-flowers": (200, zipped({"iris.csv": b"a\n1\n"}))}
    page = PluginStorePage(ctx)
    qtbot.addWidget(page)
    page.select("kaggle")
    assert page.ctx.plugins.plugin("kaggle").status == "not_connected" and not page.browse.isEnabled()
    assert not page.install_card.isVisibleTo(page) and page.account.isVisibleTo(page)
    for row, value in ((0, KAGGLE_USER), (1, KAGGLE_KEY)):
        page.account_table.selectRow(row)
        page.value.setText(value)
        page.save_credential()
        assert not page.value.text() and "stored in the OS keyring" in page.account_text.text()
    assert ctx.plugins.plugin("kaggle").status == "available" and page.browse.isEnabled()
    with qtbot.waitSignal(ctx.plugins.finished, timeout=30_000):
        page.test()
    assert page.account_text.text().startswith("Connected.")
    page.browse.query.setText("iris")
    with qtbot.waitSignal(ctx.plugins.finished, timeout=30_000):
        page.browse.search()
    assert page.browse.table.rowCount() == 1
    page.browse.table.selectRow(0)
    assert not page.browse.download_button.isEnabled()  # licence not confirmed yet
    page.browse.accept.setChecked(True)
    with qtbot.waitSignal(ctx.plugins.finished, timeout=30_000):
        page.browse.download()
    folder = ctx.paths.datasets / "kaggle" / "ada__iris-flowers"
    assert (folder / "iris.csv").is_file() and "Saved to" in page.browse.progress.text()
    page.account_table.selectRow(1)
    page.remove_credential()
    assert ctx.plugins.plugin("kaggle").status == "not_connected"


def test_plugin_store_lists_every_plugin_honestly(ctx, qtbot) -> None:
    from app.ui.pages.plugin_store import PluginStorePage

    page = PluginStorePage(ctx)
    qtbot.addWidget(page)
    assert page.table.rowCount() == len(ctx.catalog.plugins.manifests())
    page.select("paddleocr")
    assert "Planned for v0.5" in page.actions_text.text() and not page.install_button.isEnabled()
    assert not page.browse.isVisibleTo(page)
    page.select("huggingface")
    assert page.browse.isVisibleTo(page) and page.browse.isEnabled()  # public repositories need no token
    page.select("mlflow")
    assert page.page_button.isVisibleTo(page) and "MLflow" in page.page_button.text()


def test_shell_command_per_platform() -> None:
    from app.services.terminal import shell_command
    from core.common import AppConfig

    program, args = shell_command(AppConfig(), "dir", windows=True)
    assert "powershell" in program.lower() or "pwsh" in program.lower()
    assert args[-2:] == ["-Command", "dir"]
    posix = shell_command(AppConfig(terminal_shell="/bin/sh"), "ls", windows=False)
    assert posix == ("/bin/sh", ["-c", "ls"])
    assert shell_command(AppConfig(terminal_shell="cmd.exe"), "dir")[1] == ["/d", "/c", "dir"]


@pytest.mark.skipif(sys.platform == "win32", reason="uses POSIX shell commands")
def test_terminal_runs_commands_in_the_workspace(ctx, qtbot) -> None:
    from PySide6.QtCore import QEvent, Qt
    from PySide6.QtGui import QKeyEvent

    from app.ui.pages.terminal import TerminalPage

    page = TerminalPage(ctx)
    qtbot.addWidget(page)
    with qtbot.waitSignal(page.session.finished, timeout=20_000):
        page.prompt.setText('echo "$TEDO_LAB_WORKSPACE"; pwd; echo oops >&2')
        page.run()
    assert page.text().count(str(ctx.paths.workspace)) >= 3 and "oops" in page.text()  # prompt, env, pwd
    with qtbot.waitSignal(page.session.finished, timeout=5_000):
        page.run("cd datasets")
    assert page.session.cwd == ctx.paths.datasets
    with qtbot.waitSignal(page.session.finished, timeout=5_000):
        page.run("cd no-such-folder")
    assert "no such folder" in page.text() and page.session.cwd == ctx.paths.datasets
    with qtbot.waitSignal(page.session.finished, timeout=20_000):
        page.run("which python")
    assert str(Path(sys.executable).parent) in page.text()  # the experiment Python comes first
    with qtbot.waitSignal(page.session.finished, timeout=20_000) as signal:
        page.run("sleep 30")
        assert page.stop_button.isEnabled() and not page.run("echo busy")
        page.session.stop()
    assert signal.args[1] is True and "[stopped]" in page.text()
    page.prompt.setFocus()
    qtbot.keyClick(page.prompt, Qt.Key.Key_Up)
    assert page.prompt.text() == "sleep 30"
    down = QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_Down, Qt.KeyboardModifier.NoModifier)
    page.eventFilter(page.prompt, down)
    assert page.prompt.text() == ""
    page.run("clear")
    assert page.text() == ""


def test_jupyter_page_notebooks_and_lab(ctx, qtbot) -> None:
    pytest.importorskip("sklearn")
    from app.ui.pages.jupyter_page import JupyterPage
    from core.common.masking import mask_text
    from labs.classical_ml.presets import CLASSICAL_PRESETS

    run_id = ctx.experiments.launch(CLASSICAL_PRESETS[0].make())
    assert wait_for(qtbot, ctx, run_id, ("completed", "failed"), timeout=120_000).status == "completed"
    page = JupyterPage(ctx)
    qtbot.addWidget(page)
    qtbot.waitUntil(lambda: page._check_job is None, timeout=30_000)
    assert page.state.text() in ("Not installed", "Stopped") and not page.browser_button.isEnabled()
    assert page.install_button.isVisibleTo(page) == (page.version is None)
    assert page.table.rowCount() == 0 and not page.viewer_card.isVisibleTo(page)
    page.run_choice.setCurrentIndex(page.run_choice.findData(run_id))
    page.name.setText("Iris look")
    path = page.create()
    assert path == ctx.paths.notebooks / "iris-look.ipynb" and page.table.rowCount() == 1
    assert page.files[0].runs == (run_id,) and page.table.item(0, 2).text() == run_id[:8]
    text = page.viewer.toPlainText()
    assert "Iris look" in text and "metrics.jsonl" in text and "../experiments/" in text
    assert page.create() != path  # a second notebook gets its own name
    lab = ctx.jupyter
    lab.prepare()
    assert f"--ServerApp.root_dir={ctx.paths.notebooks}" in lab.command(8888)
    assert lab.token not in mask_text(f"token={lab.token}")
    lab.port, lab.state = 8888, "running"
    assert lab.link_for(path) == f"http://127.0.0.1:8888/lab/tree/iris-look.ipynb?token={lab.token}"
    lab.state = "stopped"


def test_colab_page_writes_notebooks_and_reports_bad_imports(ctx, qtbot, tmp_path) -> None:
    from app.ui.pages.colab_page import ColabPage
    from labs.computer_vision.presets import CV_PRESETS

    page = ColabPage(ctx)
    qtbot.addWidget(page)
    assert page.source.count() == 0 and not page.export_button.isEnabled()
    ctx.experiments.open_in_builder(CV_PRESETS[0].make())
    assert page.source.currentData() == "draft" and page.export_button.isEnabled()
    path = page.export()
    assert path and path.suffix == ".ipynb" and page.table.rowCount() == 1
    notebook = json.loads(path.read_text())
    assert notebook["nbformat"] == 4 and any("mnist" in "".join(c["source"]) for c in notebook["cells"])
    bad = tmp_path / "bad.zip"
    bad.write_bytes(b"not a zip")
    with qtbot.waitSignal(ctx.colab.imported, timeout=30_000):
        page.start_import(bad)
    assert page.import_text.text().startswith("Not imported") and not page.show_button.isEnabled()
