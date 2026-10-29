"""The Plugin Store's Environments card lists the packs and their state; nothing is installed until
the person asks."""

from __future__ import annotations

from app.ui.pages.plugin_envs import EnvironmentsCard
from core.common import packs


def test_environments_card(ctx, qtbot, tmp_path, monkeypatch) -> None:
    monkeypatch.setenv(packs.PACKS_ENV, str(tmp_path / "packs"))
    card = EnvironmentsCard(ctx)
    qtbot.addWidget(card)
    assert card.names == ["language", "explain", "onnx"] and card.table.rowCount() == 3
    assert card.selected == "language" and card.install_button.text() == "Install"
    assert not card.remove_button.isEnabled() and "transformers" in card.table.item(0, 1).text()
    assert "PyTorch is not installed twice" in card.note.text()
    assert card.remove() is False


def test_designed_here_run_on_a_target(ctx, qtbot, tmp_path) -> None:
    """Settings → Remote targets, the builder's Run on, and the run through the job queue."""
    import sys

    import pytest

    pytest.importorskip("sklearn")
    from app.ui.pages.builder import ExperimentBuilderPage
    from app.ui.pages.remote_targets_card import RemoteTargetsCard
    from labs.classical_ml.presets import CLASSICAL_PRESETS
    from tests.ui.test_experiment_service import wait_for

    card = RemoteTargetsCard(ctx)
    qtbot.addWidget(card)
    card.name.setText("box")
    card.python.setText(sys.executable)
    card.folder.setText(str(tmp_path / "remote"))
    card.transport.setCurrentIndex(card.transport.findData("local"))
    assert card.save() is not None and card.names == ["box"]
    card.table.selectRow(0)
    with qtbot.waitSignal(ctx.jobs.job_finished, timeout=60_000):
        card.test()
    assert "Reached it" in card.note.text()
    builder = ExperimentBuilderPage(ctx)
    qtbot.addWidget(builder)
    builder.load(CLASSICAL_PRESETS[0].make())
    builder.refresh_targets()
    builder.target.setCurrentIndex(builder.target.findData("box"))
    builder.refresh()
    assert builder.spec.runtime.target == "box" and "target: box" in builder.yaml.toPlainText()
    run_id = builder.run()
    assert run_id
    view = wait_for(qtbot, ctx, run_id, ("completed", "failed"), timeout=180_000)
    assert view.status == "completed", view.error
    assert (view.run_dir / "results.json").is_file()
    assert (tmp_path / "remote" / "runs" / run_id / "results.json").is_file()
    assert any("Running on box" in line for line in ctx.experiments.log_lines(run_id))
