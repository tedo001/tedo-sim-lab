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


def test_runs_never_see_the_persons_credentials(ctx, qtbot, tmp_path, monkeypatch) -> None:
    """The command every run gets, and what a process started by the job queue really sees."""
    import json
    import sys

    from app.services.context import _command_for_run
    from app.services.jobs import JobQueue, WorkerCommand
    from core.remote.targets import Target, save_targets

    monkeypatch.setenv("TEDO_TEST_API_TOKEN", "must-not-leak")
    monkeypatch.setenv("WANDB_API_KEY", "passed-on-request")
    monkeypatch.setenv("SSH_AUTH_SOCK", str(tmp_path / "agent"))
    monkeypatch.setenv("PYTHONPATH", str(tmp_path / "host-site"))
    run = tmp_path / "run"
    run.mkdir()
    command = _command_for_run(run, ctx.config, ctx.catalog, ctx.paths)
    assert "TEDO_TEST_API_TOKEN" not in command.env and "WANDB_API_KEY" not in command.env
    assert "SSH_AUTH_SOCK" not in command.env and command.env["PYTHONPATH"] == str(ctx.paths.code_root)
    ctx.settings.update(pass_environment=["WANDB_API_KEY"])
    passed = _command_for_run(run, ctx.config, ctx.catalog, ctx.paths)
    assert passed.env["WANDB_API_KEY"] == "passed-on-request"

    from core.experiment_engine.spec import dump_spec
    from labs.classical_ml.presets import CLASSICAL_PRESETS

    box = Target("box", python=sys.executable, folder=str(tmp_path / "remote"), transport="local")
    save_targets(ctx.paths.configs, {"box": box})
    spec = CLASSICAL_PRESETS[0].make()
    dump_spec(spec.model_copy(update={"runtime": spec.runtime.model_copy(update={"target": "box"})}),
              run / "experiment.yaml")
    dispatch = _command_for_run(run, ctx.config, ctx.catalog, ctx.paths)
    assert "core.remote.dispatch" in dispatch.arguments
    assert dispatch.env["SSH_AUTH_SOCK"] == str(tmp_path / "agent")
    assert "TEDO_TEST_API_TOKEN" not in dispatch.env  # the agent's socket, nothing else of the person's

    seen = tmp_path / "seen.json"
    script = f"import json, os; json.dump(dict(os.environ), open({str(seen)!r}, 'w'))"
    queue = JobQueue(lambda _run: WorkerCommand(sys.executable, ["-c", script], tmp_path, {"EXTRA": "1"}))
    job = queue.submit_run(run)
    qtbot.waitUntil(lambda: queue.job(job).status in ("completed", "failed"), timeout=60_000)
    env = json.loads(seen.read_text(encoding="utf-8"))
    assert "TEDO_TEST_API_TOKEN" not in env and "SSH_AUTH_SOCK" not in env and env["EXTRA"] == "1"
    assert "PATH" in env and "host-site" not in env.get("PYTHONPATH", "")
    queue.shutdown()


def test_a_project_with_its_own_virtual_environment_runs_in_it(tmp_path) -> None:
    from app.services.project import project_python
    from core.common.experiment_env import env_python

    assert project_python(tmp_path, "/experiment/python") == "/experiment/python"
    venv = tmp_path / ".venv"
    python = env_python(venv)
    python.parent.mkdir(parents=True)
    python.write_text("")
    assert project_python(tmp_path, "/experiment/python") == "/experiment/python"  # not a venv yet
    (venv / "pyvenv.cfg").write_text("home = /usr/bin\n")
    assert project_python(tmp_path, "/experiment/python") == str(python)
