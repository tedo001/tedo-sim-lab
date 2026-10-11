"""The experiment Python: requirements in step with pyproject.toml, the setup plan, running its
steps (fallback, failure, cancel), describing a Python, and the installed app's default."""

from __future__ import annotations

import re
import sys
import tomllib

import pytest

from app.services.experiment_setup import run_steps
from core.common import AppConfig, experiment_python
from core.common.cancel import Cancelled, CancelToken
from core.common.experiment_env import (
    Step,
    describe_python,
    env_python,
    requirements_file,
    setup_steps,
    torch_build,
)
from core.common.paths import CODE_ROOT


def _names(lines: list[str]) -> set[str]:
    return {re.split(r"[<>=!~ \[]", line)[0].lower() for line in lines}


def test_requirements_follow_pyproject() -> None:
    project = tomllib.loads((CODE_ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    lines = [line for line in requirements_file().read_text(encoding="utf-8").splitlines()
             if line and not line.startswith("#")]
    window_only = {"pyside6-essentials", "pyqtgraph", "torch", "torchvision"}
    assert _names(lines) == _names(project["dependencies"]) - window_only


def test_setup_plan(tmp_path) -> None:
    steps = setup_steps(tmp_path / "env", ["python3"], "cuda")
    assert steps[0].command[:3] == ["python3", "-m", "venv"]
    torch = steps[2]
    assert "cu126" in torch.command[-1] and torch.fallback and "--index-url" not in torch.fallback
    assert steps[-1].command[-2:] == ["-r", str(requirements_file())]
    assert torch_build(cpu_only=True, system="Windows") == "cpu"
    assert torch_build(gpu=True, system="Linux") == "cuda" and torch_build(system="Darwin") == "default"
    (tmp_path / "env").mkdir()
    made = env_python(tmp_path / "env")
    made.parent.mkdir(parents=True)
    made.touch()
    assert "venv" not in setup_steps(tmp_path / "env", ["python3"], "cpu")[0].command  # already made


def test_steps_fall_back_fail_and_cancel() -> None:
    py = sys.executable
    seen: list[str] = []
    run_steps([Step("first", [py, "-c", "import sys; print('a'); sys.exit(1)"], [py, "-c", "print('b')"])],
              CancelToken(), lambda _f, message: seen.append(message))
    assert any(message.endswith(": b") for message in seen) and seen[-1] == "Done"
    with pytest.raises(RuntimeError, match="second failed"):
        run_steps([Step("second", [py, "-c", "import sys; print('oops'); sys.exit(3)"])], CancelToken(),
                  lambda *_: None)
    token = CancelToken()
    token.cancel()
    with pytest.raises(Cancelled):
        run_steps([Step("long", [py, "-c", "print('x'); import time; time.sleep(30)"])], token,
                  lambda *_: None)


def test_describe_python() -> None:
    facts = describe_python(sys.executable)
    assert facts.ok and facts.version.startswith(f"{sys.version_info[0]}.{sys.version_info[1]}")
    assert "yaml" in facts.packages
    assert describe_python("/no/such/python").error and not describe_python("").ok


def test_installed_app_uses_the_made_environment(monkeypatch, tmp_path) -> None:
    assert experiment_python(AppConfig()) == sys.executable  # a source checkout
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr("core.common.experiment_env.default_env_dir", lambda: tmp_path / "env")
    assert experiment_python(AppConfig()) == ""  # installed, nothing made yet
    made = env_python(tmp_path / "env")
    made.parent.mkdir(parents=True)
    made.touch()
    assert experiment_python(AppConfig()) == str(made)
    assert experiment_python(AppConfig(python_executable="/opt/py")) == "/opt/py"


def test_a_run_without_an_experiment_python_fails_with_the_way_out(ctx, qtbot) -> None:
    from app.services.jobs import worker_command
    from labs.classical_ml.presets import CLASSICAL_PRESETS
    from tests.ui.test_experiment_service import wait_for

    ctx.jobs._command_factory = lambda run_dir: worker_command(run_dir, python="", code_root=CODE_ROOT)
    run_id = ctx.experiments.launch(CLASSICAL_PRESETS[0].make())
    view = wait_for(qtbot, ctx, run_id, ("failed", "completed"), timeout=10_000)
    assert view.status == "failed" and "Experiment Python" in (view.error or "")
