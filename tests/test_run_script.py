"""run.py, the one-file launcher: which PyTorch build it picks and where its Python lives."""

from __future__ import annotations

from pathlib import Path

import run


def test_torch_build_follows_the_machine() -> None:
    assert run.torch_build(False, system="Windows", gpu=True) == "cuda"
    assert run.torch_build(False, system="Windows", gpu=False) == "cpu"
    assert run.torch_build(True, system="Linux", gpu=True) == "cpu"  # --cpu wins
    assert run.torch_build(False, system="Darwin", gpu=False) == "default"  # Apple GPU via PyPI wheels
    assert set(run.TORCH_INDEX) == {"cuda", "cpu"}


def test_environment_layout_and_fingerprint(tmp_path) -> None:
    python = run.venv_python(tmp_path)
    assert python.parent.name in ("Scripts", "bin") and python.parent.parent == tmp_path
    first, again = run.fingerprint("cpu"), run.fingerprint("cpu")
    assert first == again and first["torch"] == "cpu" and len(first["pyproject"]) == 64
    assert run.fingerprint("cuda") != first  # a different build reinstalls
    assert run.ROOT == Path(__file__).resolve().parents[1]


def test_own_options_are_not_passed_to_the_app(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(run, "make_venv", lambda: calls.append("venv"))
    monkeypatch.setattr(run, "install", lambda build: calls.append(f"install {build}"))
    monkeypatch.setattr(run, "venv_python", lambda venv=run.VENV: Path(run.__file__))  # "exists"
    monkeypatch.setattr(run, "MARKER", Path("/nonexistent/marker.json"))
    monkeypatch.setattr(run, "has_nvidia_gpu", lambda: False)
    monkeypatch.setattr(run.Path, "write_text", lambda self, *a, **k: 0)
    started = []
    monkeypatch.setattr(run, "run", lambda command, **kwargs: started.append(command) or 0)
    assert run.main(["--setup-only", "--cpu"]) == 0
    assert calls == ["install cpu"] and not started
    assert run.main(["--cpu", "--workspace", "D:/lab"]) == 0
    assert started[-1][-3:] == ["app.main", "--workspace", "D:/lab"]
