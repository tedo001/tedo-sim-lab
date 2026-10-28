"""Running files that live inside packages: the Code page runs them from their source root, as a
module, with the lab's own code last on ``PYTHONPATH`` so it never hides a project's ``app``."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from app.services.project import project_environment, python_launch, run_command, server_hint
from core.terminal.debugger import pdb_command

CONFIG = "class Settings:\n    APP_NAME = 'CraftLink'\n\nsettings = Settings()\n"
MAIN = "from app.config import settings\n\nprint('name', settings.APP_NAME)\n"


def backend(tmp_path: Path, *, package: bool) -> tuple[Path, Path]:
    """A project shaped like a FastAPI service: project/craftlink/backend/app/main.py."""
    root = tmp_path / "Artisian"
    app = root / "craftlink" / "backend" / "app"
    app.mkdir(parents=True)
    (root / "craftlink" / "backend" / "requirements.txt").write_text("fastapi\n")
    if package:
        (app / "__init__.py").write_text("")
    (app / "config.py").write_text(CONFIG)
    (app / "main.py").write_text(MAIN)
    return root, app / "main.py"


def run(path: Path, root: Path, tmp_path: Path) -> subprocess.CompletedProcess:
    launch = python_launch(path, root)
    env = {**os.environ, **project_environment(root, tmp_path, sys.executable, dict(os.environ),
                                               launch.paths)}
    return subprocess.run(run_command(path, [], sys.executable, root), cwd=launch.folder, env=env,
                          capture_output=True, text=True, stdin=subprocess.DEVNULL, timeout=60)


@pytest.mark.parametrize("package", [True, False])
def test_a_file_in_a_package_runs_from_its_source_root(tmp_path, package) -> None:
    root, main = backend(tmp_path, package=package)
    launch = python_launch(main, root)
    assert launch.target == ("-m", "app.main")
    assert launch.folder == main.parent.parent.resolve() and launch.paths == (launch.folder,)
    done = run(main, root, tmp_path)
    assert done.returncode == 0, done.stderr
    assert "name CraftLink" in done.stdout  # the project's app, not the lab's own app package


def test_plain_scripts_still_run_by_path_from_the_project(tmp_path) -> None:
    (tmp_path / "tool.py").write_text("print('ok')\n")
    launch = python_launch(tmp_path / "tool.py", tmp_path)
    assert launch.target == (str((tmp_path / "tool.py").resolve()),) and launch.folder == tmp_path.resolve()
    assert run_command(tmp_path / "tool.py", ["-v"], "/py", tmp_path)[-1] == "-v"


def test_folder_names_that_are_not_modules_fall_back_to_the_path(tmp_path) -> None:
    code = tmp_path / "my-service" / "src" / "pkg"
    code.mkdir(parents=True)
    (code / "__init__.py").write_text("")
    (code.parent / "__init__.py").write_text("")
    (code / "run.py").write_text("print(1)\n")
    launch = python_launch(code / "run.py", tmp_path)
    assert launch.target == ("-m", "src.pkg.run") and launch.folder == (tmp_path / "my-service").resolve()
    (code.parent / "__init__.py").unlink()
    bad = tmp_path / "my-service" / "src" / "my-pkg"
    bad.mkdir()
    (bad / "__init__.py").write_text("")
    (bad / "run.py").write_text("print(1)\n")
    launch = python_launch(bad / "run.py", tmp_path)
    assert launch.target == (str((bad / "run.py").resolve()),) and launch.paths == (bad.parent.resolve(),)


def test_only_the_labs_own_packages_are_shared(tmp_path) -> None:
    from core.common.paths import CODE_ROOT

    env = project_environment(tmp_path, tmp_path, "", {"PYTHONPATH": "/extra"}, (tmp_path / "src",))
    parts = env["PYTHONPATH"].split(os.pathsep)
    assert parts[:3] == [str(tmp_path / "src"), str(tmp_path), "/extra"]
    assert str(CODE_ROOT) not in parts  # its `app` package would hide a project's own
    shared = Path(parts[-1])
    assert sorted(p.name for p in shared.iterdir()) == ["core", "labs"]
    code = ("import labs, core, labs.simulation.quantum as q; from core.common.cancel import CancelToken; "
            "print(q.__name__, labs.__doc__.split(':')[0])")
    done = subprocess.run([sys.executable, "-c", code], env={**os.environ, "PYTHONPATH": str(shared)},
                          cwd=tmp_path, capture_output=True, text=True, stdin=subprocess.DEVNULL, timeout=60)
    assert done.returncode == 0, done.stderr
    assert "labs.simulation.quantum Domain labs" in done.stdout
    mine = tmp_path / "service"
    (mine / "app").mkdir(parents=True)
    (mine / "app" / "config.py").write_text(CONFIG)  # a namespace package: no __init__.py
    done = subprocess.run([sys.executable, "-c", "from app.config import settings; print(settings.APP_NAME)"],
                          env={**os.environ, "PYTHONPATH": os.pathsep.join([str(mine), str(shared)])},
                          cwd=mine, capture_output=True, text=True, stdin=subprocess.DEVNULL, timeout=60)
    assert done.returncode == 0 and done.stdout.strip() == "CraftLink", done.stderr


def test_web_apps_that_start_no_server_get_a_hint(tmp_path) -> None:
    root, main = backend(tmp_path, package=True)
    main.write_text("from fastapi import FastAPI\n\napp = FastAPI(title='x')\n")
    hint = server_hint(main, python_launch(main, root))
    assert "uvicorn app.main:app" in hint
    main.write_text("app = FastAPI()\nif __name__ == '__main__':\n    import uvicorn\n    uvicorn.run(app)\n")
    assert server_hint(main, python_launch(main, root)) == ""
    assert server_hint(root / "missing.py", python_launch(main, root)) == ""


def test_debugging_a_module_uses_pdb_dash_m() -> None:
    assert pdb_command("py", Path("x/app/main.py"), ["a"], module="app.main") == ["py", "-m", "pdb", "-m",
                                                                                  "app.main", "a"]
    assert pdb_command("py", Path("t.py"))[-1] == "t.py"
