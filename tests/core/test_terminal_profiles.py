"""Terminal profiles: what is installed is offered, what is not says why; Docker commands; the
environment terminals get."""

from __future__ import annotations

import os
import stat
import sys
from pathlib import Path

import pytest

from core.terminal.profiles import (
    default_profile,
    docker_command,
    docker_containers,
    docker_images,
    find_profile,
    profiles,
    terminal_environment,
)

EXE = ".exe" if sys.platform == "win32" else ""


def fake_programs(folder: Path, *names: str) -> str:
    folder.mkdir(parents=True, exist_ok=True)
    for name in names:
        path = folder / f"{name}{EXE}"
        path.write_text("#!/bin/sh\nexit 0\n")
        path.chmod(path.stat().st_mode | stat.S_IEXEC)
    return str(folder)


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX profile list")
def test_posix_profiles_offer_what_is_installed(tmp_path) -> None:
    path = fake_programs(tmp_path / "bin", "bash", "zsh", "vim", "docker")
    found = profiles("auto", "/py/bin/python", path=path, windows=False)
    by_id = {p.id: p for p in found}
    assert by_id["bash"].available and by_id["zsh"].available and by_id["vim"].available
    assert not by_id["nvim"].available and "neovim.io" in by_id["nvim"].missing
    assert by_id["docker"].needs_choice and not by_id["docker"].shell
    assert by_id["python"].argv == ("/py/bin/python", "-q")
    assert default_profile(found).id == "bash" and find_profile(found, "vim").kind == "editor"
    nothing = profiles("auto", "", path=str(tmp_path / "empty"), windows=False)
    assert not find_profile(nothing, "python").available
    assert "Experiment Python" in find_profile(nothing, "python").missing
    assert find_profile(nothing, "sh").available  # a shell is always there to fall back on


def test_windows_profiles_and_the_configured_shell(tmp_path) -> None:
    path = fake_programs(tmp_path / "bin", "pwsh", "cmd")
    found = profiles("auto", "", path=path, windows=True)
    ids = [p.id for p in found]
    assert ids[:2] == ["powershell", "cmd"] and default_profile(found).id == "powershell"
    assert find_profile(found, "powershell").argv[1:] == ("-NoLogo",)
    configured = profiles(str(tmp_path / "bin" / f"pwsh{EXE}"), "", path=path, windows=True)
    assert configured[0].id == "configured" and configured[0].available
    missing = profiles("no-such-shell", "", path=path, windows=True)
    assert not missing[0].available and "terminal_shell" in missing[0].missing


def test_docker_commands() -> None:
    assert docker_command("docker", container="abc", shell="bash") == ["docker", "exec", "-it", "abc", "bash"]
    run = docker_command("docker", image="python:3.11-slim", mount=Path("/ws"))
    volume = f"{Path('/ws')}:/workspace"  # the folder as this system writes it (Docker Desktop takes C:\\…)
    assert run == ["docker", "run", "--rm", "-it", "-v", volume, "-w", "/workspace", "python:3.11-slim", "sh"]
    with pytest.raises(ValueError):
        docker_command("docker")


@pytest.mark.skipif(sys.platform == "win32", reason="fakes docker with a shell script")
def test_docker_lists_containers_and_reports_a_missing_daemon(tmp_path) -> None:
    docker = tmp_path / "docker"
    docker.write_text('#!/bin/sh\nif [ "$1" = ps ]; then printf "c1\\tweb\\tnginx:1\\tUp 2 minutes\\n"; '
                      'else printf "python:3.11\\n<none>:<none>\\n"; fi\n')
    docker.chmod(0o755)
    containers = docker_containers(str(docker))
    assert containers[0].name == "web" and containers[0].status == "Up 2 minutes"
    assert docker_images(str(docker)) == ["python:3.11"]
    broken = tmp_path / "broken"
    broken.write_text("#!/bin/sh\necho 'Cannot connect to the Docker daemon' >&2\nexit 1\n")
    broken.chmod(0o755)
    with pytest.raises(RuntimeError, match="Cannot connect"):
        docker_containers(str(broken))


def test_terminal_environment(tmp_path) -> None:
    base = {"PATH": "/usr/bin", "QT_QPA_PLATFORM": "offscreen", "LANG": "C"}
    env = terminal_environment(tmp_path, str(tmp_path / "venv" / "bin" / "python"), base)
    assert env["TERM"] == "xterm-256color" and env["COLORTERM"] == "truecolor"
    assert env["TEDO_LAB_WORKSPACE"] == str(tmp_path) and "QT_QPA_PLATFORM" not in env
    assert env["PATH"].split(os.pathsep)[0] == str(tmp_path / "venv" / "bin")
    if sys.platform != "win32":
        assert env["LC_CTYPE"] == "C.UTF-8"


@pytest.mark.skipif(sys.platform != "linux", reason="Linux package managers")
def test_missing_editors_offer_an_install_command(tmp_path) -> None:
    path = fake_programs(tmp_path / "bin", "bash", "apt-get", "sudo")
    nvim = find_profile(profiles("auto", "", path=path, windows=False), "nvim")
    assert nvim.install[-3:] == ("install", "-y", "neovim") and "apt-get" in nvim.install
    winget = fake_programs(tmp_path / "w", "winget")
    windows = find_profile(profiles("auto", "", path=winget, windows=True), "vim")
    assert windows.install == ("winget", "install", "-e", "--id", "vim.vim")
    nothing = profiles("auto", "", path=str(tmp_path / "none"), windows=False)
    assert find_profile(nothing, "nvim").install == ()
