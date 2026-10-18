"""What a terminal can run: shells (bash, zsh, PowerShell, Command Prompt, WSL), editors (vim,
Neovim), the experiment Python, and Docker (a shell in a running container, or a new one from
an image). Each profile says whether it is installed here and, when it is not, why.

Also the environment every terminal gets: ``TERM=xterm-256color``, the workspace as
``TEDO_LAB_WORKSPACE`` and the experiment Python first on ``PATH``.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

__all__ = ["DockerContainer", "Profile", "default_profile", "docker_command", "docker_containers",
           "docker_images", "find_profile", "profiles", "terminal_environment"]


@dataclass(frozen=True)
class Profile:
    id: str
    title: str
    argv: tuple[str, ...] = ()
    icon: str = "terminal"
    #: Why it cannot run here ("" = it can).
    missing: str = ""
    #: A shell keeps running between commands; editors and Docker are programs.
    shell: bool = True
    #: Needs more input before it starts (Docker: which container or image).
    needs_choice: bool = False
    kind: str = field(default="shell")
    #: A command that installs the program (run in a terminal, so the person sees it and can
    #: type a password); empty when there is no package manager to use.
    install: tuple[str, ...] = ()

    @property
    def available(self) -> bool:
        return not self.missing


def _which(*names: str, path: str | None = None) -> str | None:
    for name in names:
        found = shutil.which(name, path=path)
        if found:
            return found
    return None


def profiles(terminal_shell: str = "auto", python: str = "", *, path: str | None = None,
             windows: bool | None = None) -> list[Profile]:
    """Every profile, the default shell first; missing programs are listed with the reason."""
    windows = sys.platform == "win32" if windows is None else windows
    found: list[Profile] = []
    custom = terminal_shell.strip()
    if custom and custom != "auto":
        program = _which(custom, path=path) or (custom if Path(custom).is_file() else None)
        found.append(Profile("configured", f"{Path(custom).name} (settings)", (program or custom,),
                             missing="" if program else f"{custom} (terminal_shell) was not found"))
    if windows:
        pwsh = _which("pwsh", "powershell", path=path)
        found.append(Profile("powershell", "PowerShell", (pwsh, "-NoLogo") if pwsh else (),
                             missing="" if pwsh else "PowerShell was not found"))
        cmd = _which("cmd", path=path) or os.environ.get("COMSPEC", "")
        found.append(Profile("cmd", "Command Prompt", (cmd,) if cmd else (),
                             missing="" if cmd else "cmd.exe was not found"))
        git_bash = _git_bash()
        found.append(Profile("bash", "Git Bash" if git_bash else "Bash", (git_bash, "--login", "-i")
                             if git_bash else (), missing="" if git_bash else
                             "Git Bash was not found (install Git for Windows)"))
        wsl = _which("wsl", path=path)
        found.append(Profile("wsl", "WSL", (wsl,) if wsl else (), icon="server",
                             missing="" if wsl else "WSL is not installed"))
    else:
        login = os.environ.get("SHELL", "")
        for name, title in (("bash", "Bash"), ("zsh", "Zsh"), ("fish", "Fish")):
            program = _which(name, path=path)
            if program or name == "bash":
                found.append(Profile(name, title, (program,) if program else (),
                                     missing="" if program else f"{name} was not found"))
        if not any(p.available for p in found):
            sh = _which("sh", path=path) or "/bin/sh"
            found.append(Profile("sh", "sh", (sh,)))
        if login and Path(login).name not in {p.id for p in found}:
            found.append(Profile("login", Path(login).name, (login,)))
        pwsh = _which("pwsh", path=path)
        if pwsh:
            found.append(Profile("powershell", "PowerShell", (pwsh, "-NoLogo")))
    found.append(Profile("python", "Python (experiment)", (python, "-q") if python else (),
                         icon="code", missing="" if python else
                         "No experiment Python yet: Settings → Experiment Python"))
    for name, title, home in (("vim", "Vim", "https://www.vim.org"),
                              ("nvim", "Neovim", "https://neovim.io")):
        program = _which(name, path=path)
        found.append(Profile(name, title, (program,) if program else (), icon="file-code",
                             shell=False, kind="editor",
                             missing="" if program else f"{title} is not installed ({home})",
                             install=() if program else _installer(name, windows, path)))
    docker = _which("docker", path=path)
    found.append(Profile("docker", "Docker…", (docker,) if docker else (), icon="container",
                         shell=False, needs_choice=True, kind="docker",
                         missing="" if docker else "Docker is not installed (https://docs.docker.com)"))
    return found


#: Package names per package manager: (winget id, brew, apt/dnf/pacman).
_PACKAGES = {"vim": ("vim.vim", "vim", "vim"), "nvim": ("Neovim.Neovim", "neovim", "neovim")}


def _installer(name: str, windows: bool, path: str | None) -> tuple[str, ...]:
    winget_id, brew, linux = _PACKAGES[name]
    if windows:
        return ("winget", "install", "-e", "--id", winget_id) if _which("winget", path=path) else ()
    if sys.platform == "darwin":
        return ("brew", "install", brew) if _which("brew", path=path) else ()
    for manager, args in (("apt-get", ("install", "-y")), ("dnf", ("install", "-y")),
                          ("pacman", ("-S", "--noconfirm"))):
        if _which(manager, path=path):
            sudo = () if hasattr(os, "geteuid") and os.geteuid() == 0 else ("sudo",)
            return (*sudo, manager, *args, linux)
    return ()


def _git_bash() -> str | None:
    """Git for Windows' bash (not System32\\bash.exe, which starts WSL instead)."""
    for root in (os.environ.get("ProgramFiles", r"C:\Program Files"),
                 os.environ.get("LOCALAPPDATA", "") + r"\Programs"):
        candidate = Path(root) / "Git" / "bin" / "bash.exe"
        if candidate.is_file():
            return str(candidate)
    git = shutil.which("git")
    if git:
        candidate = Path(git).resolve().parents[1] / "bin" / "bash.exe"
        if candidate.is_file():
            return str(candidate)
    return None


def default_profile(found: list[Profile]) -> Profile | None:
    """The first available shell."""
    return next((p for p in found if p.available and p.shell), None)


def find_profile(found: list[Profile], profile_id: str) -> Profile | None:
    return next((p for p in found if p.id == profile_id), None)


def terminal_environment(workspace: Path, python: str, base: dict[str, str] | None = None) -> dict[str, str]:
    """The environment of a terminal's program."""
    env = dict(os.environ if base is None else base)
    env.update(TERM="xterm-256color", COLORTERM="truecolor", TERM_PROGRAM="tedo-lab",
               TEDO_LAB_WORKSPACE=str(workspace))
    for name in ("QT_QPA_PLATFORM", "PYTHONHOME"):  # the app's own settings, not the shell's
        env.pop(name, None)
    if python:
        env["PATH"] = str(Path(python).parent) + os.pathsep + env.get("PATH", "")
    if sys.platform != "win32":
        locale = env.get("LC_ALL") or env.get("LC_CTYPE") or env.get("LANG") or ""
        if "utf" not in locale.lower().replace("-", ""):
            env["LC_CTYPE"] = "C.UTF-8"  # the terminal speaks UTF-8
    return env


# Docker -------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class DockerContainer:
    id: str
    name: str
    image: str
    status: str


def _docker(args: list[str], docker: str = "docker") -> str:
    """Run a docker command that prints a list (blocking: call from a task)."""
    try:
        result = subprocess.run([docker, *args], capture_output=True, text=True, timeout=20,
                                stdin=subprocess.DEVNULL)
    except FileNotFoundError:
        raise RuntimeError("Docker is not installed") from None
    except subprocess.TimeoutExpired:
        raise RuntimeError("Docker did not answer within 20 seconds") from None
    if result.returncode != 0:
        message = (result.stderr or result.stdout).strip().splitlines()
        raise RuntimeError(message[-1] if message else f"docker {args[0]} failed")
    return result.stdout


def docker_containers(docker: str = "docker") -> list[DockerContainer]:
    out = _docker(["ps", "--format", "{{.ID}}\t{{.Names}}\t{{.Image}}\t{{.Status}}"], docker)
    return [DockerContainer(*(line.split("\t") + ["", "", ""])[:4]) for line in out.splitlines() if line]


def docker_images(docker: str = "docker") -> list[str]:
    out = _docker(["images", "--format", "{{.Repository}}:{{.Tag}}"], docker)
    return [line for line in out.splitlines() if line and "<none>" not in line]


def docker_command(docker: str, *, container: str = "", image: str = "", shell: str = "sh",
                   mount: Path | None = None) -> list[str]:
    """``docker exec -it <container> <shell>`` or ``docker run --rm -it [-v workspace] <image> <shell>``."""
    if container:
        return [docker, "exec", "-it", container, shell]
    if not image:
        raise ValueError("Choose a running container or an image")
    command = [docker, "run", "--rm", "-it"]
    if mount is not None:
        command += ["-v", f"{mount}:/workspace", "-w", "/workspace"]
    return [*command, image, shell]
