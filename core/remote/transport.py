"""How the dispatcher reaches a target: OpenSSH's ``ssh`` and ``scp`` command-line tools (BSD
licence, installed with Windows 10+, macOS and Linux; never a Python dependency), or, for tests and
a second local disk, a plain folder on this machine. Commands are argument lists, quoted for the
remote shell; ``BatchMode`` stops SSH from ever asking for a password in a hidden process."""

from __future__ import annotations

import shlex
import shutil
import subprocess
from pathlib import Path, PurePosixPath

from .targets import Target

__all__ = ["LocalTransport", "SshTransport", "Transport", "transport_for"]


class Transport:
    target: Target

    def path(self, *parts: str) -> str:
        raise NotImplementedError

    def command(self, argv: list[str]) -> list[str]:
        raise NotImplementedError

    def put(self, local: Path, remote: str) -> None:
        raise NotImplementedError

    def get(self, remote: str, local: Path) -> None:
        raise NotImplementedError

    def run(self, argv: list[str], timeout: float = 600) -> subprocess.CompletedProcess[str]:
        return subprocess.run(self.command(argv), capture_output=True, text=True, timeout=timeout,
                              stdin=subprocess.DEVNULL, encoding="utf-8", errors="replace")

    def popen(self, argv: list[str]) -> subprocess.Popen[str]:
        return subprocess.Popen(self.command(argv), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace",
                                bufsize=1)


class SshTransport(Transport):
    def __init__(self, target: Target) -> None:
        self.target = target
        folder = target.folder
        # Remote commands start in the home folder, so "~/x" is the relative path "x" (quoting would
        # stop the shell from expanding "~").
        self.base = PurePosixPath(folder[2:] if folder.startswith("~/") else folder)

    def _options(self, port_flag: str) -> list[str]:
        options = [port_flag, str(self.target.port), "-o", "BatchMode=yes", "-o", "ConnectTimeout=20"]
        if self.target.identity_file:
            options += ["-i", str(Path(self.target.identity_file).expanduser())]
        return options

    def path(self, *parts: str) -> str:
        return str(self.base.joinpath(*parts))

    def command(self, argv: list[str]) -> list[str]:
        return ["ssh", *self._options("-p"), self.target.address, shlex.join(argv)]

    def put(self, local: Path, remote: str) -> None:
        self._copy([str(local), f"{self.target.address}:{remote}"])

    def get(self, remote: str, local: Path) -> None:
        self._copy([f"{self.target.address}:{remote}", str(local)])

    def _copy(self, paths: list[str]) -> None:
        done = subprocess.run(["scp", "-q", *self._options("-P"), *paths], capture_output=True, text=True,
                              stdin=subprocess.DEVNULL, timeout=3600)
        if done.returncode != 0:
            raise OSError(f"scp failed: {done.stderr.strip()[-300:] or done.returncode}")


class LocalTransport(Transport):
    """A "remote" that is a folder here: the same steps, without a network."""

    def __init__(self, target: Target) -> None:
        self.target = target
        self.base = Path(target.folder).expanduser().resolve()

    def path(self, *parts: str) -> str:
        return str(self.base.joinpath(*parts))

    def command(self, argv: list[str]) -> list[str]:
        return list(argv)

    def put(self, local: Path, remote: str) -> None:
        Path(remote).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(local, remote)

    def get(self, remote: str, local: Path) -> None:
        shutil.copy2(remote, local)


def transport_for(target: Target) -> Transport:
    return LocalTransport(target) if target.transport == "local" else SshTransport(target)
