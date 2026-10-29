"""Remote targets: machines an experiment can run on, kept in the workspace's
``configs/remotes.yaml``. A target names a host, a user, a port, the Python to run there, the
folder the lab works in, and optionally an SSH key *file*; no password, token or key content is
ever stored (authentication is SSH's own: keys, the agent, ``~/.ssh/config``)."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import yaml

__all__ = ["REMOTES_FILE", "Target", "TargetError", "load_targets", "save_targets"]

REMOTES_FILE = "remotes.yaml"
_NAME = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")
_SECRET = re.compile(r"pass|token|secret|key_content|private_key$", re.IGNORECASE)


class TargetError(ValueError):
    """A remote target that cannot be used, in words."""


@dataclass(frozen=True)
class Target:
    name: str
    host: str = ""
    user: str = ""
    port: int = 22
    #: The interpreter on the remote machine (with PyTorch etc. installed).
    python: str = "python3"
    #: Where the lab keeps code, runs and datasets there.
    folder: str = "~/tedo-lab-remote"
    identity_file: str = ""
    #: ``ssh`` (OpenSSH's ``ssh``/``scp``), or ``local`` (a folder on this machine; for tests).
    transport: Literal["ssh", "local"] = "ssh"

    def __post_init__(self) -> None:
        if not _NAME.match(self.name):
            raise TargetError(f"{self.name!r}: use letters, digits, dots, dashes and underscores")
        if self.transport not in ("ssh", "local"):
            raise TargetError(f"{self.name}: transport is ssh or local")
        if self.transport == "ssh" and not self.host:
            raise TargetError(f"{self.name}: an SSH target needs a host")
        if not 0 < int(self.port) < 65536:
            raise TargetError(f"{self.name}: port {self.port} is not a port")

    @property
    def address(self) -> str:
        return f"{self.user}@{self.host}" if self.user else self.host

    def describe(self) -> str:
        if self.transport == "local":
            return f"local folder {self.folder}"
        return f"{self.address}:{self.port} · {self.python} · {self.folder}"


def load_targets(configs: Path) -> dict[str, Target]:
    path = Path(configs) / REMOTES_FILE
    if not path.is_file():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    out = {}
    for name, entry in (data.get("targets") or {}).items():
        entry = dict(entry or {})
        secret = [key for key in entry if _SECRET.search(key)]
        if secret:
            raise TargetError(f"{name}: {', '.join(secret)} looks like a secret; remotes.yaml never holds "
                              "credentials (use an SSH key file or the SSH agent)")
        unknown = set(entry) - set(Target.__dataclass_fields__)
        if unknown:
            raise TargetError(f"{name}: unknown setting(s) {', '.join(sorted(unknown))}")
        out[str(name)] = Target(name=str(name), **entry)
    return out


def save_targets(configs: Path, targets: dict[str, Target]) -> Path:
    path = Path(configs) / REMOTES_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    body = {name: {k: v for k, v in asdict(t).items() if k != "name"} for name, t in sorted(targets.items())}
    path.write_text("# Machines experiments can run on. Never put passwords or keys here.\n"
                    + yaml.safe_dump({"targets": body}, sort_keys=False), encoding="utf-8")
    return path
