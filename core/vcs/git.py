"""Git for the Source Control page, through the ``git`` command line (blocking: call from a task).

Status, diffs, staging, commits, history, branches, remotes, fetch/pull/push, init and clone.
A GitHub token, when there is one, reaches git only through the environment (``GIT_CONFIG_*``,
as an HTTP header for github.com), never on a command line or in a file; output and errors pass
through the secret masking. Git never waits for a password it cannot ask for: terminal prompts
are off and SSH runs in batch mode, so such a command fails with a message instead of hanging
(the person can run it in a terminal to type a passphrase).
"""

from __future__ import annotations

import base64
import os
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from core.common.masking import mask_text

__all__ = ["Branch", "Commit", "FileChange", "Git", "GitError", "RepoStatus", "git_executable",
           "github_slug", "repository_name"]

NETWORK_TIMEOUT = 600
LOCAL_TIMEOUT = 60
_FIELD, _RECORD = "\x1f", "\x1e"


class GitError(RuntimeError):
    """A git command failed; the message is git's own (masked)."""


def git_executable() -> str | None:
    return shutil.which("git")


@dataclass(frozen=True)
class FileChange:
    path: str
    index: str = " "      # staged state: M A D R C T, ? untracked, ' ' none
    worktree: str = " "   # unstaged state
    original: str = ""    # a rename's old path

    @property
    def untracked(self) -> bool:
        return self.index == "?"

    @property
    def staged(self) -> bool:
        return self.index not in (" ", "?", "!")

    @property
    def unstaged(self) -> bool:
        return self.worktree not in (" ", "!") or self.untracked

    @property
    def conflicted(self) -> bool:
        return self.index == "U" or self.worktree == "U"


@dataclass(frozen=True)
class RepoStatus:
    root: Path
    branch: str = ""
    head: str = ""
    upstream: str = ""
    ahead: int = 0
    behind: int = 0
    changes: list[FileChange] = field(default_factory=list)

    @property
    def detached(self) -> bool:
        return not self.branch

    @property
    def staged(self) -> list[FileChange]:
        return [c for c in self.changes if c.staged]

    @property
    def unstaged(self) -> list[FileChange]:
        return [c for c in self.changes if c.unstaged]


@dataclass(frozen=True)
class Commit:
    sha: str
    short: str
    author: str
    email: str
    date: str
    subject: str
    refs: str = ""


@dataclass(frozen=True)
class Branch:
    name: str
    current: bool = False
    remote: bool = False
    upstream: str = ""
    sha: str = ""


def _xy(code: str) -> tuple[str, str]:
    """Porcelain v2's XY field: ``.`` means unchanged."""
    return code[0].replace(".", " "), code[1].replace(".", " ")


def github_slug(url: str) -> tuple[str, str] | None:
    """(owner, repository) of a github.com remote URL (HTTPS or SSH), else None."""
    match = re.match(r"(?:https?://(?:[^@/]+@)?github\.com/|git@github\.com:|ssh://git@github\.com/)"
                     r"([^/]+)/([^/]+?)(?:\.git)?/?$", url.strip())
    return (match.group(1), match.group(2)) if match else None


def repository_name(url: str) -> str:
    """The folder ``git clone`` would make for ``url``: an HTTPS or SSH URL (``host:owner/repo``)
    or a local path with either slash, without ``.git``."""
    name = re.split(r"[/\\:]", url.strip().rstrip("/\\"))[-1].removesuffix(".git")
    return name if name not in ("", ".", "..") else "repository"


class Git:
    def __init__(self, root: Path, *, token: str = "", git: str | None = None) -> None:
        self.root = Path(root)
        self.git = git or git_executable() or "git"
        self._token = token

    # Running -------------------------------------------------------------------------------------
    def environment(self) -> dict[str, str]:
        env = dict(os.environ)
        env["GIT_TERMINAL_PROMPT"] = "0"
        env.setdefault("GIT_SSH_COMMAND", "ssh -o BatchMode=yes")
        if self._token:
            header = base64.b64encode(f"x-access-token:{self._token}".encode()).decode()
            count = int(env.get("GIT_CONFIG_COUNT", "0") or 0)
            env["GIT_CONFIG_COUNT"] = str(count + 1)
            env[f"GIT_CONFIG_KEY_{count}"] = "http.https://github.com/.extraheader"
            env[f"GIT_CONFIG_VALUE_{count}"] = f"AUTHORIZATION: basic {header}"
        return env

    def run(self, *args: str, timeout: int = LOCAL_TIMEOUT, cwd: Path | None = None,
            check: bool = True) -> str:
        try:
            result = subprocess.run([self.git, *args], cwd=str(cwd or self.root), env=self.environment(),
                                    capture_output=True, timeout=timeout, stdin=subprocess.DEVNULL)
        except FileNotFoundError:
            raise GitError("git is not installed (https://git-scm.com)") from None
        except subprocess.TimeoutExpired:
            raise GitError(f"git {args[0]} did not finish within {timeout} seconds") from None
        out = result.stdout.decode("utf-8", errors="replace")
        if check and result.returncode != 0:
            message = (result.stderr.decode("utf-8", errors="replace") or out).strip()
            raise GitError(mask_text(message) or f"git {args[0]} failed ({result.returncode})")
        return out

    # Where -------------------------------------------------------------------------------------------
    @classmethod
    def find_root(cls, folder: Path, git: str | None = None) -> Path | None:
        """The repository ``folder`` is in, or None."""
        try:
            out = cls(folder, git=git).run("rev-parse", "--show-toplevel", cwd=folder)
        except (GitError, OSError):
            return None
        return Path(out.strip()) if out.strip() else None

    @classmethod
    def init(cls, folder: Path, branch: str = "main") -> Git:
        folder.mkdir(parents=True, exist_ok=True)
        repo = cls(folder)
        repo.run("init", "-b", branch)
        return repo

    @classmethod
    def clone(cls, url: str, destination: Path, *, token: str = "") -> Git:
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() and any(destination.iterdir()):
            raise GitError(f"{destination} already exists and is not empty")
        cls(destination.parent, token=token).run("clone", "--", url, str(destination),
                                                 timeout=NETWORK_TIMEOUT)
        return cls(destination, token=token)

    # State -------------------------------------------------------------------------------------------
    def status(self) -> RepoStatus:
        out = self.run("status", "--porcelain=v2", "--branch", "-z", "--untracked-files=all")
        branch = head = upstream = ""
        ahead = behind = 0
        changes: list[FileChange] = []
        records = out.split("\0")
        i = 0
        while i < len(records):
            record = records[i]
            i += 1
            if record.startswith("# branch.oid "):
                oid = record.split(" ", 2)[2]
                head = "" if oid == "(initial)" else oid[:8]
            elif record.startswith("# branch.head "):
                name = record.split(" ", 2)[2]
                branch = "" if name == "(detached)" else name
            elif record.startswith("# branch.upstream "):
                upstream = record.split(" ", 2)[2]
            elif record.startswith("# branch.ab "):
                plus, minus = record.split(" ")[2:4]
                ahead, behind = int(plus), abs(int(minus))
            elif record.startswith("1 "):
                parts = record.split(" ", 8)
                changes.append(FileChange(parts[8], *_xy(parts[1])))
            elif record.startswith("2 "):
                parts = record.split(" ", 9)
                original = records[i] if i < len(records) else ""
                i += 1
                changes.append(FileChange(parts[9], *_xy(parts[1]), original))
            elif record.startswith("u "):
                parts = record.split(" ", 10)
                changes.append(FileChange(parts[10], "U", "U"))
            elif record.startswith("? "):
                changes.append(FileChange(record[2:], "?", "?"))
        return RepoStatus(self.root, branch, head, upstream, ahead, behind, changes)

    def diff(self, change: FileChange, *, staged: bool = False) -> str:
        """The change as a unified diff (a new, untracked file shows as all added)."""
        if change.untracked:
            path = self.root / change.path
            try:
                lines = path.read_text(encoding="utf-8").splitlines()
            except (OSError, UnicodeDecodeError):
                return f"(binary or unreadable file: {change.path})"
            body = "\n".join(f"+{line}" for line in lines[:5000])
            return f"new file: {change.path}\n@@ -0,0 +1,{len(lines)} @@\n{body}"
        args = ["diff", "--no-color", "--no-ext-diff"] + (["--cached"] if staged else [])
        return self.run(*args, "--", change.path)

    def stage(self, paths: list[str]) -> None:
        if paths:
            self.run("add", "--all", "--", *paths)

    def unstage(self, paths: list[str]) -> None:
        if not paths:
            return
        if self.status().head:
            self.run("restore", "--staged", "--", *paths)
        else:  # nothing committed yet: there is no HEAD to restore from
            self.run("rm", "--cached", "--quiet", "-r", "--", *paths)

    def discard(self, paths: list[str]) -> None:
        """Throw away unstaged changes to tracked files (untracked files are left alone)."""
        if paths:
            self.run("restore", "--worktree", "--", *paths)

    def identity(self) -> tuple[str, str]:
        name = self.run("config", "user.name", check=False).strip()
        email = self.run("config", "user.email", check=False).strip()
        return name, email

    def set_identity(self, name: str, email: str, *, everywhere: bool = False) -> None:
        scope = "--global" if everywhere else "--local"
        self.run("config", scope, "user.name", name)
        self.run("config", scope, "user.email", email)

    def commit(self, message: str, *, amend: bool = False) -> str:
        if not message.strip() and not amend:
            raise GitError("Write a commit message first")
        name, email = self.identity()
        if not name or not email:
            raise GitError("Git does not know who you are yet: set your name and email "
                           "(Branches and remotes → Identity)")
        args = ["commit", "--quiet", "-F", "-"]
        if amend:
            args += ["--amend"] + (["--no-edit"] if not message.strip() else [])
        result = subprocess.run([self.git, *args], cwd=str(self.root), env=self.environment(),
                                input=message.encode("utf-8"), capture_output=True, timeout=LOCAL_TIMEOUT)
        if result.returncode != 0:
            text = (result.stderr or result.stdout).decode("utf-8", errors="replace").strip()
            raise GitError(mask_text(text) or "git commit failed")
        return self.run("rev-parse", "--short", "HEAD").strip()

    def log(self, limit: int = 200, path: str | None = None) -> list[Commit]:
        if not self.status().head:
            return []
        fmt = _FIELD.join(["%H", "%h", "%an", "%ae", "%aI", "%s", "%D"]) + _RECORD
        args = ["log", f"--max-count={limit}", f"--format={fmt}"] + (["--", path] if path else [])
        commits = []
        for record in self.run(*args).split(_RECORD):
            parts = record.strip("\n").split(_FIELD)
            if len(parts) == 7:
                commits.append(Commit(*parts))
        return commits

    def show(self, sha: str) -> str:
        return self.run("show", "--stat", "--patch", "--no-color", "--format=fuller", sha)

    # Branches and remotes --------------------------------------------------------------------------
    def branches(self) -> list[Branch]:
        fmt = _FIELD.join(["%(refname)", "%(HEAD)", "%(upstream:short)", "%(objectname:short)"])
        found = []
        for line in self.run("for-each-ref", f"--format={fmt}", "refs/heads", "refs/remotes").splitlines():
            ref, head, upstream, sha = (line.split(_FIELD) + ["", "", ""])[:4]
            if ref.endswith("/HEAD"):
                continue
            remote = ref.startswith("refs/remotes/")
            name = ref.removeprefix("refs/remotes/" if remote else "refs/heads/")
            found.append(Branch(name, head == "*", remote, upstream, sha))
        return found

    def create_branch(self, name: str, *, checkout: bool = True) -> None:
        self.run("check-ref-format", "--branch", name)
        self.run("switch", "-c", name) if checkout else self.run("branch", name)

    def checkout(self, name: str) -> None:
        local = {b.name for b in self.branches() if not b.remote}
        if name in local:
            self.run("switch", name)
        else:  # a remote branch: a local one that tracks it
            self.run("switch", "--track", name)

    def remotes(self) -> dict[str, str]:
        found: dict[str, str] = {}
        for line in self.run("remote", "-v").splitlines():
            parts = line.split()
            if len(parts) >= 2:
                found.setdefault(parts[0], parts[1])
        return found

    def add_remote(self, name: str, url: str) -> None:
        if name in self.remotes():
            self.run("remote", "set-url", name, url)
        else:
            self.run("remote", "add", name, url)

    def fetch(self) -> str:
        return self.run("fetch", "--all", "--prune", timeout=NETWORK_TIMEOUT)

    def pull(self) -> str:
        return self.run("pull", "--ff-only", timeout=NETWORK_TIMEOUT)

    def push(self, remote: str = "origin") -> str:
        status = self.status()
        if status.detached:
            raise GitError("Not on a branch: check out a branch to push")
        if status.upstream:
            return self.run("push", timeout=NETWORK_TIMEOUT)
        return self.run("push", "--set-upstream", remote, status.branch, timeout=NETWORK_TIMEOUT)
