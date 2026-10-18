"""Git through its command line (real repositories, a bare one as the remote) and GitHub
through its REST API (a local stand-in server)."""

from __future__ import annotations

import json
import shutil
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from core.vcs import Git, GitError, GitHub, GitHubError, github_slug

needs_git = pytest.mark.skipif(shutil.which("git") is None, reason="needs git")


@pytest.fixture(autouse=True)
def _no_personal_git_config(monkeypatch, tmp_path):
    """Tests never read or write the person's own git settings."""
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(tmp_path / "no-global-gitconfig"))
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")


def repo_with_identity(folder: Path) -> Git:
    repo = Git.init(folder)
    repo.set_identity("Tester", "tester@example.com")
    return repo


@needs_git
def test_status_stage_diff_commit_and_history(tmp_path) -> None:
    repo = repo_with_identity(tmp_path / "project")
    assert Git.find_root(tmp_path / "project") == (tmp_path / "project").resolve()
    assert Git.find_root(tmp_path) is None
    (repo.root / "a.py").write_text("x = 1\n")
    (repo.root / "data").mkdir()
    (repo.root / "data" / "b.txt").write_text("hello\n")
    status = repo.status()
    assert status.branch == "main" and status.head == "" and status.detached is False
    assert sorted(c.path for c in status.unstaged) == ["a.py", "data/b.txt"]
    assert all(c.untracked for c in status.changes) and repo.log() == []
    assert "+x = 1" in repo.diff(status.changes[0])
    repo.stage(["a.py"])
    staged = repo.status().staged
    assert [c.path for c in staged] == ["a.py"] and staged[0].index == "A"
    repo.unstage(["a.py"])  # before the first commit there is no HEAD to restore from
    assert repo.status().staged == []
    repo.stage(["a.py", "data/b.txt"])
    with pytest.raises(GitError, match="message"):
        repo.commit("   ")
    sha = repo.commit("First commit\n\nWith a body.")
    assert len(sha) >= 7 and repo.status().changes == []
    (repo.root / "a.py").write_text("x = 2\n")
    change = repo.status().changes[0]
    assert change.worktree == "M" and not change.staged
    assert "-x = 1" in repo.diff(change) and "+x = 2" in repo.diff(change)
    repo.stage(["a.py"])
    assert "+x = 2" in repo.diff(repo.status().changes[0], staged=True)
    repo.unstage(["a.py"])
    repo.discard(["a.py"])
    assert (repo.root / "a.py").read_text() == "x = 1\n"
    log = repo.log()
    assert [c.subject for c in log] == ["First commit"] and log[0].author == "Tester"
    assert "With a body." in repo.show(log[0].sha)


@needs_git
def test_commits_need_an_identity(tmp_path) -> None:
    repo = Git.init(tmp_path / "anonymous")
    (repo.root / "f").write_text("1")
    repo.stage(["f"])
    with pytest.raises(GitError, match="who you are"):
        repo.commit("no identity")


@needs_git
def test_renames_and_amend(tmp_path) -> None:
    repo = repo_with_identity(tmp_path / "project")
    (repo.root / "old.txt").write_text("same content\n" * 5)
    repo.stage(["old.txt"])
    repo.commit("add")
    (repo.root / "old.txt").rename(repo.root / "new.txt")
    repo.stage(["old.txt", "new.txt"])
    renamed = repo.status().staged[0]
    assert (renamed.index, renamed.path, renamed.original) == ("R", "new.txt", "old.txt")
    repo.commit("rename")
    repo.commit("rename, better worded", amend=True)
    assert [c.subject for c in repo.log()] == ["rename, better worded", "add"]


@needs_git
def test_branches_remotes_push_clone_pull(tmp_path) -> None:
    remote = tmp_path / "remote.git"
    Git(tmp_path).run("init", "--bare", "-b", "main", str(remote))
    first = repo_with_identity(tmp_path / "first")
    (first.root / "README.md").write_text("# hello\n")
    first.stage(["README.md"])
    first.commit("start")
    first.add_remote("origin", str(remote))
    assert first.remotes() == {"origin": str(remote)}
    first.push()
    status = first.status()
    assert status.upstream == "origin/main" and (status.ahead, status.behind) == (0, 0)

    second = Git.clone(str(remote), tmp_path / "second")
    second.set_identity("Other", "other@example.com")
    (second.root / "README.md").write_text("# hello\nmore\n")
    second.stage(["README.md"])
    second.commit("more")
    second.create_branch("feature")
    assert [b.name for b in second.branches() if b.current] == ["feature"]
    second.checkout("main")
    second.push()

    first.fetch()
    assert first.status().behind == 1
    first.pull()
    assert (first.root / "README.md").read_text().endswith("more\n") and first.status().behind == 0
    names = {(b.name, b.remote) for b in first.branches()}
    assert ("main", False) in names and ("origin/main", True) in names
    with pytest.raises(GitError):
        first.create_branch("bad name..")
    with pytest.raises(GitError, match="not empty"):
        Git.clone(str(remote), tmp_path / "second")


def test_tokens_reach_git_only_through_the_environment(tmp_path) -> None:
    env = Git(tmp_path, token="ghp_secret123").environment()
    count = int(env["GIT_CONFIG_COUNT"])
    assert env[f"GIT_CONFIG_KEY_{count - 1}"] == "http.https://github.com/.extraheader"
    assert env[f"GIT_CONFIG_VALUE_{count - 1}"].startswith("AUTHORIZATION: basic ")
    assert env["GIT_TERMINAL_PROMPT"] == "0"
    assert "GIT_CONFIG_COUNT" not in Git(tmp_path).environment() or count > 1


def test_github_slugs() -> None:
    assert github_slug("https://github.com/tedo001/tedo-sim-lab.git") == ("tedo001", "tedo-sim-lab")
    assert github_slug("git@github.com:tedo001/sentra.git") == ("tedo001", "sentra")
    assert github_slug("https://user@github.com/a/b") == ("a", "b")
    assert github_slug("https://gitlab.com/a/b.git") is None


class FakeGitHub(BaseHTTPRequestHandler):
    seen: list[tuple[str, str, str]] = []

    def log_message(self, *args) -> None:
        pass

    def _send(self, code: int, body) -> None:
        data = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:  # noqa: N802
        FakeGitHub.seen.append(("GET", self.path, self.headers.get("Authorization", "")))
        if self.headers.get("Authorization") != "Bearer good-token":
            return self._send(401, {"message": "Bad credentials"})
        if self.path == "/user":
            return self._send(200, {"login": "tedo001", "name": "Tedo", "html_url": "https://github.com/tedo001"})
        if self.path.startswith("/user/repos"):
            from urllib.parse import parse_qs, urlparse
            page = int(parse_qs(urlparse(self.path).query)["page"][0])
            count = 100 if page == 1 else 3
            return self._send(200, [{"full_name": f"tedo001/r{page}-{i}", "private": i % 2 == 0,
                                     "clone_url": f"https://github.com/tedo001/r{page}-{i}.git",
                                     "pushed_at": "2026-10-01T00:00:00Z"} for i in range(count)])
        if self.path.startswith("/repos/tedo001/lab/pulls"):
            return self._send(200, [{"number": 7, "title": "Add simulation", "user": {"login": "a"},
                                     "html_url": "https://github.com/tedo001/lab/pull/7"}])
        if self.path.startswith("/repos/tedo001/lab/issues"):
            return self._send(200, [{"number": 7, "title": "PR", "pull_request": {}},
                                    {"number": 9, "title": "Bug", "user": {"login": "b"}}])
        if self.path == "/repos/tedo001/lab/releases/latest":
            return self._send(200, {"tag_name": "v0.2.0", "html_url": "https://github.com/x",
                                    "published_at": "2026-11-01T00:00:00Z"})
        return self._send(404, {"message": "Not Found"})

    def do_POST(self) -> None:  # noqa: N802
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        FakeGitHub.seen.append(("POST", self.path, json.dumps(body)))
        if body["name"] == "taken":
            return self._send(422, {"message": "name already exists on this account"})
        return self._send(201, {"full_name": f"tedo001/{body['name']}", "private": body["private"],
                                "clone_url": f"https://github.com/tedo001/{body['name']}.git"})


@pytest.fixture
def api(monkeypatch):
    for name in ("http_proxy", "HTTP_PROXY", "https_proxy", "HTTPS_PROXY", "all_proxy", "ALL_PROXY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("no_proxy", "*")
    FakeGitHub.seen = []
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), FakeGitHub)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()


def test_github_account_repositories_and_project(api) -> None:
    hub = GitHub("good-token", api=api)
    assert hub.user()["login"] == "tedo001"
    repos = hub.repos(limit=500)
    assert len(repos) == 103 and repos[0].full_name == "tedo001/r1-0" and repos[0].private
    created = hub.create_repo("experiments", private=True, description="lab")
    assert created.clone_url.endswith("/experiments.git")
    assert json.loads(FakeGitHub.seen[-1][2])["private"] is True
    with pytest.raises(GitHubError, match="422"):
        hub.create_repo("taken")
    assert [p.number for p in hub.pulls("tedo001", "lab")] == [7]
    assert [i.number for i in hub.issues("tedo001", "lab")] == [9]  # pull requests are not issues
    assert hub.latest_release("tedo001", "lab").tag == "v0.2.0"
    assert hub.latest_release("tedo001", "none") is None


def test_github_refuses_a_bad_token_without_echoing_it(api) -> None:
    from core.common.masking import register_secret

    register_secret("bad-token-value")
    with pytest.raises(GitHubError) as caught:
        GitHub("bad-token-value", api=api).user()
    assert caught.value.status == 401 and "bad-token-value" not in str(caught.value)
    with pytest.raises(GitHubError, match="could not be reached"):
        GitHub("x", api="http://127.0.0.1:9", timeout=2).user()
