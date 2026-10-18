"""The Git & GitHub page against real repositories (a bare one as the remote) and a local
stand-in for GitHub's API."""

from __future__ import annotations

import shutil

import pytest

from app.ui.pages.source_control import SourceControlPage
from core.vcs import Git
from tests.core.test_vcs import FakeGitHub, api  # noqa: F401 (the stand-in server fixture)

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="needs git")


@pytest.fixture(autouse=True)
def _no_personal_git_config(monkeypatch, tmp_path):
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(tmp_path / "no-global-gitconfig"))
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")


@pytest.fixture
def repo(tmp_path) -> Git:
    git = Git.init(tmp_path / "project")
    git.set_identity("Tester", "tester@example.com")
    (git.root / "README.md").write_text("# project\n")
    git.stage(["README.md"])
    git.commit("Start")
    return git


@pytest.fixture
def page(ctx, qtbot) -> SourceControlPage:
    widget = SourceControlPage(ctx)
    qtbot.addWidget(widget)
    widget.resize(1100, 900)
    widget.show()
    qtbot.waitUntil(lambda: not widget.control.busy(), timeout=10_000)
    return widget


def opened(page, qtbot, folder) -> None:
    page.open_folder(folder)
    qtbot.waitUntil(lambda: page.control.root is not None and page.status is not None
                    and not page.control.busy(), timeout=10_000)


def test_changes_are_staged_committed_and_shown_in_history(page, qtbot, repo) -> None:
    opened(page, qtbot, repo.root)
    assert page.branch_box.currentText() == "main" and page.changes.changes_table.rowCount() == 0
    (repo.root / "README.md").write_text("# project\nmore\n")
    (repo.root / "new.py").write_text("x = 1\n")
    page.refresh()
    qtbot.waitUntil(lambda: page.changes.changes_table.rowCount() == 2, timeout=10_000)
    assert page.changes.diff.toPlainText() == "Choose a file to see what changed in it."
    page.changes.changes_table.selectRow(0)  # a row becoming current shows its diff
    qtbot.waitUntil(lambda: "+more" in page.changes.diff.toPlainText(), timeout=10_000)
    page.changes.stage_all_button.click()
    qtbot.waitUntil(lambda: page.changes.staged_table.rowCount() == 2, timeout=10_000)
    assert page.changes.commit_button.isEnabled()
    page.changes.message_box.setPlainText("Add new.py and more text")
    page.changes.commit()
    qtbot.waitUntil(lambda: "Committed" in page.message.text() and page.changes.staged_table.rowCount() == 0,
                    timeout=10_000)
    assert [c.subject for c in repo.log()][:2] == ["Add new.py and more text", "Start"]
    page.views.select("history")
    qtbot.waitUntil(lambda: page.history.table.rowCount() == 2, timeout=10_000)
    page.history.show_commit(0)
    qtbot.waitUntil(lambda: "new.py" in page.history.diff.toPlainText(), timeout=10_000)


def test_branches_remotes_identity_push_and_clone(page, qtbot, repo, tmp_path) -> None:
    remote = tmp_path / "remote.git"
    Git(tmp_path).run("init", "--bare", "-b", "main", str(remote))
    opened(page, qtbot, repo.root)
    page.views.select("branches")
    page.branches.add_remote("origin", str(remote))
    qtbot.waitUntil(lambda: page.branches.remotes.rowCount() == 1, timeout=10_000)
    page.push()
    qtbot.waitUntil(lambda: page.message.text() == "Pushed." and "origin/main" in page.sync_label.text(),
                    timeout=20_000)
    page.branches.new_branch("feature/sim")
    qtbot.waitUntil(lambda: page.branch_box.currentText() == "feature/sim", timeout=10_000)
    page.branches.name.setText("Lab Person")
    page.branches.email.setText("lab@example.com")
    page.branches.save_identity()
    qtbot.waitUntil(lambda: repo.identity() == ("Lab Person", "lab@example.com"), timeout=10_000)
    page.clone_url(str(remote), parent_folder=tmp_path / "clones")
    qtbot.waitUntil(lambda: page.control.root == (tmp_path / "clones" / "remote").resolve(), timeout=20_000)
    assert (tmp_path / "clones" / "remote" / "README.md").is_file()


def test_a_plain_folder_can_become_a_repository(page, qtbot, tmp_path) -> None:
    folder = tmp_path / "plain"
    folder.mkdir()
    page.open_folder(folder)
    qtbot.waitUntil(lambda: page.init_row.isVisibleTo(page)
                    and "not a git repository" in page.init_hint.text(), timeout=10_000)
    assert not page.push_button.isEnabled()
    page.init_repository()
    qtbot.waitUntil(lambda: page.control.root == folder.resolve() and page.status is not None, timeout=10_000)
    assert not page.init_row.isVisibleTo(page) and page.status.branch == "main"


def test_github_account_repositories_and_publishing(page, qtbot, ctx, api, tmp_path) -> None:  # noqa: F811
    page.control.api = api
    page.views.select("github")
    view = page.github
    assert "Not connected" in view.account_state.text() and not view.repo_table.isEnabled()
    view.connect_account("good-token")
    qtbot.waitUntil(lambda: "Connected as tedo001" in view.account_state.text(), timeout=10_000)
    assert ctx.credentials.source("GITHUB_TOKEN") == "keyring" and view.repo_table.rowCount() == 103
    view.filter.setText("r2-")
    assert view.repo_table.rowCount() == 3
    fresh = Git.init(tmp_path / "unpublished")  # no commits: publishing needs no push
    opened(page, qtbot, fresh.root)
    page.views.select("github")
    qtbot.waitUntil(lambda: view.publish_button.isVisibleTo(view) and view.publish_button.isEnabled(),
                    timeout=10_000)
    assert view.repo_name.text() == "unpublished"
    view.publish()
    qtbot.waitUntil(lambda: "Published to github.com/tedo001/unpublished" in page.message.text(),
                    timeout=10_000)
    assert fresh.remotes()["origin"] == "https://github.com/tedo001/unpublished.git"
    view.disconnect_account()
    assert ctx.credentials.source("GITHUB_TOKEN") == "missing"
    assert "Not connected" in view.account_state.text()


def test_a_bad_token_is_refused_and_not_stored(page, qtbot, ctx, api) -> None:  # noqa: F811
    page.control.api = api
    page.github.connect_account("wrong")
    qtbot.waitUntil(lambda: "refused the token" in page.github.account_state.text(), timeout=10_000)
    assert ctx.credentials.source("GITHUB_TOKEN") == "missing"
    assert FakeGitHub.seen[-1][2] == "Bearer wrong"
