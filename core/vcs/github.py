"""GitHub through its REST API (standard library only; blocking: call from a task).

The account behind a token, its repositories, creating one, a repository's open pull requests
and issues, and the newest release (the app's update check). The token travels only in the
``Authorization`` header and never appears in an error message.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any

from core.common.masking import mask_text

__all__ = ["GitHub", "GitHubError", "Release", "Repo"]

API = "https://api.github.com"


class GitHubError(RuntimeError):
    def __init__(self, message: str, status: int = 0) -> None:
        super().__init__(mask_text(message))
        self.status = status


@dataclass(frozen=True)
class Repo:
    full_name: str
    private: bool
    description: str
    clone_url: str
    html_url: str
    default_branch: str
    updated_at: str
    language: str = ""
    stars: int = 0

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> Repo:
        return cls(data.get("full_name", ""), bool(data.get("private")), data.get("description") or "",
                   data.get("clone_url", ""), data.get("html_url", ""), data.get("default_branch") or "main",
                   (data.get("pushed_at") or data.get("updated_at") or "")[:10], data.get("language") or "",
                   int(data.get("stargazers_count") or 0))


@dataclass(frozen=True)
class Item:
    """A pull request or an issue."""

    number: int
    title: str
    author: str
    html_url: str
    updated_at: str


@dataclass(frozen=True)
class Release:
    tag: str
    name: str
    html_url: str
    published_at: str


class GitHub:
    def __init__(self, token: str = "", *, api: str = API, timeout: float = 30) -> None:
        self.api = api.rstrip("/")
        self._token = token
        self.timeout = timeout

    @property
    def connected(self) -> bool:
        return bool(self._token)

    def request(self, method: str, path: str, body: dict | None = None,
                query: dict[str, Any] | None = None) -> Any:
        url = f"{self.api}{path}"
        if query:
            url += "?" + urllib.parse.urlencode(query)
        headers = {"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28",
                   "User-Agent": "tedo-ai-research-lab"}
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"
        data = json.dumps(body).encode() if body is not None else None
        if data is not None:
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(url, data=data, method=method, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                raw = response.read()
        except urllib.error.HTTPError as exc:
            raise GitHubError(_explain(exc), exc.code) from None
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            reason = getattr(exc, "reason", exc)
            raise GitHubError(f"GitHub could not be reached: {reason}") from None
        return json.loads(raw) if raw else None

    # Account and repositories -------------------------------------------------------------------
    def user(self) -> dict[str, str]:
        data = self.request("GET", "/user")
        return {"login": data.get("login", ""), "name": data.get("name") or "",
                "html_url": data.get("html_url", "")}

    def repos(self, *, limit: int = 100) -> list[Repo]:
        """The account's repositories, most recently pushed first."""
        found: list[Repo] = []
        page = 1
        while len(found) < limit:
            query = {"sort": "pushed", "per_page": 100, "page": page, "affiliation": "owner,collaborator"}
            batch = self.request("GET", "/user/repos", query=query)
            found += [Repo.from_json(item) for item in batch or []]
            if not batch or len(batch) < 100:
                break
            page += 1
        return found[:limit]

    def create_repo(self, name: str, *, private: bool = True, description: str = "") -> Repo:
        data = self.request("POST", "/user/repos", {"name": name, "private": private,
                                                    "description": description, "auto_init": False})
        return Repo.from_json(data)

    def repo(self, owner: str, name: str) -> Repo:
        return Repo.from_json(self.request("GET", f"/repos/{_quote(owner)}/{_quote(name)}"))

    def pulls(self, owner: str, name: str) -> list[Item]:
        data = self.request("GET", f"/repos/{_quote(owner)}/{_quote(name)}/pulls",
                            query={"state": "open", "per_page": 50})
        return [_item(entry) for entry in data or []]

    def issues(self, owner: str, name: str) -> list[Item]:
        data = self.request("GET", f"/repos/{_quote(owner)}/{_quote(name)}/issues",
                            query={"state": "open", "per_page": 50})
        return [_item(entry) for entry in data or [] if "pull_request" not in entry]

    def latest_release(self, owner: str, name: str) -> Release | None:
        try:
            data = self.request("GET", f"/repos/{_quote(owner)}/{_quote(name)}/releases/latest")
        except GitHubError as exc:
            if exc.status == 404:  # no release published yet
                return None
            raise
        return Release(data.get("tag_name", ""), data.get("name") or data.get("tag_name", ""),
                       data.get("html_url", ""), (data.get("published_at") or "")[:10])


def _quote(part: str) -> str:
    return urllib.parse.quote(part, safe="")


def _item(entry: dict[str, Any]) -> Item:
    return Item(int(entry.get("number", 0)), entry.get("title", ""),
                (entry.get("user") or {}).get("login", ""), entry.get("html_url", ""),
                (entry.get("updated_at") or "")[:10])


def _explain(exc: urllib.error.HTTPError) -> str:
    try:
        message = json.loads(exc.read() or b"{}").get("message", "")
    except (ValueError, OSError):
        message = ""
    if exc.code == 401:
        return "GitHub refused the token (401): it is wrong, expired or revoked"
    if exc.code == 403 and "rate limit" in message.lower():
        return "GitHub's rate limit was reached; connect a token or try again later"
    if exc.code == 404:
        return "Not found on GitHub (404), or the token cannot see it"
    if exc.code == 422:
        return f"GitHub did not accept it (422): {message or 'invalid request'}"
    return f"GitHub answered {exc.code}: {message or exc.reason}"
