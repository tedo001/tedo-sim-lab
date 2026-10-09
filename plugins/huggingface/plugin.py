"""Hugging Face Hub: search models and datasets and download one on request, through the Hub's
public REST API (``https://huggingface.co/api``). The token (``HF_TOKEN``) is optional for
public repositories. Gated repositories, whose terms must be accepted on the Hub first, and
private ones are never downloaded by the lab: it shows their page instead."""

from __future__ import annotations

import urllib.parse
from pathlib import Path
from typing import Any

from core.common.cancel import CancelToken, ProgressFn
from core.plugin_api import ActionSpec, ConnectionResult, PluginError
from labs.common.download import DownloadError, download_file
from labs.common.remote import RemoteError, RemoteItem, get_json
from plugins.sources import SourcePlugin

__all__ = ["HuggingFacePlugin", "hub_item"]

HUB = "https://huggingface.co"
GATED = ("gated on the Hub: its owner asks you to accept terms (and may review your request) on its "
         "page first. Download it yourself after reading them.")
#: Largest repository the lab downloads in one go; bigger ones are better fetched file by file.
MAX_BYTES = 20e9


def _get(info: Any, key: str, default: Any = None) -> Any:
    return info.get(key, default) if isinstance(info, dict) else getattr(info, key, default)


def _licence(info: Any) -> str:
    card = _get(info, "cardData") or _get(info, "card_data") or {}
    licence = card.get("license") if isinstance(card, dict) else getattr(card, "license", None)
    if not licence:
        licence = next((tag.split(":", 1)[1] for tag in (_get(info, "tags") or [])
                        if str(tag).startswith("license:")), "")
    return ", ".join(licence) if isinstance(licence, list) else str(licence or "")


def hub_item(info: Any, kind: str) -> RemoteItem:
    """A Hub record (the API's JSON, or anything with the same fields) as an item, restricted
    when gated or private."""
    repo = str(_get(info, "id") or _get(info, "modelId"))
    restriction = ""
    if _get(info, "gated", False):
        restriction = GATED
    elif _get(info, "private", False):
        restriction = "private: only its owner and members can see it."
    files = [s for s in (_get(info, "siblings") or []) if isinstance(s, dict) and s.get("rfilename")]
    sizes = [s.get("size") for s in files]
    size = sum(sizes) if files and all(isinstance(v, int) for v in sizes) else None
    url = f"{HUB}/{'datasets/' if kind == 'dataset' else ''}{repo}"
    return RemoteItem("huggingface", repo, repo, kind, url, licence=_licence(info), size_bytes=size,
                      restriction=restriction,
                      details={"downloads": _get(info, "downloads"), "task": _get(info, "pipeline_tag"),
                               "files": [s["rfilename"] for s in files]})


class HuggingFacePlugin(SourcePlugin):
    FOLDER = "huggingface"
    ACTIONS = (
        ActionSpec("search", "Search the Hub", "Find models and datasets by keyword"),
        ActionSpec("download", "Download", "Download a public repository into the workspace"),
    )

    def _headers(self) -> dict[str, str]:
        token = self.secret("HF_TOKEN")
        return {"Authorization": f"Bearer {token}"} if token else {}

    def _api(self, path: str, **query: Any) -> Any:
        suffix = f"?{urllib.parse.urlencode(query)}" if query else ""
        return get_json(f"{HUB}/api/{path}{suffix}", headers=self._headers())

    def search(self, query: str, limit: int = 15) -> list[RemoteItem]:
        found = []
        for kind, path in (("model", "models"), ("dataset", "datasets")):
            rows = self._api(path, search=query, limit=limit, sort="downloads", direction=-1)
            if not isinstance(rows, list):
                raise RemoteError(f"unexpected answer from the Hub's {path} list")
            found += [hub_item(row, kind) for row in rows if isinstance(row, dict) and row.get("id")]
        return found

    def info(self, item: RemoteItem) -> RemoteItem:
        """The full record (gate, licence, every file with its size)."""
        path = "datasets" if item.kind == "dataset" else "models"
        return hub_item(self._api(f"{path}/{item.id}", blobs="true"), item.kind)

    def _fetch(self, item: RemoteItem, folder: Path, *, cancel: CancelToken, progress: ProgressFn) -> None:
        checked = self.info(item)  # the search listing can lack the gate flag
        if checked.restriction:
            raise PluginError(f"{item.id} is not downloaded by the lab: {checked.restriction} See {item.url}")
        if checked.size_bytes and checked.size_bytes > MAX_BYTES:
            raise PluginError(f"{item.id} holds {checked.size_text}; download what you need from {item.url}")
        prefix = "datasets/" if item.kind == "dataset" else ""
        files = checked.details["files"]
        for index, name in enumerate(files, start=1):
            parts = Path(name).parts
            if Path(name).is_absolute() or ".." in parts:
                raise PluginError(f"refusing the file name {name!r}")
            url = f"{HUB}/{prefix}{item.id}/resolve/main/{urllib.parse.quote(name)}"

            def step(fraction: float, message: str, i: int = index) -> None:
                progress(fraction, f"{i}/{len(files)} · {message}")

            try:
                download_file([url], folder.joinpath(*parts), progress=step, cancel=cancel,
                              headers=self._headers(), timeout=120)
            except DownloadError as exc:
                raise PluginError(f"Hugging Face download failed: {exc}") from None

    def _test_connection(self) -> ConnectionResult:
        if not self.secret("HF_TOKEN"):
            return ConnectionResult(True, "No token set: public repositories only.")
        try:
            who = self._api("whoami-v2")
        except RemoteError as exc:
            return ConnectionResult(False, str(exc))
        return ConnectionResult(True, f"Connected as {who.get('name', 'unknown')}.")
