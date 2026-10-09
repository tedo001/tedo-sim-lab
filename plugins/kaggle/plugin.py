"""Kaggle: search public datasets and download one on request, through Kaggle's public REST API
(``https://www.kaggle.com/api/v1``) with your username and API key (HTTP basic auth). Every
dataset shows the licence its owner chose; competitions (which need their rules accepted on
Kaggle first) are not downloaded."""

from __future__ import annotations

import base64
import urllib.parse
from pathlib import Path

from core.common.cancel import CancelToken, ProgressFn
from core.plugin_api import ActionSpec, ConnectionResult, PluginError
from labs.common.download import DownloadError, download_file
from labs.common.remote import RemoteError, RemoteItem, get_json, unpack_zip
from plugins.sources import SourcePlugin

__all__ = ["KagglePlugin"]

API = "https://www.kaggle.com/api/v1"


class KagglePlugin(SourcePlugin):
    FOLDER = "kaggle"
    ACTIONS = (
        ActionSpec("search", "Search datasets", "Find datasets on Kaggle by keyword", needs_connection=True),
        ActionSpec("download", "Download dataset", "Download a dataset into the workspace",
                   needs_connection=True),
    )

    def _headers(self) -> dict[str, str]:
        user, key = self.secret("KAGGLE_USERNAME"), self.secret("KAGGLE_KEY")
        if not user or not key:
            raise PluginError("Kaggle is not connected: set KAGGLE_USERNAME and KAGGLE_KEY.")
        token = base64.b64encode(f"{user}:{key}".encode()).decode()
        return {"Authorization": f"Basic {token}"}

    @staticmethod
    def item(row: dict) -> RemoteItem:
        ref = str(row.get("ref", ""))
        return RemoteItem("kaggle", ref, str(row.get("title") or ref), "dataset",
                          str(row.get("url") or f"https://www.kaggle.com/datasets/{ref}"),
                          licence=str(row.get("licenseName") or ""),
                          size_bytes=int(row["totalBytes"]) if row.get("totalBytes") else None,
                          details={"downloads": row.get("downloadCount"), "updated": row.get("lastUpdated"),
                                   "owner": row.get("ownerName")})

    def search(self, query: str) -> list[RemoteItem]:
        url = f"{API}/datasets/list?" + urllib.parse.urlencode({"search": query, "page": 1})
        rows = get_json(url, headers=self._headers())
        if not isinstance(rows, list):
            raise RemoteError("unexpected answer from Kaggle's dataset list")
        return [self.item(row) for row in rows if isinstance(row, dict) and row.get("ref")]

    def _fetch(self, item: RemoteItem, folder: Path, *, cancel: CancelToken, progress: ProgressFn) -> None:
        owner, _, name = item.id.partition("/")
        if not owner or not name or "/" in name:
            raise PluginError(f"{item.id!r} is not a Kaggle dataset reference (owner/name)")
        url = f"{API}/datasets/download/{urllib.parse.quote(owner)}/{urllib.parse.quote(name)}"
        archive = folder / "dataset.zip"
        try:
            download_file([url], archive, progress=progress, cancel=cancel, headers=self._headers(),
                          timeout=120)
        except DownloadError as exc:
            raise PluginError(f"Kaggle download failed: {exc}") from None
        progress(-1.0, "Unpacking")
        unpack_zip(archive, folder)
        archive.unlink()

    def _test_connection(self) -> ConnectionResult:
        try:
            get_json(f"{API}/datasets/list?page=1&search=mnist", headers=self._headers())
        except (RemoteError, PluginError) as exc:
            return ConnectionResult(False, str(exc))
        return ConnectionResult(True, "Kaggle accepted the credentials.")
