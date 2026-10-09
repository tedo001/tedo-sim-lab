"""The part of the Kaggle, Hugging Face and Roboflow plugins that is the same: search returns
:class:`~labs.common.remote.RemoteItem`s; download refuses restricted items, saves into the
workspace and records the source and licence next to the files."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, ClassVar

from core.common.cancel import CancelToken, ProgressFn
from core.plugin_api import ManifestPlugin, PluginError
from labs.common.remote import RemoteError, RemoteItem, safe_folder_name, write_source

__all__ = ["SourcePlugin"]


class SourcePlugin(ManifestPlugin):
    """Actions ``search`` (``query``) and ``download`` (``item``, ``workspace``, ``cancel``,
    ``progress``). Subclasses implement :meth:`search` and :meth:`_fetch`."""

    #: Folder under ``datasets/`` and ``models/`` that this source downloads into.
    FOLDER: ClassVar[str] = ""

    def _run(self, action: str, **kwargs: Any) -> Any:
        try:
            if action == "search":
                return self.search(str(kwargs.get("query", "")).strip())
            if action == "download":
                return self.download(kwargs["item"], Path(kwargs["workspace"]), cancel=kwargs["cancel"],
                                     progress=kwargs["progress"])
        except RemoteError as exc:
            raise PluginError(f"{self.manifest.title}: {exc}") from None
        return super()._run(action, **kwargs)

    def secret(self, key: str) -> str | None:
        value = self.credentials.get(key)
        return value.reveal() if value else None

    def search(self, query: str) -> list[RemoteItem]:
        raise NotImplementedError

    def destination(self, item: RemoteItem, workspace: Path) -> Path:
        area = "models" if item.kind == "model" else "datasets"
        return workspace / area / self.FOLDER / safe_folder_name(item.id)

    def download(self, item: RemoteItem, workspace: Path, *, cancel: CancelToken,
                 progress: ProgressFn) -> Path:
        """Fetch ``item`` into ``<workspace>/datasets|models/<source>/<id>/``; the folder is
        only kept when the download finished."""
        if item.restriction:
            raise PluginError(f"{item.title} is not downloaded by the lab: {item.restriction} "
                              f"See {item.url}")
        folder = self.destination(item, workspace)
        staging = folder.with_name(folder.name + ".partial")
        shutil.rmtree(staging, ignore_errors=True)
        staging.mkdir(parents=True)
        try:
            self._fetch(item, staging, cancel=cancel, progress=progress)
            write_source(staging, item)
            shutil.rmtree(folder, ignore_errors=True)
            staging.rename(folder)
        finally:
            shutil.rmtree(staging, ignore_errors=True)
        return folder

    def _fetch(self, item: RemoteItem, folder: Path, *, cancel: CancelToken, progress: ProgressFn) -> None:
        raise NotImplementedError
