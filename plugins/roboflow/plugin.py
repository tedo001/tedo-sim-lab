"""Roboflow: list the projects in your Roboflow workspace and export a dataset version into the
lab, through Roboflow's REST API (``https://api.roboflow.com``) with your API key. Each project
shows the licence Roboflow reports for it ("unspecified" when it reports none)."""

from __future__ import annotations

import urllib.parse
from pathlib import Path

from core.common.cancel import CancelToken, ProgressFn
from core.common.masking import mask_text
from core.plugin_api import ActionSpec, ConnectionResult, PluginError
from labs.common.download import DownloadError, download_file
from labs.common.remote import RemoteError, RemoteItem, get_json, unpack_zip
from plugins.sources import SourcePlugin

__all__ = ["RoboflowPlugin"]

API = "https://api.roboflow.com"
#: Export formats the lab can read today (COCO in the Dataset Hub's inspector).
FORMAT = "coco"


class RoboflowPlugin(SourcePlugin):
    FOLDER = "roboflow"
    ACTIONS = (
        ActionSpec("search", "List projects", "Your Roboflow workspace's projects", needs_connection=True),
        ActionSpec("download", "Download dataset", "Export a dataset version (COCO) into the workspace",
                   needs_connection=True),
    )

    def _key(self) -> str:
        key = self.secret("ROBOFLOW_API_KEY")
        if not key:
            raise PluginError("Roboflow is not connected: set ROBOFLOW_API_KEY.")
        return key

    def _get(self, path: str) -> dict:
        query = urllib.parse.urlencode({"api_key": self._key()})
        answer = get_json(f"{API}/{path}?{query}")
        if not isinstance(answer, dict):
            raise RemoteError("unexpected answer from Roboflow")
        return answer

    def workspace(self) -> str:
        name = self._get("").get("workspace")
        if not name:
            raise RemoteError("Roboflow did not name a workspace for this API key")
        return str(name)

    def search(self, query: str) -> list[RemoteItem]:
        workspace = self._get(urllib.parse.quote(self.workspace())).get("workspace", {})
        items = []
        for project in workspace.get("projects", []):
            versions = int(project.get("versions") or 0)
            project_id = str(project.get("id", ""))
            title = str(project.get("name") or project_id)
            if query and query.lower() not in f"{title} {project_id}".lower():
                continue
            restriction = "" if versions else "it has no generated dataset version to export yet."
            items.append(RemoteItem("roboflow", f"{project_id}/{versions}", f"{title} · version {versions}",
                                    "dataset", f"https://app.roboflow.com/{project_id}",
                                    licence=str(project.get("license") or ""), restriction=restriction,
                                    details={"type": project.get("type"), "images": project.get("images"),
                                             "classes": list((project.get("classes") or {}).keys())}))
        return items

    def _fetch(self, item: RemoteItem, folder: Path, *, cancel: CancelToken, progress: ProgressFn) -> None:
        parts = item.id.split("/")
        if len(parts) != 3 or not parts[2].isdigit():
            raise PluginError(f"{item.id!r} is not a Roboflow dataset version (workspace/project/version)")
        progress(-1.0, "Asking Roboflow to prepare the export")
        path = "/".join(urllib.parse.quote(part) for part in parts)
        export = self._get(f"{path}/{FORMAT}").get("export", {})
        link = export.get("link") if isinstance(export, dict) else None
        if not link:
            raise PluginError("Roboflow has not finished preparing this export; try again in a minute.")
        archive = folder / "dataset.zip"
        try:
            download_file([link], archive, progress=progress, cancel=cancel, timeout=120)
        except DownloadError as exc:
            raise PluginError(mask_text(f"Roboflow download failed: {exc}")) from None
        progress(-1.0, "Unpacking")
        unpack_zip(archive, folder)
        archive.unlink()

    def _test_connection(self) -> ConnectionResult:
        try:
            return ConnectionResult(True, f"Connected to workspace {self.workspace()}.")
        except (RemoteError, PluginError) as exc:
            return ConnectionResult(False, str(exc))
