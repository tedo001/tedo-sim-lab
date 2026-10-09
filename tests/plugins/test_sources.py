"""Kaggle and Roboflow against a local stand-in for their REST APIs, and the rules every source
follows: credentials in headers (never in errors), restricted items refused, archives unpacked
only inside their folder, the licence recorded next to the files."""

from __future__ import annotations

import base64
import io
import json
import threading
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import pytest

from core.common import CredentialStore
from core.common.cancel import CancelToken
from core.common.paths import CODE_ROOT
from core.plugin_api import PluginError, PluginRegistry
from labs.common.remote import RemoteError, RemoteItem, unpack_zip

KAGGLE_USER, KAGGLE_KEY, ROBOFLOW_KEY = "ada", "kaggle-key-0123456789", "rf-key-0123456789abcdef"


def zipped(files: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as bundle:
        for name, data in files.items():
            bundle.writestr(name, data)
    return buffer.getvalue()


class FakeApis(BaseHTTPRequestHandler):
    routes: dict[str, tuple[int, bytes]] = {}
    seen: list[tuple[str, str | None]] = []

    def do_GET(self) -> None:  # noqa: N802 - the http.server API
        self.seen.append((self.path, self.headers.get("Authorization")))
        status, body = self.routes.get(self.path, (404, b"{}"))
        expected = "Basic " + base64.b64encode(f"{KAGGLE_USER}:{KAGGLE_KEY}".encode()).decode()
        if self.path.startswith("/api/v1/") and self.headers.get("Authorization") != expected:
            status, body = 401, b"{}"
        self.send_response(status)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_args) -> None:
        pass


@pytest.fixture
def server(monkeypatch):
    for name in ("http_proxy", "HTTP_PROXY", "https_proxy", "HTTPS_PROXY", "all_proxy", "ALL_PROXY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("no_proxy", "*")
    FakeApis.routes, FakeApis.seen = {}, []
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), FakeApis)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()


def plugin(name: str, env: dict[str, str]):
    registry = PluginRegistry(CredentialStore(env=env, backend=None))
    registry.discover(CODE_ROOT / "plugins")
    return registry.get(name)


def fetch(source, item, root):
    return source.run("download", item=item, workspace=root, cancel=CancelToken(),
                      progress=lambda *_: None)


def test_kaggle_search_and_download(server, monkeypatch, tmp_path) -> None:
    import plugins.kaggle.plugin as kaggle_module

    monkeypatch.setattr(kaggle_module, "API", f"{server}/api/v1")
    rows = [{"ref": "ada/iris-flowers", "title": "Iris flowers", "licenseName": "CC0: Public Domain",
             "totalBytes": 4500, "url": "https://www.kaggle.com/datasets/ada/iris-flowers"}]
    archive = zipped({"iris.csv": b"a,b\n1,2\n"})
    FakeApis.routes = {"/api/v1/datasets/list?search=iris&page=1": (200, json.dumps(rows).encode()),
                       "/api/v1/datasets/download/ada/iris-flowers": (200, archive)}
    kaggle = plugin("kaggle", {"KAGGLE_USERNAME": KAGGLE_USER, "KAGGLE_KEY": KAGGLE_KEY})
    assert kaggle.status == "available"
    [item] = kaggle.run("search", query="iris")
    assert item.id == "ada/iris-flowers" and item.licence == "CC0: Public Domain"
    assert item.size_text == "4.5 KB"
    folder = fetch(kaggle, item, tmp_path)
    assert folder == tmp_path / "datasets" / "kaggle" / "ada__iris-flowers"
    assert (folder / "iris.csv").read_text() == "a,b\n1,2\n" and not (folder / "dataset.zip").exists()
    note = json.loads((folder / "source.json").read_text())
    assert note["licence"] == "CC0: Public Domain" and note["url"].endswith("iris-flowers")
    assert [p.name for p in folder.parent.iterdir()] == ["ada__iris-flowers"]  # no partial folder left
    wrong = plugin("kaggle", {"KAGGLE_USERNAME": KAGGLE_USER, "KAGGLE_KEY": "wrong-key-0123456789"})
    result = wrong.test_connection()
    assert not result.ok and "HTTP 401" in result.message and "wrong-key" not in result.message


def test_roboflow_lists_projects_and_exports_a_version(server, monkeypatch, tmp_path) -> None:
    import plugins.roboflow.plugin as roboflow_module

    monkeypatch.setattr(roboflow_module, "API", server)
    key = f"api_key={ROBOFLOW_KEY}"
    projects = {"workspace": {"projects": [
        {"id": "lab/cells", "name": "Cells", "versions": 2, "license": "CC BY 4.0", "classes": {"cell": 10}},
        {"id": "lab/empty", "name": "Empty", "versions": 0}]}}
    FakeApis.routes = {f"/?{key}": (200, b'{"workspace": "lab"}'),
                       f"/lab?{key}": (200, json.dumps(projects).encode()),
                       f"/lab/cells/2/coco?{key}": (200, json.dumps({"export": {"link": f"{server}/zip"}})
                                                    .encode()),
                       "/zip": (200, zipped({"train/_annotations.coco.json": b"{}"}))}
    roboflow = plugin("roboflow", {"ROBOFLOW_API_KEY": ROBOFLOW_KEY})
    cells, empty = roboflow.run("search", query="")
    assert cells.id == "lab/cells/2" and cells.licence == "CC BY 4.0" and not cells.restriction
    assert empty.restriction and empty.licence_text.startswith("No licence")
    folder = fetch(roboflow, cells, tmp_path)
    assert (folder / "train" / "_annotations.coco.json").is_file()
    with pytest.raises(PluginError, match="not downloaded by the lab"):
        fetch(roboflow, empty, tmp_path)
    assert roboflow.test_connection().ok


def test_hugging_face_search_and_download(server, monkeypatch, tmp_path) -> None:
    import plugins.huggingface.plugin as hub_module

    monkeypatch.setattr(hub_module, "HUB", server)
    models = [{"id": "org/tiny", "tags": ["license:apache-2.0"], "downloads": 5},
              {"id": "org/locked", "gated": "manual", "tags": []}]
    full = {"id": "org/tiny", "tags": ["license:apache-2.0"],
            "siblings": [{"rfilename": "config.json", "size": 2}, {"rfilename": "sub/w.bin", "size": 3}]}
    query = "search=tiny&limit=15&sort=downloads&direction=-1"
    FakeApis.routes = {f"/api/models?{query}": (200, json.dumps(models).encode()),
                       f"/api/datasets?{query}": (200, b"[]"),
                       "/api/models/org/tiny?blobs=true": (200, json.dumps(full).encode()),
                       "/org/tiny/resolve/main/config.json": (200, b"{}"),
                       "/org/tiny/resolve/main/sub/w.bin": (200, b"abc")}
    hub = plugin("huggingface", {})
    assert hub.status == "available"  # no token needed for public repositories
    tiny, locked = hub.run("search", query="tiny")
    assert tiny.licence == "apache-2.0" and not tiny.restriction and "gated" in locked.restriction
    folder = fetch(hub, tiny, tmp_path)
    assert folder == tmp_path / "models" / "huggingface" / "org__tiny"
    assert (folder / "sub" / "w.bin").read_bytes() == b"abc" and (folder / "source.json").is_file()
    with pytest.raises(PluginError, match="gated"):
        fetch(hub, locked, tmp_path)


def test_gated_hub_repositories_are_refused(tmp_path) -> None:
    from plugins.huggingface.plugin import hub_item

    info = SimpleNamespace(id="meta/some-model", gated="manual", private=False, tags=["license:other"])
    item = hub_item(info, "model")
    assert "gated" in item.restriction and item.licence == "other"
    assert item.url == "https://huggingface.co/meta/some-model"
    public = SimpleNamespace(id="ds/x", gated=False, private=False, tags=["license:mit"])
    open_item = hub_item(public, "dataset")
    assert not open_item.restriction and open_item.url == "https://huggingface.co/datasets/ds/x"
    hub = plugin("huggingface", {})
    with pytest.raises(PluginError, match="gated"):
        hub.download(item, tmp_path, cancel=CancelToken(), progress=lambda *_: None)
    assert not (tmp_path / "models").exists()


def test_archives_cannot_escape_their_folder(tmp_path) -> None:
    archive = tmp_path / "evil.zip"
    archive.write_bytes(zipped({"../outside.txt": b"x"}))
    with pytest.raises(RemoteError, match="outside"):
        unpack_zip(archive, tmp_path / "into")
    assert not (tmp_path / "outside.txt").exists()
    assert RemoteItem("s", "a/b", "t", "dataset", "u", size_bytes=2_500_000).size_text == "2.5 MB"
