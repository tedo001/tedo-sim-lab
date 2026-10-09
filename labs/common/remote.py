"""What the dataset and model source plugins (Kaggle, Hugging Face, Roboflow) share: one item
found at a source, JSON requests with masked errors, safe unpacking, and the ``source.json``
note that records where a download came from and under which licence."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
import zipfile
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any

from core.common.masking import mask_text

__all__ = ["RemoteError", "RemoteItem", "get_json", "safe_folder_name", "unpack_zip", "write_source"]

SOURCE_FILE = "source.json"


class RemoteError(RuntimeError):
    """A source refused or failed a request; the message is safe to show."""


@dataclass(frozen=True)
class RemoteItem:
    """A dataset or model at a source, with what a person must know before downloading it."""

    source: str
    id: str
    title: str
    kind: str  # "dataset" or "model"
    url: str
    #: The licence as the source states it; "" when the source states none.
    licence: str = ""
    size_bytes: int | None = None
    #: Why the lab will not download it ("" = it may be downloaded on request).
    restriction: str = ""
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def licence_text(self) -> str:
        return self.licence or "No licence stated (unspecified)"

    @property
    def size_text(self) -> str:
        if self.size_bytes is None:
            return "—"
        for unit, scale in (("GB", 1e9), ("MB", 1e6), ("KB", 1e3)):
            if self.size_bytes >= scale:
                return f"{self.size_bytes / scale:.1f} {unit}"
        return f"{self.size_bytes} B"


def get_json(url: str, *, headers: Mapping[str, str] | None = None, timeout: float = 30.0) -> Any:
    """GET ``url`` and parse JSON. Errors name the HTTP status, never the credentials."""
    request = urllib.request.Request(url, headers={"User-Agent": "tedo-ai-research-lab",
                                                   "Accept": "application/json", **(headers or {})})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        if exc.code in (401, 403):
            raise RemoteError(f"the source refused the request (HTTP {exc.code}): check the "
                              "credentials, and that you accepted the item's terms on its page") from None
        if exc.code == 404:
            raise RemoteError("not found (HTTP 404)") from None
        raise RemoteError(mask_text(f"HTTP {exc.code}: {exc.reason}")) from None
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        reason = getattr(exc, "reason", exc)
        raise RemoteError(mask_text(f"could not reach the source: {reason}")) from None
    except ValueError:
        raise RemoteError("the source answered with something that is not JSON") from None


def safe_folder_name(identifier: str) -> str:
    """``owner/name`` → ``owner__name``: one folder, nothing that climbs out of it."""
    cleaned = "".join(c if c.isalnum() or c in "-_." else "_" for c in identifier.replace("/", "__"))
    cleaned = cleaned.strip(".")
    if not cleaned:
        raise ValueError(f"{identifier!r} cannot be used as a folder name")
    return cleaned


def unpack_zip(archive: Path, destination: Path) -> list[Path]:
    """Extract every member inside ``destination``; refuses a member that would land outside."""
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    written = []
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.infolist():
            name = PurePosixPath(member.filename.replace("\\", "/"))
            target = (destination / Path(*name.parts)).resolve() if name.parts else root
            if name.is_absolute() or ".." in name.parts or not target.is_relative_to(root):
                raise RemoteError(f"refusing to unpack {member.filename!r}: it points outside the folder")
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with bundle.open(member) as source, target.open("wb") as out:
                while block := source.read(1 << 20):
                    out.write(block)
            written.append(target)
    return written


def write_source(folder: Path, item: RemoteItem) -> Path:
    """Record where ``folder`` came from: source, id, page, licence, time."""
    note = {**asdict(item), "downloaded_at": datetime.now().astimezone().isoformat(timespec="seconds")}
    path = folder / SOURCE_FILE
    path.write_text(json.dumps(note, indent=2, default=str), encoding="utf-8")
    return path
