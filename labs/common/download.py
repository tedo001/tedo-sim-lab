"""Downloads the lab starts on request: streamed to a temporary file, checked against an
MD5 sum when the source publishes one, reported as progress, and cancellable between
chunks. A failed or cancelled download leaves nothing half-written behind."""

from __future__ import annotations

import hashlib
import shutil
import urllib.request
from collections.abc import Sequence
from pathlib import Path

from core.common.cancel import CancelToken, ProgressFn

__all__ = ["DownloadError", "download_file", "md5_of"]

_CHUNK = 1 << 16
_USER_AGENT = "tedo-ai-research-lab"


class DownloadError(RuntimeError):
    """Every mirror failed, or the file did not match its checksum."""


def md5_of(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _fetch(url: str, partial: Path, progress: ProgressFn, cancel: CancelToken, label: str,
           timeout: float) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as response, partial.open("wb") as out:
        total = int(response.headers.get("Content-Length") or 0)
        done = 0
        while True:
            cancel.raise_if_cancelled()
            chunk = response.read(_CHUNK)
            if not chunk:
                break
            out.write(chunk)
            done += len(chunk)
            fraction = done / total if total else -1.0
            of = f" of {total / 1e6:.1f} MB" if total else ""
            progress(fraction, f"{label}: {done / 1e6:.1f} MB{of}")


def download_file(urls: Sequence[str], destination: Path, *, md5: str | None = None,
                  progress: ProgressFn, cancel: CancelToken, timeout: float = 60.0) -> Path:
    """Fetch the first mirror that works into ``destination``. Skips the download when the file
    is already there and matches ``md5``."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and (md5 is None or md5_of(destination) == md5):
        return destination
    partial = destination.with_name(destination.name + ".part")
    errors: list[str] = []
    for url in urls:
        try:
            _fetch(url, partial, progress, cancel, destination.name, timeout)
            if md5 is not None and md5_of(partial) != md5:
                raise DownloadError(f"{destination.name} from {url} does not match its published checksum")
            shutil.move(str(partial), destination)
            return destination
        except (OSError, DownloadError) as exc:
            errors.append(f"{url}: {exc}")
        finally:
            partial.unlink(missing_ok=True)
    raise DownloadError(f"could not download {destination.name}:\n  " + "\n  ".join(errors))
