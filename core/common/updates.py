"""Is there a newer release of the lab? Release tags are compared as version numbers
(``v0.10.0`` is newer than ``v0.9.2``); the check itself asks GitHub, only when the person asks
or has turned on checking at start."""

from __future__ import annotations

import re

__all__ = ["RELEASES", "is_newer", "version_key"]

#: Where the lab's releases are published (owner, repository).
RELEASES = ("tedo001", "tedo-sim-lab")


def version_key(text: str) -> tuple[int, ...]:
    """The numbers of a version or tag: ``"v0.2.1-beta"`` → ``(0, 2, 1)``; ``()`` if it has none."""
    match = re.search(r"\d+(?:\.\d+)*", text or "")
    return tuple(int(part) for part in match.group(0).split(".")) if match else ()


def is_newer(tag: str, current: str) -> bool:
    """``tag`` names a later version than ``current`` (missing parts count as 0)."""
    new, old = version_key(tag), version_key(current)
    if not new:
        return False
    width = max(len(new), len(old))
    return new + (0,) * (width - len(new)) > old + (0,) * (width - len(old))
