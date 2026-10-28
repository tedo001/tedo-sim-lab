"""Folder names made from identifiers that come from elsewhere (``owner/name`` of a Hub repository,
a Kaggle dataset, ...)."""

from __future__ import annotations

__all__ = ["safe_folder_name"]


def safe_folder_name(identifier: str) -> str:
    """``owner/name`` → ``owner__name``: one folder, nothing that climbs out of it."""
    cleaned = "".join(c if c.isalnum() or c in "-_." else "_" for c in identifier.replace("/", "__"))
    cleaned = cleaned.strip(".")
    if not cleaned:
        raise ValueError(f"{identifier!r} cannot be used as a folder name")
    return cleaned
