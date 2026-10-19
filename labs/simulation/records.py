"""Saved simulations: each is a folder in ``<workspace>/results/simulations/`` holding
``simulation.json`` (what it was, when, and every setting needed to run it again) and its outputs
(images, tables). The Simulation Lab lists them and opens any of them again."""

from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

__all__ = ["KINDS", "SavedSimulation", "delete", "list_saved", "load", "save", "simulations_folder"]

KINDS = {"circuit": "Quantum circuit", "waves": "Wave mechanics", "lattice": "Electrons in a lattice",
         "image": "Image pipeline", "scenes": "Synthetic scenes", "camera": "Camera model"}
FILE = "simulation.json"


@dataclass(frozen=True)
class SavedSimulation:
    folder: Path
    kind: str
    name: str
    created: str

    @property
    def title(self) -> str:
        return KINDS.get(self.kind, self.kind)


def simulations_folder(results: Path) -> Path:
    return Path(results) / "simulations"


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "-", text.strip()).strip("-").lower()[:40] or "simulation"


def save(results: Path, kind: str, name: str, settings: dict,
         outputs: dict[str, bytes | str] | None = None) -> Path:
    """A new folder with the settings and outputs (file name → bytes or text); returns it."""
    if kind not in KINDS:
        raise ValueError(f"unknown simulation kind {kind!r}")
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    base = simulations_folder(results) / f"{kind}-{stamp}-{_slug(name)}"
    folder, number = base, 2
    while folder.exists():
        folder = base.with_name(f"{base.name}-{number}")
        number += 1
    folder.mkdir(parents=True)
    record = {"kind": kind, "name": name.strip() or KINDS[kind],
              "created": datetime.now().isoformat(timespec="seconds"), "settings": settings}
    (folder / FILE).write_text(json.dumps(record, indent=2, default=_plain), encoding="utf-8")
    for file_name, content in (outputs or {}).items():
        target = folder / Path(file_name).name  # outputs stay inside the folder
        if isinstance(content, bytes):
            target.write_bytes(content)
        else:
            target.write_text(content, encoding="utf-8")
    return folder


def _plain(value):
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, (set, tuple)):
        return list(value)
    raise TypeError(f"{type(value).__name__} cannot be saved")


def load(folder: Path) -> dict:
    record = json.loads((Path(folder) / FILE).read_text(encoding="utf-8"))
    if record.get("kind") not in KINDS or not isinstance(record.get("settings"), dict):
        raise ValueError(f"{folder} is not a saved simulation")
    return record


def list_saved(results: Path) -> list[SavedSimulation]:
    """Every readable saved simulation, newest first."""
    found = []
    folder = simulations_folder(results)
    for candidate in folder.iterdir() if folder.is_dir() else ():
        try:
            record = load(candidate)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        found.append(SavedSimulation(candidate, record["kind"], record.get("name", ""),
                                     record.get("created", "")))
    return sorted(found, key=lambda s: s.created, reverse=True)


def delete(saved: SavedSimulation, results: Path) -> None:
    """Remove a saved simulation's folder (only folders inside the simulations folder)."""
    root = simulations_folder(results).resolve()
    target = saved.folder.resolve()
    if target.parent != root or not (target / FILE).is_file():
        raise ValueError(f"{saved.folder} is not a saved simulation")
    shutil.rmtree(target)
