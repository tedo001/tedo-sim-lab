"""Notebooks in the workspace's ``notebooks/`` folder: list, create (optionally starting from a
run's results), link to runs, read for the viewer; and Jupyter Lab itself, installed into and
run by the experiment Python, bound to 127.0.0.1 with a fresh access token each start."""

from __future__ import annotations

import json
import os
import secrets
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core.common import AppPaths, experiment_python
from core.common.masking import register_secret
from core.tracking import LabStore

from .runs import RunView, slug
from .web_ui import LocalWebUi

__all__ = ["JUPYTER_REQUIREMENTS", "JupyterLab", "NotebookFile", "Notebooks", "jupyter_version",
           "notebook_markdown"]

#: Installed on request into the experiment Python (both BSD-3-Clause).
JUPYTER_REQUIREMENTS = ("jupyterlab>=4.1", "ipykernel>=6.29")


def jupyter_version(python: str) -> str | None:
    """Jupyter Lab's version in ``python``'s environment, or ``None``. Blocking (a subprocess):
    call it from a task."""
    code = "import importlib.metadata as m; print(m.version('jupyterlab'))"
    try:
        result = subprocess.run((python, "-c", code), capture_output=True, text=True, timeout=30,
                                stdin=subprocess.DEVNULL)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return result.stdout.strip() or None if result.returncode == 0 else None


class JupyterLab(LocalWebUi):
    START_PORT = 8888

    def __init__(self, paths: AppPaths, config: Any, parent: Any = None) -> None:
        super().__init__(paths, config, parent)
        self.token = ""

    def prepare(self) -> None:
        self.paths.notebooks.mkdir(parents=True, exist_ok=True)
        self.token = secrets.token_urlsafe(24)
        register_secret(self.token)  # never in a log line

    def command(self, port: int) -> list[str]:
        return ["-m", "jupyterlab", "--no-browser", "--ServerApp.ip=127.0.0.1", f"--ServerApp.port={port}",
                "--ServerApp.port_retries=0", f"--ServerApp.root_dir={self.paths.notebooks}",
                f"--IdentityProvider.token={self.token}",
                *(["--allow-root"] if getattr(os, "geteuid", lambda: 1)() == 0 else [])]  # containers

    def link_for(self, notebook: Path | None = None) -> str | None:
        """The browser address (with the token) of Jupyter Lab, or of one notebook in it."""
        base = self.url
        if base is None:
            return None
        where = ""
        if notebook is not None:
            where = "/tree/" + notebook.relative_to(self.paths.notebooks).as_posix()
        return f"{base}/lab{where}?token={self.token}"

    def install_command(self) -> list[str]:
        python = experiment_python(self.config)
        if getattr(sys, "frozen", False) and python == sys.executable:
            raise RuntimeError("The installed app cannot add packages to itself. Choose a Python "
                               "environment for experiments in Settings first.")
        return [python, "-m", "pip", "install", *JUPYTER_REQUIREMENTS]


@dataclass(frozen=True)
class NotebookFile:
    path: Path
    relative: str
    modified: float
    runs: tuple[str, ...]


class Notebooks:
    def __init__(self, paths: AppPaths, store: LabStore) -> None:
        self.paths, self.store = paths, store

    def list(self) -> list[NotebookFile]:
        links: dict[str, list[str]] = {}
        for row in self.store.notebook_links():
            if row["run_id"]:
                links.setdefault(row["notebook_path"], []).append(row["run_id"])
        found = []
        for path in sorted(self.paths.notebooks.rglob("*.ipynb")):
            if ".ipynb_checkpoints" in path.parts:
                continue
            stored = self.store.stored_path(path)
            found.append(NotebookFile(path, path.relative_to(self.paths.notebooks).as_posix(),
                                      path.stat().st_mtime, tuple(links.get(stored, ()))))
        return sorted(found, key=lambda notebook: notebook.modified, reverse=True)

    def create(self, name: str, run: RunView | None = None) -> Path:
        """A new notebook; started from ``run`` it reads that run's metrics and links to it."""
        self.paths.notebooks.mkdir(parents=True, exist_ok=True)
        stem = slug(name or (run.name if run else "notebook"), 60)
        path = self.paths.notebooks / f"{stem}.ipynb"
        number = 2
        while path.exists():
            path, number = self.paths.notebooks / f"{stem}-{number}.ipynb", number + 1
        notebook = starter_notebook(name or stem, run, self.paths)
        path.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
        if run is not None:
            self.link(path, run)
        return path

    def link(self, path: Path, run: RunView) -> None:
        self.store.link_notebook(path, experiment_id=run.experiment_id, run_id=run.id)


def _cell(kind: str, text: str) -> dict[str, Any]:
    cell: dict[str, Any] = {"cell_type": kind, "metadata": {},
                            "source": text.strip().splitlines(keepends=True)}
    if kind == "code":
        cell.update(execution_count=None, outputs=[])
    return cell


def starter_notebook(title: str, run: RunView | None, paths: AppPaths) -> dict[str, Any]:
    cells = [_cell("markdown", f"# {title}\n\nA notebook in the TEDO AI Research Lab workspace.")]
    if run is not None:
        relative = Path(os.path.relpath(run.run_dir, paths.notebooks)).as_posix()
        cells += [
            _cell("markdown", f"## Run {run.id[:8]}: {run.name}\n\n{run.dataset} · {run.model} · "
                              f"{run.status}. The path below is relative, so it keeps working if the "
                              "project folder moves."),
            _cell("code", f"""
import json
from pathlib import Path

import pandas as pd

run = Path({relative!r})
print((run / "experiment.yaml").read_text())
log = run / "metrics.jsonl"
metrics = pd.read_json(log, lines=True) if log.exists() else pd.DataFrame()
metrics.tail()
"""),
            _cell("code", """
curves = [c for c in ("train_loss", "val_loss", "train_acc", "val_acc") if c in metrics]
if curves and "epoch" in metrics:
    metrics.plot(x="epoch", y=curves, subplots=True, figsize=(7, 2 * len(curves)))
"""),
        ]
    else:
        cells.append(_cell("code", "import numpy as np\nimport pandas as pd"))
    return {"nbformat": 4, "nbformat_minor": 5, "cells": cells,
            "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                         "language_info": {"name": "python"}}}


def notebook_markdown(path: Path, max_output: int = 2_000) -> str:
    """A notebook as Markdown for the viewer: text cells as they are, code in fenced blocks,
    text outputs after them (images and HTML outputs are named, not drawn)."""
    try:
        notebook = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return f"*Could not read {path.name}: {exc}*"
    language = (notebook.get("metadata", {}).get("language_info") or {}).get("name", "python")
    parts = []
    for cell in notebook.get("cells", []):
        source = cell.get("source", "")
        source = "".join(source) if isinstance(source, list) else str(source)
        if cell.get("cell_type") == "markdown":
            parts.append(source)
        elif cell.get("cell_type") == "code":
            parts.append(f"```{language}\n{source}\n```")
            for output in cell.get("outputs", []):
                text = output.get("text") or (output.get("data") or {}).get("text/plain") or ""
                text = "".join(text) if isinstance(text, list) else str(text)
                kinds = sorted(set(output.get("data") or {}) - {"text/plain"})
                if output.get("output_type") == "error":
                    text = f"{output.get('ename', 'Error')}: {output.get('evalue', '')}"
                if text:
                    parts.append(f"```text\n{text[:max_output]}\n```")
                if kinds:
                    parts.append(f"*Output shown in Jupyter: {', '.join(kinds)}*")
    return "\n\n".join(parts) or "*This notebook is empty.*"
