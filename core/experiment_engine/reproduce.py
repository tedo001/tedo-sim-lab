"""Reproduce a run from its bundle, in a workspace of your choice (a clean one is the real test):

    python -m core.experiment_engine.reproduce <bundle.zip | reproduction folder | run folder>
                                               --workspace <folder> [--download] [--tolerance 1e-9]

It puts the bundle's ``experiment.yaml`` in a new run folder under
``<workspace>/experiments/reproductions/``, downloads the dataset when asked and allowed (only
datasets anyone may fetch without accepting terms), runs it exactly as the lab does, then compares
every recorded score with the new one. It prints one JSON line with the verdict and exits 0 when
every score matches within ``--tolerance`` (absolute), 3 when some differ, 1 when the run failed.
The environment differences (the bundle's snapshot against this Python) are listed too."""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Any

from core.common.cancel import CancelToken
from core.common.paths import CODE_ROOT, WORKSPACE_ENV, AppPaths

from .events import JsonLinesCallbacks
from .repro import RESULT_FILE, TIMING
from .snapshot import compare, environment
from .spec import load_spec_text

__all__ = ["EXIT_DIFFERENT", "compare_scores", "main", "reproduce"]

EXIT_DIFFERENT = 3


def _read_bundle(source: Path) -> dict[str, bytes]:
    """File name → bytes of the bundle (a zip, a reproduction folder or a run folder)."""
    if source.is_file() and zipfile.is_zipfile(source):
        with zipfile.ZipFile(source) as archive:
            return {Path(name).name: archive.read(name) for name in archive.namelist()
                    if not name.endswith("/")}
    folder = source / "reproduction" if (source / "reproduction" / "experiment.yaml").is_file() else source
    if not (folder / "experiment.yaml").is_file():
        raise FileNotFoundError(f"{source} is not a reproduction bundle (no experiment.yaml)")
    return {path.name: path.read_bytes() for path in folder.iterdir() if path.is_file()}


def compare_scores(recorded: dict[str, Any], now: dict[str, Any], tolerance: float) -> list[dict[str, Any]]:
    """Each recorded score beside the new one; ``match`` within ``tolerance`` (absolute)."""
    rows = []
    for key, before in sorted(recorded.items()):
        if not isinstance(before, int | float) or any(key.endswith(t) for t in TIMING):
            continue
        after = now.get(key)
        same = isinstance(after, int | float) and abs(float(after) - float(before)) <= tolerance
        rows.append({"metric": key, "recorded": float(before),
                     "reproduced": None if after is None else float(after), "match": bool(same)})
    return rows


def _prepare(spec: Any, paths: AppPaths, download: bool) -> str | None:
    """Make the dataset available; a sentence when it cannot be (nothing is fetched unless asked)."""
    from core.catalog import load_cards

    datasets, _models = load_cards(paths)
    if spec.data.dataset not in datasets:
        return f"no dataset card {spec.data.dataset!r} in this checkout"
    try:
        adapter = datasets.adapter(spec.data.dataset)
    except NotImplementedError as exc:
        return str(exc)
    if adapter.local_status(paths.datasets) == "present":
        return None
    if not download:
        return f"{adapter.card.name} is not in {paths.datasets}; add --download to fetch it (licence: " \
               f"{adapter.card.license.name})"
    adapter.prepare(paths.datasets, lambda fraction, message: None, CancelToken())  # refuses gated / terms
    return None


def reproduce(source: Path, workspace: Path, *, download: bool = False, tolerance: float = 1e-9,
              events: io.TextIOBase | None = None) -> dict[str, Any]:
    from core.tracking import new_id

    from .worker import run_experiment

    files = _read_bundle(source)
    spec = load_spec_text(files["experiment.yaml"].decode("utf-8"))
    recorded = json.loads(files[RESULT_FILE]) if RESULT_FILE in files else {}
    snapshot = json.loads(files["snapshot.json"]) if "snapshot.json" in files else {}
    os.environ[WORKSPACE_ENV] = str(workspace)
    paths = AppPaths.resolve(workspace).ensure()
    run_dir = paths.experiments / "reproductions" / f"{spec.name}-{new_id()[:8]}" / new_id()
    run_dir.mkdir(parents=True)
    (run_dir / "experiment.yaml").write_bytes(files["experiment.yaml"])
    note = {"source": str(source), "recorded": recorded}
    (run_dir / "reproduces.json").write_text(json.dumps(note, indent=2), encoding="utf-8")
    differences = (compare(snapshot, environment(CODE_ROOT)) if snapshot
                   else ["the bundle has no snapshot.json"])
    verdict: dict[str, Any] = {"run_dir": str(run_dir), "name": spec.name,
                               "environment_differences": differences}
    problem = _prepare(spec, paths, download)
    if problem:
        return {**verdict, "status": "failed", "error": problem, "scores": [], "reproduced": False}
    callbacks = JsonLinesCallbacks(events or io.StringIO(), run_dir.name)
    result = run_experiment(run_dir, callbacks, CancelToken())
    scores = compare_scores(recorded.get("metrics", {}), result.metrics, tolerance)
    return {**verdict, "status": result.status, "error": result.error, "scores": scores,
            "reproduced": result.status in ("completed", "early_stopped") and bool(scores)
            and all(row["match"] for row in scores)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="core.experiment_engine.reproduce")
    parser.add_argument("source", type=Path)
    parser.add_argument("--workspace", type=Path, default=None,
                        help="where to run it (default: a new temporary folder)")
    parser.add_argument("--download", action="store_true", help="download the dataset when its terms allow")
    parser.add_argument("--tolerance", type=float, default=1e-9)
    args = parser.parse_args(argv)
    workspace = args.workspace or Path(tempfile.mkdtemp(prefix="tedo-reproduce-"))
    verdict = reproduce(args.source.resolve(), workspace.resolve(), download=args.download,
                        tolerance=args.tolerance, events=sys.stderr)
    print(json.dumps(verdict))
    if verdict["status"] not in ("completed", "early_stopped"):
        return 1
    return 0 if verdict["reproduced"] else EXIT_DIFFERENT


if __name__ == "__main__":
    sys.exit(main())

