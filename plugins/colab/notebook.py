"""A Colab notebook that runs one experiment with the lab's own worker, and the reader for the
results zip it produces.

The notebook clones this repository at a commit, writes the experiment's ``experiment.yaml``,
downloads the dataset only if its card allows it (and, when its licence asks, only after the
person sets ``ACCEPT_TERMS = True``), runs ``python -m core.experiment_engine.worker`` with its
event stream saved to ``events.jsonl``, and zips the run folder. Importing the zip replays those
events into the lab, exactly as if the run had happened here.
"""

from __future__ import annotations

import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from core.common.licensing import download_policy
from core.experiment_engine.events import ProgressEvent, parse_line
from core.experiment_engine.spec import ExperimentSpec, dump_spec_text, load_spec_text

__all__ = ["EVENTS_FILE", "ColabResults", "build_notebook", "read_results", "summary"]

EVENTS_FILE = "events.jsonl"
#: Packages the worker needs beyond what Colab ships (it has PyTorch, numpy, pandas, scikit-learn).
COLAB_PACKAGES = ("pydantic>=2.6", "PyYAML>=6.0", "packaging>=23", "psutil>=5.9", "keyring>=24",
                  "GitPython>=3.1.40")
#: Files never imported from a zip (the worker's own state, not results).
_SKIP = {".lock"}


def _markdown(text: str) -> dict[str, Any]:
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip().splitlines(keepends=True)}


def _code(text: str) -> dict[str, Any]:
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": text.strip().splitlines(keepends=True)}


def build_notebook(spec: ExperimentSpec, *, dataset: Any, repository: str, revision: str,
                   run_id: str) -> dict[str, Any]:
    """The notebook as nbformat 4 JSON. ``dataset`` is the spec's :class:`DatasetCard`."""
    spec_text = dump_spec_text(spec)
    decision = download_policy(dataset)
    folder = f"/content/lab/experiments/{run_id}"
    licence = dataset.license
    if not decision.allowed:
        data_note = (f"**{dataset.name} is not downloaded by this notebook.** {decision.reason} "
                     f"Put it under `/content/lab/datasets/{dataset.id}/` yourself before running.")
    elif decision.acknowledgement:
        data_note = (f"{dataset.name} is downloaded from its source under **{licence.name}**. Read the "
                     f"terms ({licence.url or dataset.source_url}), then set `ACCEPT_TERMS = True` in "
                     f"the cell below:\n\n> {decision.acknowledgement}")
    else:
        data_note = f"{dataset.name} is downloaded from its source under **{licence.name}**."
    steps = [
        _markdown(f"""
# {spec.name} on Google Colab

Written by TEDO AI Research Lab. It trains **{spec.model.model}** on **{dataset.name}** with the lab's
own worker, then zips the run folder so you can import it back (Google Colab page in the lab →
*Import results*).

For a GPU: *Runtime → Change runtime type → GPU*. Colab's terms of service and usage limits apply.
"""),
        _markdown("## 1. The lab's code\nThe repository at the commit this notebook was made from. If that "
                  "commit was never pushed, change `REVISION` to one that was."),
        _code(f"""
REPOSITORY = {repository!r}
REVISION = {revision!r}
!git clone --quiet {{REPOSITORY}} /content/tedo-sim-lab
%cd /content/tedo-sim-lab
!git checkout --quiet {{REVISION}}
!pip install --quiet {" ".join(repr(p) for p in COLAB_PACKAGES)}
"""),
        _markdown("## 2. The experiment\nThe same `experiment.yaml` the lab runs."),
        _code(f"!mkdir -p {folder}"),
        _code(f"%%writefile {folder}/experiment.yaml\n{spec_text}"),
        _markdown(f"## 3. The dataset\n{data_note}"),
        _code(f"""
ACCEPT_TERMS = False  # set to True only after reading the dataset's terms (see above)
import os
os.environ["TEDO_LAB_WORKSPACE"] = "/content/lab"
from core.common import AppPaths, CredentialStore
from core.common.cancel import CancelToken
from core.catalog import load_catalog
paths = AppPaths.resolve("/content/lab").ensure()
catalog = load_catalog(paths, CredentialStore(env={{}}, backend=None))
catalog.datasets.adapter({dataset.id!r}).prepare(paths.datasets, lambda fraction, message: print(message),
                                                  CancelToken(), acknowledged=ACCEPT_TERMS)
"""),
        _markdown("## 4. Train\nProgress is printed below; the event stream is kept in `events.jsonl`."),
        _code(f"""
%env TEDO_LAB_WORKSPACE=/content/lab
%env TEDO_LAB_MLFLOW=0
RUN = {folder!r}
!python -m core.experiment_engine.worker {{RUN}} < /dev/null > {{RUN}}/{EVENTS_FILE}
!tail -n 3 {{RUN}}/{EVENTS_FILE}
"""),
        _markdown("## 5. Take the results home\nDownloads a zip; import it on the lab's Google Colab page."),
        _code(f"""
import shutil
from google.colab import files
archive = shutil.make_archive("/content/{run_id}", "zip", {folder!r})
files.download(archive)
"""),
    ]
    return {"nbformat": 4, "nbformat_minor": 5, "cells": steps,
            "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                         "language_info": {"name": "python"}, "colab": {"provenance": []},
                         "accelerator": "GPU", "tedo": {"run_id": run_id, "experiment": spec.name}}}


@dataclass(frozen=True)
class ColabResults:
    spec: ExperimentSpec
    spec_text: str
    events: list[ProgressEvent]
    #: Archive member name → path inside the run folder, for every file to copy.
    files: dict[str, PurePosixPath]


def read_results(archive: Path) -> ColabResults:
    """Check a results zip (``experiment.yaml`` and ``events.jsonl`` at its top level, or inside
    one folder) and read it. Raises ``ValueError`` with a readable reason."""
    try:
        bundle = zipfile.ZipFile(archive)
    except (OSError, zipfile.BadZipFile) as exc:
        raise ValueError(f"{archive.name} is not a zip file: {exc}") from None
    with bundle:
        names = [n for n in bundle.namelist() if not n.endswith("/")]
        spec_members = [n for n in names if PurePosixPath(n).name == "experiment.yaml"]
        if len(spec_members) != 1:
            raise ValueError("the zip must hold exactly one run folder (one experiment.yaml)")
        base = PurePosixPath(spec_members[0]).parent
        files: dict[str, PurePosixPath] = {}
        for name in names:
            path = PurePosixPath(name.replace("\\", "/"))
            if path.is_absolute() or ".." in path.parts:
                raise ValueError(f"refusing {name!r}: it points outside the run folder")
            if base.parts and path.parts[:len(base.parts)] != base.parts:
                continue
            inner = PurePosixPath(*path.parts[len(base.parts):])
            if inner.name not in _SKIP:
                files[name] = inner
        events_member = next((n for n, inner in files.items() if str(inner) == EVENTS_FILE), None)
        if events_member is None:
            raise ValueError(f"the zip has no {EVENTS_FILE}: was it made by the lab's Colab notebook?")
        spec_text = bundle.read(spec_members[0]).decode("utf-8")
        spec = load_spec_text(spec_text)
        lines = bundle.read(events_member).decode("utf-8", errors="replace").splitlines()
    events = [parse_line(line) for line in lines if line.strip()]
    if not any(event.type == "run_end" for event in events):
        raise ValueError("the run did not finish on Colab (no run_end event); nothing to import")
    return ColabResults(spec, spec_text, events, files)


def summary(results: ColabResults) -> dict[str, Any]:
    end = next(e for e in reversed(results.events) if e.type == "run_end").payload
    return {"status": end.get("status"), "metrics": end.get("metrics", {}), "files": len(results.files),
            "events": len(results.events), "name": results.spec.name}
