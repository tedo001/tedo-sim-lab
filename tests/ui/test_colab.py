"""Google Colab round trip: the notebook's own dataset and training cells run here (with
Colab's /content/lab swapped for a temporary folder), its results zip imports as a finished run."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import zipfile

import pytest

from core.common.paths import CODE_ROOT
from plugins.colab.notebook import EVENTS_FILE, read_results


def cell_sources(notebook: dict) -> list[str]:
    return ["".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"]


def test_notebook_runs_and_its_results_import(ctx, qtbot, tmp_path) -> None:
    pytest.importorskip("sklearn")
    from labs.classical_ml.presets import CLASSICAL_PRESETS

    spec = CLASSICAL_PRESETS[0].make()  # iris, bundled with scikit-learn
    path = ctx.colab.export(spec)
    assert path.parent == ctx.paths.notebooks / "colab" and ctx.colab.notebooks() == [path]
    notebook = json.loads(path.read_text())
    cells = cell_sources(notebook)
    clone, _mkdir, write_spec, data, train, download = cells
    assert "core.experiment_engine.worker" in train and "\\\n" not in train
    assert "git clone" in clone and "tedo-sim-lab" in clone and "google.colab" in download
    assert write_spec.startswith("%%writefile ") and "iris" in write_spec
    remote = tmp_path / "colab"
    folder = write_spec.splitlines()[0].split()[1].rsplit("/", 1)[0].replace("/content/lab", str(remote))
    os.makedirs(folder)
    (remote / "experiments").mkdir(exist_ok=True)
    with open(f"{folder}/experiment.yaml", "w", encoding="utf-8") as file:
        file.write(write_spec.split("\n", 1)[1])
    env = {**os.environ, "PYTHONPATH": str(CODE_ROOT), "TEDO_LAB_MLFLOW": "0"}
    code = data.replace("/content/lab", str(remote))
    done = subprocess.run([sys.executable, "-c", code], cwd=CODE_ROOT, env=env, capture_output=True,
                          text=True, stdin=subprocess.DEVNULL, timeout=120)
    assert done.returncode == 0, done.stderr
    with open(f"{folder}/{EVENTS_FILE}", "w", encoding="utf-8") as events:
        command = [sys.executable, "-m", "core.experiment_engine.worker", folder]
        worker = subprocess.run(command, cwd=CODE_ROOT,
                                env={**env, "TEDO_LAB_WORKSPACE": str(remote)}, stdout=events,
                                stderr=subprocess.PIPE, stdin=subprocess.DEVNULL, text=True, timeout=180)
    assert worker.returncode == 0, worker.stderr
    archive = shutil.make_archive(str(tmp_path / "results"), "zip", folder)
    assert read_results(tmp_path / "results.zip").spec.name == spec.name

    with qtbot.waitSignal(ctx.colab.imported, timeout=60_000) as signal:
        ctx.colab.start_import(tmp_path / "results.zip")
    _job, run_id, error = signal.args
    assert run_id and not error, error
    view = ctx.experiments.view(run_id)
    assert view.status == "completed" and view.run_dir.is_relative_to(ctx.paths.experiments)
    assert (view.run_dir / "results.json").is_file() and (view.run_dir / "colab.json").is_file()
    assert ctx.store.latest_metrics(run_id)["test_acc"] > 0.5
    assert archive and (view.run_dir / "run.log").read_text()  # log replayed
    assert not [p for p in ctx.paths.experiments.iterdir() if p.name.startswith("colab-")]  # staging removed


def test_import_refuses_unfinished_or_foreign_zips(ctx, qtbot, tmp_path) -> None:
    foreign = tmp_path / "foreign.zip"
    with zipfile.ZipFile(foreign, "w") as bundle:
        bundle.writestr("notes.txt", "hello")
    with qtbot.waitSignal(ctx.colab.imported, timeout=30_000) as signal:
        ctx.colab.start_import(foreign)
    assert signal.args[1] == "" and "experiment.yaml" in signal.args[2]
    unfinished = tmp_path / "unfinished.zip"
    with zipfile.ZipFile(unfinished, "w") as bundle:
        bundle.writestr("run/experiment.yaml", "name: x\n")
        bundle.writestr(f"run/{EVENTS_FILE}", '{"type": "log", "payload": {"line": "hi"}}\n')
    with pytest.raises(ValueError):
        read_results(unfinished)
