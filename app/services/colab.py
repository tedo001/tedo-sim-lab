"""Google Colab from the lab's side: write an experiment as a Colab notebook into the
workspace's ``notebooks/colab/`` folder, and bring a finished Colab run back (the zip its last
cell downloads) as an ordinary run of this workspace."""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path

from PySide6.QtCore import QObject, Signal

from core.common import AppPaths
from core.experiment_engine.spec import ExperimentSpec
from core.tracking import new_id
from plugins.colab.notebook import ColabResults, build_notebook, read_results

from .experiments import ExperimentService
from .jobs import JobQueue
from .runs import slug

__all__ = ["DEFAULT_REPOSITORY", "ColabService", "code_revision"]

DEFAULT_REPOSITORY = "https://github.com/tedo001/tedo-sim-lab"


def _git(code_root: Path, *args: str) -> str:
    try:
        result = subprocess.run(("git", "-C", str(code_root), *args), capture_output=True, text=True,
                                stdin=subprocess.DEVNULL, timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return result.stdout.strip() if result.returncode == 0 else ""


def code_revision(code_root: Path) -> tuple[str, str]:
    """(repository URL, commit) of the lab's code; the public repository and its ``tedo`` branch
    when the code is not a git checkout (an installed app)."""
    remote = _git(code_root, "remote", "get-url", "origin")
    if remote.startswith("git@github.com:"):
        remote = "https://github.com/" + remote.removeprefix("git@github.com:")
    if "@" in remote.split("//", 1)[-1].split("/", 1)[0]:  # never write credentials into a notebook
        remote = ""
    return remote.removesuffix(".git") or DEFAULT_REPOSITORY, _git(code_root, "rev-parse", "HEAD") or "tedo"


class ColabService(QObject):
    imported = Signal(str, str, str)  # job id, run id ("" if it failed), error

    def __init__(self, paths: AppPaths, experiments: ExperimentService, jobs: JobQueue,
                 parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.paths, self.experiments, self.jobs = paths, experiments, jobs
        self._staged: dict[str, Path] = {}
        jobs.job_finished.connect(self._finished)

    @property
    def folder(self) -> Path:
        return self.paths.notebooks / "colab"

    def export(self, spec: ExperimentSpec) -> Path:
        """Write the notebook; returns its path."""
        dataset = self.experiments.catalog.datasets.get(spec.data.dataset)
        repository, revision = code_revision(self.paths.code_root)
        run_id = new_id()
        notebook = build_notebook(spec, dataset=dataset, repository=repository, revision=revision,
                                  run_id=run_id)
        self.folder.mkdir(parents=True, exist_ok=True)
        path = self.folder / f"{slug(spec.name)}-{run_id[:8]}.ipynb"
        path.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
        return path

    def notebooks(self) -> list[Path]:
        return sorted(self.folder.glob("*.ipynb"), key=lambda p: p.stat().st_mtime, reverse=True)

    def start_import(self, archive: Path) -> str:
        """Check and unpack ``archive`` in the background; the run is recorded when it is done
        (``imported`` signal). Returns the job id."""
        def work(cancel, progress) -> tuple[ColabResults, Path]:
            progress(-1.0, f"Reading {archive.name}")
            results = read_results(archive)
            staging = Path(tempfile.mkdtemp(prefix="colab-", dir=self.paths.experiments))
            with zipfile.ZipFile(archive) as bundle:
                for member, inner in results.files.items():
                    cancel.raise_if_cancelled()
                    target = staging.joinpath(*inner.parts)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with bundle.open(member) as source, target.open("wb") as out:
                        shutil.copyfileobj(source, out)
            return results, staging

        job_id = self.jobs.submit_task(work, title=f"Import Colab results {archive.name}")
        return job_id

    def _finished(self, job_id: str, status: str) -> None:
        job = self.jobs.job(job_id)
        if job.kind != "task" or not job.title.startswith("Import Colab results"):
            return
        if status != "completed":
            self.imported.emit(job_id, "", job.error or status)
            return
        results, staging = job.result
        try:
            run_id = self.experiments.import_run(results.spec, results.events,
                                                 lambda run_dir: self._move(staging, run_dir))
        except Exception as exc:  # a spec this lab cannot run, a full disk: say so
            self.imported.emit(job_id, "", f"{type(exc).__name__}: {exc}")
            return
        finally:
            shutil.rmtree(staging, ignore_errors=True)
        self.imported.emit(job_id, run_id, "")

    @staticmethod
    def _move(staging: Path, run_dir: Path) -> None:
        for item in staging.iterdir():
            shutil.move(str(item), run_dir / item.name)
        note = json.dumps({"source": "Google Colab"}, indent=2)
        (run_dir / "colab.json").write_text(note, encoding="utf-8")
