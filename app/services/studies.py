"""Seed studies and sweeps from the interface's side: queue one (every run an ordinary run), find
its runs again, and say what they show: per-metric mean, spread and 95 % interval over seeds,
Welch's test between two studies with their compute budgets checked, and, for a sweep, each run's
parameters beside its score with how much each parameter matters (random-forest importance) and
how it correlates with the score. Exports carry every number's run, seed, spec hash, budget and the
formula behind it."""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from core.common import AppPaths
from core.experiment_engine import studies
from core.experiment_engine.spec import ExperimentSpec, spec_hash
from core.experiment_engine.stats import summarize, welch
from core.tracking import LabStore, new_id

from .experiments import ExperimentService
from .runs import RunView

__all__ = ["StudyService", "TIMING"]

TIMING = ("epoch_time_s", "lr", "gpu_mem_gb", "duration_s")
SUCCEEDED = ("completed", "early_stopped")


class StudyService:
    def __init__(self, paths: AppPaths, store: LabStore, experiments: ExperimentService) -> None:
        self.paths, self.store, self.experiments = paths, store, experiments

    # Launching ----------------------------------------------------------------------
    def _launch_all(self, specs: list[ExperimentSpec]) -> list[str]:
        problems = self.experiments.problems(specs[0])
        if problems:
            raise ValueError("; ".join(problems))
        return [self.experiments.launch(spec) for spec in specs]

    def launch_study(self, spec: ExperimentSpec, seeds: list[int]) -> str:
        """Queue ``spec`` once per seed; returns the study id."""
        study_id = new_id()[:10]
        specs = studies.seed_study(spec, seeds, study_id)
        runs = self._launch_all(specs)
        studies.write_definition(self.paths.results, study_id, "seeds", spec, seeds=list(seeds), runs=runs,
                                 spec_hash=spec_hash(spec.model_copy(update={"seed": 0})))
        return study_id

    def launch_sweep(self, spec: ExperimentSpec, space: dict[str, list[Any]], *, method: str = "grid",
                     count: int | None = None) -> str:
        sweep_id = new_id()[:10]
        specs = studies.sweep(spec, space, sweep_id, method=method, count=count, seed=spec.seed)
        runs = self._launch_all(specs)
        studies.write_definition(self.paths.results, sweep_id, "sweep", spec, space=space, method=method,
                                 count=len(specs), runs=runs)
        return sweep_id

    # Reading ------------------------------------------------------------------------
    def groups(self, kind: str | None = None) -> list[dict[str, Any]]:
        folder = self.paths.results / studies.STUDIES
        found = []
        for path in sorted(folder.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True) \
                if folder.is_dir() else []:
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if kind is None or data.get("kind") == kind:
                found.append(data)
        return found

    def group(self, group_id: str) -> dict[str, Any]:
        path = self.paths.results / studies.STUDIES / f"{group_id}.json"
        return json.loads(path.read_text(encoding="utf-8"))

    def runs(self, group_id: str) -> list[RunView]:
        tags = {f"study:{group_id}", f"sweep:{group_id}"}
        return [view for view in self.experiments.views(limit=5000)
                if view.spec is not None and tags & set(view.spec.tags)]

    def metrics(self, view: RunView) -> dict[str, float]:
        return {key: float(value) for key, value in self.store.latest_metrics(view.id).items()
                if not key.endswith(TIMING)}

    def budget(self, view: RunView) -> dict[str, Any]:
        info_file = view.run_dir / "run_info.json"
        info = {}
        if info_file.is_file():
            try:
                info = json.loads(info_file.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                info = {}
        info["epochs_run"] = len(self.store.metric_history(view.id, "train_loss")) or view.epochs
        return studies.budget(view.spec, info) if view.spec else {}

    def rows(self, group_id: str) -> list[dict[str, Any]]:
        """One row per run: id, status, seed, swept values, duration, budget and scores."""
        definition = self.group(group_id)
        paths = list(definition.get("space", {}))
        out = []
        for view in sorted(self.runs(group_id), key=lambda v: v.created_at):
            data = view.spec.model_dump(mode="json")
            out.append({"run_id": view.id, "name": view.name, "status": view.status, "seed": view.spec.seed,
                        "spec_hash": spec_hash(view.spec), "device": view.device or "",
                        "duration_s": view.duration_s,
                        "params": {p: studies.get_path(data, p) for p in paths},
                        "budget": self.budget(view), "metrics": self.metrics(view)})
        return out

    @staticmethod
    def metric_names(rows: list[dict[str, Any]]) -> list[str]:
        names = sorted({key for row in rows for key in row["metrics"]})
        preferred = [k for k in ("test_acc", "test_f1", "test_r2", "best_val_acc", "val_acc", "silhouette")
                     if k in names]
        return preferred + [k for k in names if k not in preferred]

    def summary(self, group_id: str) -> dict[str, dict[str, Any]]:
        """Per metric, over the finished runs: :func:`summarize` (mean, sd, 95 % t-interval…)."""
        rows = [row for row in self.rows(group_id) if row["status"] in SUCCEEDED]
        return {name: summarize([row["metrics"][name] for row in rows if name in row["metrics"]])
                for name in self.metric_names(rows)}

    def compare(self, first: str, second: str, metric: str) -> dict[str, Any]:
        """Welch's test of study ``second`` against ``first`` on ``metric``, with the budgets checked."""
        a = [r for r in self.rows(first) if metric in r["metrics"]]
        b = [r for r in self.rows(second) if metric in r["metrics"]]
        result = welch([r["metrics"][metric] for r in a], [r["metrics"][metric] for r in b])
        budgets = [studies.budget_mismatch(x["budget"], y["budget"]) for x in a[:1] for y in b[:1]]
        result["budget_differences"] = budgets[0] if budgets else []
        result["matched_budget"] = not result["budget_differences"]
        result["metric"], result["first"], result["second"] = metric, first, second
        return result

    def importance(self, group_id: str, metric: str) -> list[dict[str, Any]]:
        """For a sweep: each swept parameter's importance for ``metric`` (a random forest fitted on
        the parameters, impurity importance) and its correlation with it (numeric ones, Pearson)."""
        rows = [r for r in self.rows(group_id) if metric in r["metrics"]]
        names = list(self.group(group_id).get("space", {}))
        if len(rows) < 3 or not names:
            return []
        import pandas as pd
        from sklearn.ensemble import RandomForestRegressor

        frame = pd.DataFrame([{n: r["params"][n] for n in names} for r in rows])
        target = np.array([r["metrics"][metric] for r in rows], dtype=float)
        columns, owner = [], []
        for name in names:
            values = pd.to_numeric(frame[name], errors="coerce")
            if values.notna().all():
                columns.append(values.to_numpy(dtype=float))
                owner.append(name)
            else:
                for level in sorted(map(str, frame[name].unique())):
                    columns.append((frame[name].astype(str) == level).to_numpy(dtype=float))
                    owner.append(name)
        features = np.column_stack(columns)
        forest = RandomForestRegressor(n_estimators=200, random_state=0).fit(features, target)
        weights: dict[str, float] = {}
        for name, weight in zip(owner, forest.feature_importances_, strict=True):
            weights[name] = weights.get(name, 0.0) + float(weight)
        out = []
        for name in names:
            values = pd.to_numeric(frame[name], errors="coerce")
            correlation = None
            if values.notna().all() and values.std() > 0 and target.std() > 0:
                correlation = float(np.corrcoef(values.to_numpy(dtype=float), target)[0, 1])
            out.append({"parameter": name, "importance": weights.get(name, 0.0), "correlation": correlation})
        return sorted(out, key=lambda row: -row["importance"])

    # Exporting ----------------------------------------------------------------------
    def export(self, group_id: str) -> dict[str, Path]:
        """``study.json`` (definition, every run, statistics with their method), ``runs.csv`` and
        ``study.md`` into ``reports/``."""
        definition, rows = self.group(group_id), self.rows(group_id)
        summary = self.summary(group_id)
        folder = self.paths.reports / f"{definition['kind']}-{definition['name'][:40]}-{group_id}-" \
                                      f"{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        folder.mkdir(parents=True, exist_ok=True)
        environment = self._environment(rows)
        data = {"definition": definition, "runs": rows, "statistics": summary, "environment": environment}
        if definition["kind"] == "sweep":
            names = self.metric_names(rows)
            data["importance"] = {names[0]: self.importance(group_id, names[0])} if names else {}
        (folder / "study.json").write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
        metrics = self.metric_names(rows)
        params = list(definition.get("space", {}))
        with (folder / "runs.csv").open("w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(["run_id", "status", "seed", "spec_hash", "device", "duration_s", *params,
                             *metrics])
            for row in rows:
                writer.writerow([row["run_id"], row["status"], row["seed"], row["spec_hash"], row["device"],
                                 row["duration_s"], *(row["params"][p] for p in params),
                                 *(row["metrics"].get(m, "") for m in metrics)])
        text = self.markdown(definition, rows, summary, environment)
        (folder / "study.md").write_text(text, encoding="utf-8")
        return {"json": folder / "study.json", "csv": folder / "runs.csv", "md": folder / "study.md"}

    def _environment(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        """Where the runs ran (from each run's snapshot): Python, platform, devices, GPUs, CUDA."""
        seen: dict[str, set] = {"python": set(), "platform": set(), "device": set(), "gpus": set(),
                                "cuda_toolkit": set(), "commit": set()}
        for row in rows:
            folder = self.store.run_dir(row["run_id"])
            try:
                snapshot = json.loads((folder / "snapshot.json").read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            seen["python"].add(snapshot.get("python"))
            seen["platform"].add(snapshot.get("platform"))
            seen["device"].add(snapshot.get("device"))
            gpus = snapshot.get("hardware", {}).get("gpus", [])
            seen["gpus"].add(", ".join(g["name"] for g in gpus) or "none")
            seen["cuda_toolkit"].add((snapshot.get("torch_build") or {}).get("cuda_toolkit") or "none")
            seen["commit"].add((snapshot.get("git") or {}).get("commit"))
        return {key: sorted(str(v) for v in values) for key, values in seen.items()}

    @staticmethod
    def markdown(definition: dict[str, Any], rows: list[dict[str, Any]], summary: dict[str, dict[str, Any]],
                 environment: dict[str, Any]) -> str:
        kind = "Seed study" if definition["kind"] == "seeds" else "Sweep"
        lines = [f"# {kind}: {definition['name']}", "",
                 f"`{definition['id']}` · {definition['task']} · {definition['dataset']} · "
                 f"{definition['model']} · "
                 f"{len(rows)} runs", "", "## Statistics", "",
                 "| Metric | n | Mean | SD | 95 % interval | Min | Max |",
                 "| --- | --- | --- | --- | --- | --- | --- |"]
        method = ""
        for name, s in summary.items():
            if not s.get("n"):
                continue
            interval = "—" if s.get("ci_low") is None else f"{s['ci_low']:.4f} – {s['ci_high']:.4f}"
            sd = "—" if s.get("sd") is None else f"{s['sd']:.4f}"
            lines.append(f"| {name} | {s['n']} | {s['mean']:.4f} | {sd} | {interval} | {s['min']:.4f} | "
                         f"{s['max']:.4f} |")
            method = s["method"]
        lines += ["", f"Method: {method}." if method else "", "", "## Runs", "",
                  "| Run | Status | Seed | Spec hash | Duration s | Samples seen |",
                  "| --- | --- | --- | --- | --- | --- |"]
        for row in rows:
            duration = "—" if row["duration_s"] is None else f"{row['duration_s']:.1f}"
            lines.append(f"| `{row['run_id'][:8]}` | {row['status']} | {row['seed']} | "
                         f"`{row['spec_hash'][:10]}` | "
                         f"{duration} | {row['budget'].get('samples_seen', '—')} |")
        lines += ["", "## Environment", ""] + [f"- {key}: {', '.join(values) or '—'}"
                                                for key, values in environment.items()]
        lines += ["", "Every run is an ordinary run with its own folder, snapshot and reproducibility "
                  "report.", ""]
        return "\n".join(lines)
