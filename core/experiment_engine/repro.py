"""The reproducibility report of a finished run, and the bundle another person reproduces it from.

At the end of every run the worker writes ``<run>/reproduction/``:

- ``report.md`` and ``report.json``: the question and hypothesis, the configuration, where the data
  came from (card, licence, citation, fingerprint, split), the environment (Python, OS, CPU, GPUs,
  driver, CUDA toolkit, packages, code commit), the baseline and how this run compares with it, the
  results (with a 95 % interval on test accuracy from the number of test samples), the limitations
  (written by the person, and the ones the lab can see: one seed, a capped dataset, uncommitted
  code, a non-deterministic run, a simulator…), and the exact commands that reproduce it;
- the files those commands need: ``experiment.yaml``, ``snapshot.json``, ``environment.lock.txt``,
  ``result.json`` and ``code.diff`` when the code had uncommitted changes.

:func:`bundle` zips that folder; ``python -m core.experiment_engine.reproduce <bundle>`` runs it again
in a clean workspace and checks the numbers."""

from __future__ import annotations

import json
import math
import shutil
import zipfile
from pathlib import Path
from typing import Any

from .snapshot import LOCK_FILE
from .spec import ExperimentSpec, dump_spec_text, load_spec, spec_hash

__all__ = ["BUNDLE_FILES", "REPORT_DIR", "RESULT_FILE", "build_report", "bundle", "find_run",
           "render_markdown", "wilson_interval", "write_reproduction"]

REPORT_DIR = "reproduction"
RESULT_FILE = "result.json"
BUNDLE_FILES = ("experiment.yaml", "snapshot.json", LOCK_FILE, RESULT_FILE, "code.diff",
                "data_validation.json")
#: Metrics that describe how long something took, not what it scored.
TIMING = ("epoch_time_s", "duration_s", "lr", "gpu_mem_gb")


def _json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def find_run(experiments: Path, run_id: str) -> Path | None:
    """The folder of run ``run_id`` under ``experiments/<experiment>/``."""
    if not run_id or not experiments.is_dir():
        return None
    found = [path for path in experiments.glob(f"*/{run_id}") if (path / "experiment.yaml").is_file()]
    return found[0] if found else None


def wilson_interval(correct: float, total: int, z: float = 1.959964) -> tuple[float, float] | None:
    """The 95 % Wilson score interval of a proportion (accuracy over ``total`` test samples)."""
    if total <= 0:
        return None
    p = correct / total
    centre = (p + z * z / (2 * total)) / (1 + z * z / total)
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / (1 + z * z / total)
    return max(0.0, centre - half), min(1.0, centre + half)


def _provenance(spec: ExperimentSpec, snapshot: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {"dataset": spec.data.dataset,
                           "fingerprint": (snapshot.get("dataset") or {}).get("fingerprint"),
                           "subset": spec.data.subset, "validation_fraction": spec.data.split.val_fraction,
                           "stratified": spec.data.split.stratify,
                           "test_fraction": getattr(spec.training, "test_fraction", None)}
    try:
        from core.catalog import load_cards
        from core.common.paths import AppPaths

        datasets, _models = load_cards(AppPaths.resolve())
        card = datasets.get(spec.data.dataset)
        out.update(name=card.name, source=card.source_url, licence=card.license.name,
                   licence_category=card.license.category.value, licence_url=card.license.url,
                   citation=card.citation, published_splits=dict(card.splits), size=card.size)
    except Exception:  # an imported or removed card: the id and fingerprint are still recorded
        pass
    return out


def _environment(snapshot: dict[str, Any]) -> dict[str, Any]:
    hardware = snapshot.get("hardware", {})
    return {"python": snapshot.get("python"), "platform": snapshot.get("platform"),
            "os": snapshot.get("os"), "cpu": hardware.get("cpu"), "cpu_cores": hardware.get("cpu_cores"),
            "gpus": hardware.get("gpus", []), "driver": hardware.get("driver"),
            "torch_build": snapshot.get("torch_build", {}), "packages": snapshot.get("packages", {}),
            "lock": snapshot.get("lock", {}), "git": snapshot.get("git", {}),
            "device": snapshot.get("device"),
            "deterministic": snapshot.get("deterministic"), "seed": snapshot.get("seed"),
            "runner": snapshot.get("runner")}


def _scores(metrics: dict[str, Any]) -> dict[str, float]:
    return {key: float(value) for key, value in metrics.items()
            if isinstance(value, int | float) and not any(key.endswith(t) for t in TIMING)}


def _limitations(spec: ExperimentSpec, snapshot: dict[str, Any], result: dict[str, Any],
                 test_total: int | None, validation: Any) -> list[str]:
    seen = []
    if not any(tag.startswith("study:") for tag in spec.tags):
        seen.append("One seed: the spread across seeds is unknown; run a seed study to measure it.")
    if spec.data.subset:
        seen.append(f"Each split was capped at {spec.data.subset:,} samples (data.subset).")
    if getattr(spec.training, "max_steps_per_epoch", None):
        seen.append(f"Epochs were capped at {spec.training.max_steps_per_epoch} batches.")
    if not snapshot.get("deterministic") and spec.training.kind == "torch":
        seen.append("Non-deterministic GPU kernels were allowed: a reproduction can differ slightly.")
    if (snapshot.get("git") or {}).get("dirty"):
        seen.append("The code had uncommitted changes (code.diff is included).")
    if not (snapshot.get("git") or {}).get("commit"):
        seen.append("The code version is unknown (not a git checkout).")
    if test_total is not None and test_total < 1000:
        seen.append(f"Only {test_total:,} test samples: the test scores have wide intervals.")
    if spec.training.kind == "quantum":
        seen.append("Simulated on a classical computer; no quantum hardware was used. Results here are "
                    "not evidence of quantum advantage.")
    if isinstance(validation, dict) and validation.get("errors"):
        seen.append(f"The data checks found {validation['errors']} problem(s); see data_validation.json.")
    if result.get("status") not in ("completed", "early_stopped"):
        seen.append(f"The run did not complete ({result.get('status')}).")
    return seen


def build_report(run_dir: Path, *, experiments: Path | None = None) -> dict[str, Any]:
    """Everything the report says about the run in ``run_dir``, as plain data."""
    spec = load_spec(run_dir / "experiment.yaml")
    snapshot = _json(run_dir / "snapshot.json") or {}
    result = _json(run_dir / RESULT_FILE) or {}
    confusion = _json(run_dir / "test_confusion.json") or {}
    validation = _json(run_dir / "data_validation.json")
    metrics = _scores(result.get("metrics", {}))
    test_total = int(sum(map(sum, confusion["matrix"]))) if confusion.get("matrix") else None
    intervals = {}
    for key in ("test_acc",):
        if key in metrics and test_total:
            interval = wilson_interval(metrics[key] * test_total, test_total)
            if interval:
                intervals[key] = {"low": interval[0], "high": interval[1], "n": test_total,
                                  "method": "Wilson 95 %"}
    baseline = None
    if spec.research.baseline:
        folder = find_run(experiments or run_dir.parent.parent, spec.research.baseline)
        base_result = _json(folder / RESULT_FILE) if folder else None
        if base_result:
            base = _scores(base_result.get("metrics", {}))
            baseline = {"run_id": spec.research.baseline, "name": load_spec(folder / "experiment.yaml").name,
                        "metrics": base, "delta": {k: metrics[k] - base[k] for k in metrics if k in base}}
        else:
            baseline = {"run_id": spec.research.baseline, "missing": True}
    return {"run_id": run_dir.name, "name": spec.name, "task": spec.task.value, "model": spec.model.model,
            "question": spec.research.question, "hypothesis": spec.research.hypothesis,
            "spec_hash": spec_hash(spec), "configuration": dump_spec_text(spec),
            "provenance": _provenance(spec, snapshot), "environment": _environment(snapshot),
            "baseline": baseline, "status": result.get("status"), "duration_s": result.get("duration_s"),
            "results": metrics, "intervals": intervals, "validation": validation,
            "quantum": (_json(run_dir / "results.json") or {}).get("protocol"),
            "limitations": {"stated": spec.research.limitations,
                            "observed": _limitations(spec, snapshot, result, test_total, validation)},
            "reproduce": _commands(run_dir, snapshot)}


def _commands(run_dir: Path, snapshot: dict[str, Any]) -> dict[str, Any]:
    git = snapshot.get("git") or {}
    remote, commit = git.get("remote") or "<the lab's repository>", git.get("commit") or "<unknown commit>"
    steps = [f"git clone {remote} tedo-lab", "cd tedo-lab", f"git checkout {commit}"]
    if git.get("dirty"):
        steps.append("git apply <bundle>/code.diff")
    steps += ["python -m venv .venv",
              "# Windows: .venv\\Scripts\\activate    Linux/macOS: source .venv/bin/activate",
              f"pip install -r <bundle>/{LOCK_FILE}",
              "python -m core.experiment_engine.reproduce <bundle>.zip --workspace <a new folder> "
              "--download"]
    return {"steps": steps, "bundle": f"{run_dir.name}-reproduction.zip"}


def _table(rows: list[tuple[str, str]]) -> str:
    return "\n".join(["| | |", "| --- | --- |", *(f"| {key} | {value} |" for key, value in rows)])


def render_markdown(report: dict[str, Any]) -> str:
    """The report as Markdown (what report.md and the PDF show)."""
    env, data = report["environment"], report["provenance"]
    gpus = ", ".join(f"{g['name']} ({g['memory_gb']} GB)" for g in env.get("gpus", [])) or "none"
    build = env.get("torch_build") or {}
    lines = [f"# Reproducibility report: {report['name']}", "",
             f"Run `{report['run_id']}` · task {report['task']} · model {report['model']} · "
             f"status {report['status']} · spec `{report['spec_hash'][:12]}`", "",
             "## Question", report["question"] or "_Not stated (Experiment Builder → Research)._", "",
             "## Hypothesis", report["hypothesis"] or "_Not stated._", "",
             "## Data provenance", _table([
                 ("Dataset", f"{data.get('name', data['dataset'])} (`{data['dataset']}`)"),
                 ("Source", data.get("source", "—")), ("Licence", f"{data.get('licence', '—')} "
                                                      f"({data.get('licence_category', 'unknown')})"),
                 ("Citation", data.get("citation") or "—"), ("Fingerprint", f"`{data.get('fingerprint')}`"),
                 ("Split", f"validation {data['validation_fraction']}, test "
                           f"{data.get('test_fraction') or 'the dataset’s own'}, "
                           f"stratified {data['stratified']}"),
                 ("Subset", str(data["subset"] or "all samples"))]), "",
             "## Environment", _table([
                 ("Python", f"{env.get('python')} on {env.get('platform')}"),
                 ("CPU", f"{env.get('cpu')} ({env.get('cpu_cores')} cores)"), ("GPUs", gpus),
                 ("Driver", str(env.get("driver") or "—")),
                 ("CUDA toolkit / cuDNN",
                  f"{build.get('cuda_toolkit') or '—'} / {build.get('cudnn') or '—'}"),
                 ("Device", str(env.get("device"))), ("Seed", str(env.get("seed"))),
                 ("Deterministic", str(env.get("deterministic"))),
                 ("Packages", ", ".join(f"{k} {v}" for k, v in env.get("packages", {}).items() if v)),
                 ("Package lock", f"{env.get('lock', {}).get('count')} packages, sha256 "
                                  f"`{str(env.get('lock', {}).get('sha256'))[:16]}…` ({LOCK_FILE})"),
                 ("Code", f"{env.get('git', {}).get('commit') or 'unknown'}"
                          f"{' (with uncommitted changes)' if env.get('git', {}).get('dirty') else ''}")]),
             "",
             "## Results"]
    lines += [_table([(key, f"{value:.6g}") for key, value in report["results"].items()]) or "_No metrics._"]
    for key, interval in report["intervals"].items():
        lines.append(f"\n{key}: 95 % interval {interval['low']:.4f} – {interval['high']:.4f} "
                     f"({interval['method']}, n = {interval['n']:,}).")
    lines += _quantum_lines(report.get("quantum"))
    validation = report.get("validation")
    lines += ["", "## Data checks"]
    if not isinstance(validation, dict):
        lines.append("_Not run (data.validation: off)._")
    else:
        lines.append(f"{validation['summary']} (rule: {validation.get('rule', 'warn')}); split sizes "
                     + ", ".join(f"{name} {count:,}" for name, count in validation.get("splits", {}).items()))
        lines += ["", _table([(c["title"], f"{c['status']}: {c['detail']}") for c in validation["checks"]])]
    baseline = report.get("baseline")
    lines += ["", "## Baseline"]
    if not baseline:
        lines.append("_None chosen._")
    elif baseline.get("missing"):
        lines.append(f"Run `{baseline['run_id']}` was named as the baseline but is not in this workspace.")
    else:
        lines.append(f"Run `{baseline['run_id']}` ({baseline['name']}):")
        lines.append(_table([(key, f"{baseline['metrics'][key]:.6g} → {report['results'][key]:.6g} "
                                   f"({delta:+.4g})") for key, delta in baseline["delta"].items()]))
    lines += ["", "## Limitations", report["limitations"]["stated"] or "_None stated._", ""]
    lines += [f"- {item}" for item in report["limitations"]["observed"]]
    lines += ["", "## Configuration", "```yaml", report["configuration"].rstrip(), "```", "",
              "## Reproduce", "From the bundle (this folder, zipped), in a clean environment:", "", "```bash",
              *report["reproduce"]["steps"], "```", ""]
    return "\n".join(lines)


def _quantum_lines(protocol: dict[str, Any] | None) -> list[str]:
    """The quantum evaluation protocol: simulator, resources, classical baselines, noise, claims."""
    if not protocol:
        return []
    sim, res, claims = protocol["simulator"], protocol["resources"], protocol["claims"]
    noise = sim.get("noise") or {}
    lines = ["", "## Quantum evaluation protocol",
             f"**{claims['results_are'].capitalize()}.** Backend: {claims['backend']}; "
             f"quantum advantage {claims['quantum_advantage']}.", "",
             _table([("Simulator", f"{sim['mode']} ({sim['kind']}), shots {sim.get('shots') or 'exact'}, "
                                   f"seed {sim.get('seed')}"),
                     ("Noise", ", ".join(f"{k} {v}" for k, v in noise.items()) or "none (ideal)"),
                     ("Qubits · depth", f"{res['qubits']} · {res['depth']}"),
                     ("Gates", f"{res['gates_1q']} one-qubit, {res['gates_2q']} two-qubit"),
                     ("Parameters", str(res["parameters"])),
                     ("Circuit evaluations · shots", f"{res['circuit_evaluations']:,} · {res['shots']:,}"),
                     ("Memory per simulated state", f"{res['state_memory_bytes']:,} bytes"),
                     ("Wall time", f"{res['wall_time_s']:.1f} s")]),
             "", "Classical baselines on the same split and features:", "",
             _table([(b["title"], f"test accuracy {b['test_acc']:.4f}, macro F1 {b['test_f1']:.4f}")
                     for b in protocol["baselines"]]) if protocol["baselines"] else "_None fitted._",
             "", f"{claims['comparison']}", "",
             f"Noise comparison ({protocol['noise_comparison']['reference']}):", "",
             _table([(f"noise × {row['factor']}", f"test accuracy {row['test_acc']:.4f}")
                     for row in protocol["noise_comparison"]["rows"]])]
    return lines


def write_reproduction(run_dir: Path, *, experiments: Path | None = None) -> Path:
    """Write ``<run_dir>/reproduction/`` (report and the files to reproduce from); returns the folder."""
    folder = run_dir / REPORT_DIR
    folder.mkdir(exist_ok=True)
    for name in BUNDLE_FILES:
        if (run_dir / name).is_file():
            shutil.copy2(run_dir / name, folder / name)
    report = build_report(run_dir, experiments=experiments)
    (folder / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    text = render_markdown(report)
    (folder / "report.md").write_text(text, encoding="utf-8")
    (folder / "REPRODUCE.md").write_text(text[text.index("## Reproduce"):], encoding="utf-8")
    return folder


def bundle(run_dir: Path, target: Path) -> Path:
    """Zip the run's reproduction folder (written first if missing) into ``target``."""
    folder = run_dir / REPORT_DIR
    if not (folder / "report.json").is_file():
        write_reproduction(run_dir)
    target.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(folder.iterdir()):
            if path.is_file():
                archive.write(path, path.name)
    return target
