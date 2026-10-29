"""The data checks a run makes before it trains (``data.validation`` in ``experiment.yaml``):
the runner describes its splits, :mod:`core.dataset_registry.validation` checks them, the report
goes to ``<run>/data_validation.json`` and the rule decides: ``warn`` logs the problems, ``fail``
stops the run when there are errors, ``off`` skips the checks.

``python -m core.experiment_engine.data_checks <experiment.yaml>`` checks a dataset without a run
and prints the report as JSON (the Dataset Hub's *Validate* runs it in the experiment Python)."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from core.dataset_registry.validation import DataCheckFailed, validate

from .runner import ExperimentRunner, RunCallbacks
from .spec import ExperimentSpec

__all__ = ["REPORT_FILE", "check_data", "report_for"]

REPORT_FILE = "data_validation.json"


def _card(dataset: str) -> Any:
    try:
        from core.catalog import load_cards
        from core.common.paths import AppPaths

        datasets, _models = load_cards(AppPaths.resolve())
        return datasets.get(dataset) if dataset in datasets else None
    except Exception:  # an unreadable catalogue: the licence check says the card is unknown
        return None


def report_for(spec: ExperimentSpec, runner: ExperimentRunner) -> dict[str, Any]:
    """The checks for ``spec``'s data as ``runner`` splits it."""
    samples = runner.data_samples(spec)
    card = _card(spec.data.dataset)
    report = validate(samples or {}, card=card)
    if samples is None:
        report["note"] = f"{runner.title} does not describe its splits; only the licence was checked"
    return {"dataset": spec.data.dataset, "task": spec.task.value, **report}


def check_data(spec: ExperimentSpec, runner: ExperimentRunner, run_dir: Path,
               callbacks: RunCallbacks) -> dict[str, Any] | None:
    """Run the checks the spec asks for; raises :class:`DataCheckFailed` under the ``fail`` rule."""
    rule = spec.data.validation
    if rule == "off":
        return None
    callbacks.on_log("Checking the data: duplicates, leakage between splits, labels, balance, licence")
    try:
        report = report_for(spec, runner)
    except Exception as exc:  # a check that cannot run must not stop a run under "warn"
        if rule == "fail":
            raise DataCheckFailed(f"the data checks could not run: {type(exc).__name__}: {exc}") from exc
        callbacks.on_log(f"Data checks skipped: {type(exc).__name__}: {exc}")
        return None
    report["rule"] = rule
    path = run_dir / REPORT_FILE
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    callbacks.on_artifact(path, "data_validation")
    for check in report["checks"]:
        if check["status"] != "ok":
            callbacks.on_log(f"Data check · {check['title']}: {check['status']} — {check['detail']}")
    callbacks.on_log(f"Data checks: {report['summary']}")
    if rule == "fail" and report["errors"]:
        failed = "; ".join(c["detail"] for c in report["checks"] if c["status"] == "error")
        raise DataCheckFailed(f"the data checks found {report['errors']} error(s) and the rule is "
                              f"'fail': {failed}")
    return report


def main(argv: list[str] | None = None) -> int:
    from core.common.paths import CODE_ROOT

    from .runner import RunnerRegistry
    from .spec import load_spec

    args = sys.argv[1:] if argv is None else argv
    spec = load_spec(Path(args[0]))
    registry = RunnerRegistry()
    registry.load_config(CODE_ROOT / "configs" / "runners.yaml")
    try:
        report = report_for(spec, registry.for_spec(spec))
    except Exception as exc:
        print(json.dumps({"error": f"{type(exc).__name__}: {exc}"}))
        return 1
    print(json.dumps(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
