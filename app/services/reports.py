"""Comparison tables and reports, exported as CSV, JSON, Markdown and PDF into the workspace's
``reports/`` folder (one folder per export, so a report travels with its images).

The PDF is Qt's own: the Markdown is laid out by ``QTextDocument`` and printed with
``QPdfWriter``, so no extra dependency is needed.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from PySide6.QtCore import QMarginsF, QSizeF, QUrl
from PySide6.QtGui import QImage, QPageLayout, QPageSize, QPdfWriter, QTextDocument

from core.tracking import LabStore

from .benchmarks import BenchmarkService
from .runs import RunView, slug

__all__ = ["COLUMNS", "FORMATS", "Column", "comparison_row", "export_report", "markdown_by_measure",
           "markdown_table", "run_markdown"]

FORMATS = ("csv", "json", "md", "pdf")


@dataclass(frozen=True)
class Column:
    key: str
    title: str
    show: Callable[[Any], str] = str


def _number(digits: int) -> Callable[[Any], str]:
    return lambda value: f"{value:.{digits}f}"


def _share(value: Any) -> str:
    return f"{value:.2%}"


def _seconds(value: Any) -> str:
    value = int(round(value))
    return f"{value // 60} min {value % 60:02d} s" if value >= 60 else f"{value} s"


COLUMNS = (
    Column("name", "Run"), Column("dataset", "Dataset"), Column("model", "Model"), Column("status", "Status"),
    Column("test_acc", "Test accuracy", _share), Column("test_f1", "Test F1", _number(4)),
    Column("best_val_acc", "Best val. accuracy", _share), Column("test_r2", "Test R²", _number(4)),
    Column("test_rmse", "Test RMSE", _number(4)), Column("silhouette", "Silhouette", _number(4)),
    Column("explained_variance", "Variance explained", _share), Column("cv_mean", "CV mean", _number(4)),
    Column("parameters", "Parameters", lambda v: f"{int(v):,}"), Column("epochs", "Epochs"),
    Column("duration_s", "Training time", _seconds), Column("device", "Device"),
    Column("peak_vram_gb", "Peak VRAM GB", _number(2)),
    Column("latency_ms", "Latency (batch 1) ms", _number(2)),
    Column("throughput", "Best throughput /s", lambda v: f"{v:,.0f}"), Column("run_id", "Run id"),
)


def comparison_row(view: RunView, store: LabStore,
                   benchmarks: BenchmarkService | None = None) -> dict[str, Any]:
    """Everything the comparison shows about one run; keys missing when the run has no such value."""
    metrics = store.latest_metrics(view.id)
    row: dict[str, Any] = {"run_id": view.id, "name": view.name, "dataset": view.dataset, "model": view.model,
                           "task": view.spec.task.value if view.spec else "", "status": view.status,
                           "device": view.device or "", "epochs": 1 if view.tabular else (view.epochs or 0),
                           "git_commit": view.git_commit or "", "mlflow_run_id": view.mlflow_run_id or ""}
    if view.duration_s is not None:
        row["duration_s"] = float(view.duration_s)
    for key in ("test_acc", "test_f1", "test_r2", "test_rmse", "silhouette", "explained_variance", "cv_mean"):
        if key in metrics:
            row[key] = float(metrics[key])
    history = {key: [value for _, value in store.metric_history(view.id, key)]
               for key in ("val_acc", "gpu_mem_gb")}
    if history["val_acc"]:
        row["best_val_acc"] = max(history["val_acc"])
    if history["gpu_mem_gb"]:
        row["peak_vram_gb"] = max(history["gpu_mem_gb"])
    info_file = view.run_dir / "run_info.json"
    if info_file.is_file():
        info = json.loads(info_file.read_text(encoding="utf-8"))
        if info.get("parameters"):
            row["parameters"] = int(info["parameters"])
    for bench in (benchmarks.for_run(view.id) if benchmarks else []):
        first = bench.row(1)
        if first and "latency_ms" not in row:
            row["latency_ms"] = first["mean_ms"]
        if bench.best_throughput and "throughput" not in row:
            row["throughput"] = bench.best_throughput
    return row


def used_columns(rows: Sequence[dict[str, Any]]) -> list[Column]:
    """The columns at least one row has a value for, in the standard order."""
    return [column for column in COLUMNS if any(column.key in row for row in rows)]


def markdown_table(rows: Sequence[dict[str, Any]], columns: Sequence[Column]) -> str:
    def cell(row: dict[str, Any], column: Column) -> str:
        return column.show(row[column.key]).replace("|", "\\|") if column.key in row else "—"

    lines = ["| " + " | ".join(c.title for c in columns) + " |", "|" + "---|" * len(columns)]
    lines += ["| " + " | ".join(cell(row, c) for c in columns) + " |" for row in rows]
    return "\n".join(lines)


def markdown_by_measure(rows: Sequence[dict[str, Any]], columns: Sequence[Column]) -> str:
    """The comparison with one row per measure and one column per run (fits a page)."""
    measures = [c for c in columns if c.key not in ("name", "run_id")]
    head = ["Measure", *(str(row["name"]).replace("|", "\\|") for row in rows)]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for column in measures:
        cells = [column.show(row[column.key]) if column.key in row else "—" for row in rows]
        lines.append("| " + " | ".join([column.title, *cells]) + " |")
    return "\n".join(lines)


def _write_pdf(markdown: str, images: dict[str, QImage], path: Path) -> None:
    document = QTextDocument()
    document.setMarkdown(markdown)  # (setting the text resets resources, so images come after)
    for name, image in images.items():
        document.addResource(QTextDocument.ResourceType.ImageResource, QUrl(name), image)
    writer = QPdfWriter(str(path))
    writer.setPageLayout(QPageLayout(QPageSize(QPageSize.PageSizeId.A4), QPageLayout.Orientation.Portrait,
                                     QMarginsF(15, 15, 15, 15), QPageLayout.Unit.Millimeter))
    writer.setResolution(110)
    writer.setTitle(markdown.splitlines()[0].lstrip("# ") if markdown else "Report")
    writer.setCreator("TEDO AI Research Lab")
    page = writer.pageLayout().paintRectPixels(writer.resolution())
    document.setPageSize(QSizeF(page.width(), page.height()))
    document.print_(writer)


def export_report(reports_root: Path, title: str, markdown: str, *, rows: Sequence[dict[str, Any]] = (),
                  images: dict[str, QImage] | None = None, extra: dict[str, Any] | None = None,
                  formats: Sequence[str] = FORMATS) -> dict[str, Path]:
    """Write the report into ``reports/<title>-<time>/``; returns format → file. ``images`` are
    saved as PNG next to the Markdown (which refers to them by name) and laid into the PDF."""
    folder = reports_root / f"{slug(title, 40)}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    folder.mkdir(parents=True, exist_ok=True)
    images = images or {}
    for name, image in images.items():
        image.save(str(folder / name))
    written: dict[str, Path] = {}
    if "csv" in formats and rows:
        keys = [c.key for c in COLUMNS if any(c.key in row for row in rows)]
        keys += sorted({key for row in rows for key in row} - set(keys))
        with (folder / "comparison.csv").open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=keys)
            writer.writeheader()
            writer.writerows(rows)
        written["csv"] = folder / "comparison.csv"
    if "json" in formats:
        data = {"title": title, "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
                "rows": list(rows), **(extra or {})}
        (folder / "report.json").write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
        written["json"] = folder / "report.json"
    if "md" in formats:
        (folder / "report.md").write_text(markdown, encoding="utf-8")
        written["md"] = folder / "report.md"
    if "pdf" in formats:
        _write_pdf(markdown, images, folder / "report.pdf")
        written["pdf"] = folder / "report.pdf"
    return written


def run_markdown(view: RunView, row: dict[str, Any], charts: Sequence[str] = ()) -> str:
    """One run as a report: what ran, on what, how it scored, and how to reproduce it."""
    from core.experiment_engine.spec import dump_spec_text

    lines = [f"# {view.name}", "", f"Run `{view.id}` · {view.dataset} · {view.model} · {view.status}", "",
             "## Results", "", markdown_by_measure([row], used_columns([row])), ""]
    for name in charts:
        lines += [f"![{name}]({name})", ""]
    snapshot_file = view.run_dir / "snapshot.json"
    if snapshot_file.is_file():
        snapshot = json.loads(snapshot_file.read_text(encoding="utf-8"))
        git, packages = snapshot.get("git", {}), snapshot.get("packages", {})
        facts = [("Python", snapshot.get("python")), ("Platform", snapshot.get("platform")),
                 ("Device", view.device), ("Git commit", git.get("commit")),
                 ("Uncommitted changes", git.get("dirty")),
                 *((name, version) for name, version in sorted(packages.items())
                   if name in ("torch", "torchvision", "scikit-learn", "xgboost", "numpy", "mlflow"))]
        lines += ["## Environment", ""] + [f"- {key}: `{value}`" for key, value in facts if value is not None]
        lines.append("")
    if view.spec is not None:
        lines += ["## experiment.yaml", "", "```yaml", dump_spec_text(view.spec).rstrip(), "```", "",
                  "Run it again: open this file in the Experiment Builder, or press Reproduce on the "
                  "Training page.", ""]
    return "\n".join(lines)
