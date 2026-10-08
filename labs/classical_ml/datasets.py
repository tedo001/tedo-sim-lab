"""Tabular datasets: the ones bundled with scikit-learn, and CSV files a person imports.

Importing a CSV copies it to ``<workspace>/datasets/imported/<id>.csv`` and writes a card
next to it (``<id>.yaml``), so the import travels with the project and the lab treats it
like any other dataset. The lab never claims a licence for someone's own data.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, ClassVar

import yaml

from core.catalog import IMPORTED
from core.common.cancel import CancelToken, ProgressFn
from core.common.optional import is_installed
from core.dataset_registry.adapter import DatasetStats, LocalState, TabularAdapter

__all__ = ["CsvAdapter", "CsvPreview", "SklearnBuiltinAdapter", "columns_of", "import_csv", "preview_csv"]


class SklearnBuiltinAdapter(TabularAdapter):
    """Iris, Wine, Breast Cancer, Digits and Diabetes, read from scikit-learn's own copies."""

    LOADERS: ClassVar[dict[str, str]] = {"iris": "load_iris", "wine": "load_wine",
                                         "breast_cancer": "load_breast_cancer", "digits": "load_digits",
                                         "diabetes": "load_diabetes"}

    def local_status(self, root: Path) -> LocalState:
        return "present" if is_installed("scikit-learn") and self.card.id in self.LOADERS else "absent"

    def _download(self, root: Path, progress: ProgressFn, cancel: CancelToken) -> None:
        raise FileNotFoundError(f"{self.card.name} ships with scikit-learn: install scikit-learn in the "
                                "experiment Python environment")

    def _bunch(self) -> Any:
        from sklearn import datasets

        return getattr(datasets, self.LOADERS[self.card.id])(as_frame=True)

    def load_frame(self, root: Path) -> tuple[Any, Any]:
        bunch = self._bunch()
        return bunch.data, bunch.target.rename("target")

    def class_names(self) -> tuple[str, ...]:
        if self.card.class_names:
            return tuple(self.card.class_names)
        names = getattr(self._bunch(), "target_names", None)
        return tuple(str(name) for name in names) if names is not None and len(names) else ()

    def describe(self, root: Path) -> DatasetStats:
        data, target = self.load_frame(root)
        return DatasetStats({"all": len(data)}, self.card.num_classes, self.class_names(),
                            {"features": list(data.columns), "target": target.name})


class CsvAdapter(TabularAdapter):
    """An imported CSV file; ``options`` names the file (relative to datasets/) and the target."""

    def file(self, root: Path) -> Path:
        return root / str(self.card.options.get("file", ""))

    def local_status(self, root: Path) -> LocalState:
        return "present" if self.file(root).is_file() else "absent"

    def _download(self, root: Path, progress: ProgressFn, cancel: CancelToken) -> None:
        raise FileNotFoundError(f"{self.file(root)} is missing; import the CSV file again")

    def load_frame(self, root: Path) -> tuple[Any, Any]:
        import pandas as pd

        frame = pd.read_csv(self.file(root))
        target = self.card.options.get("target")
        if not target:
            return frame, None
        if target not in frame.columns:
            raise KeyError(f"{self.card.name} has no column {target!r}")
        return frame.drop(columns=[target]), frame[target]

    def describe(self, root: Path) -> DatasetStats:
        data, target = self.load_frame(root)
        classes = tuple(str(v) for v in sorted(target.unique())) if target is not None and \
            self.card.options.get("task") == "tabular_classification" else ()
        return DatasetStats({"all": len(data)}, len(classes) or None, classes,
                            {"features": list(data.columns),
                             "target": None if target is None else target.name})


class CsvPreview:
    """What a CSV file holds, for the import form. ``limit`` reads only that many rows (the
    form stays quick on big files); ``complete`` says whether the whole file was read."""

    def __init__(self, path: Path, rows: int = 5, limit: int | None = None) -> None:
        import pandas as pd

        frame = pd.read_csv(path, nrows=limit)
        self.path, self.rows = path, len(frame)
        self.complete = limit is None or len(frame) < limit
        self.columns = [str(column) for column in frame.columns]
        self.kinds = {str(c): ("number" if pd.api.types.is_numeric_dtype(frame[c]) else "text")
                      for c in frame.columns}
        self.distinct = {str(c): int(frame[c].nunique()) for c in frame.columns}
        self.head = frame.head(rows).astype(str).values.tolist()

    def guess_task(self, target: str) -> str:
        """Classification for text targets or few distinct values, regression otherwise."""
        few = self.distinct.get(target, 0) <= max(20, self.rows // 50)
        return "tabular_classification" if self.kinds.get(target) == "text" or few else "tabular_regression"


def preview_csv(path: Path, rows: int = 5, limit: int | None = None) -> CsvPreview:
    return CsvPreview(path, rows, limit)


def _slug(text: str) -> str:
    cleaned = "".join(ch.lower() if ch.isalnum() else "-" for ch in text)
    return "-".join(part for part in cleaned.split("-") if part)[:40] or "dataset"


def import_csv(path: Path, datasets_root: Path, *, name: str, target: str | None, task: str) -> str:
    """Copy ``path`` into the workspace and write its card; returns the new card id."""
    preview = preview_csv(path)
    if target and target not in preview.columns:
        raise KeyError(f"{path.name} has no column {target!r}")
    folder = datasets_root / IMPORTED
    folder.mkdir(parents=True, exist_ok=True)
    card_id, number = f"csv-{_slug(name)}", 2
    while (folder / f"{card_id}.yaml").exists():
        card_id, number = f"csv-{_slug(name)}-{number}", number + 1
    shutil.copy2(path, folder / f"{card_id}.csv")
    tasks = [task] if target else ["clustering", "dimensionality_reduction"]
    card = {"cards": [{
        "id": card_id, "name": name, "description": f"Imported from {path.name}.", "tasks": tasks,
        "modality": "tabular", "source_url": path.resolve().as_uri(),
        "license": {"category": "unspecified", "name": "Your own data",
                    "notes": "Imported by you; the lab knows nothing about its licence."},
        "size": f"{preview.rows} rows × {len(preview.columns) - (1 if target else 0)} features",
        "num_classes": preview.distinct.get(target) if target and task == "tabular_classification" else None,
        "splits": {"all": preview.rows}, "auto_download_allowed": False,
        "adapter": "labs.classical_ml.datasets:CsvAdapter", "maturity": "stable",
        "options": {"file": f"{IMPORTED}/{card_id}.csv", "target": target, "task": task},
        "tags": ["csv", "imported"]}]}
    (folder / f"{card_id}.yaml").write_text(yaml.safe_dump(card, sort_keys=False, allow_unicode=True),
                                            encoding="utf-8")
    return card_id


def columns_of(adapter: TabularAdapter, root: Path) -> tuple[list[str], str | None]:
    """``(feature columns, target column)`` of a tabular dataset, cheaply (a CSV file's header
    only); ``([], None)`` when it cannot be read."""
    try:
        if isinstance(adapter, CsvAdapter):
            import pandas as pd

            columns = [str(c) for c in pd.read_csv(adapter.file(root), nrows=0).columns]
            target = adapter.card.options.get("target")
            return [c for c in columns if c != target], target
        features, target = adapter.load_frame(root)
        return [str(c) for c in features.columns], None if target is None else str(target.name)
    except (OSError, ValueError, KeyError, ImportError):
        return [], None
