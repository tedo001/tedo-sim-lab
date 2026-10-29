"""The Experiment Builder's *Repeat and sweep* card: queue the experiment once per seed (a seed
study), or over a hyperparameter space (a sweep: every combination, or a random sample); save it
as a named baseline configuration or start from one. Studies open on the Studies & Sweeps page."""

from __future__ import annotations

import re
from collections.abc import Callable

from PySide6.QtWidgets import QComboBox, QHBoxLayout, QLineEdit, QPushButton, QWidget

from core.experiment_engine import studies
from core.experiment_engine.spec import ExperimentSpec

from ....services.context import AppContext
from ...icons import icon
from ...widgets import Card, label
from .fields import combo, form, spin, yaml_box, yaml_mapping

__all__ = ["StudySection", "parse_seeds"]

_RANGE = re.compile(r"^(-?\d+)\s*-\s*(-?\d+)$")
SPACE_HINT = ("one path per line with its values, e.g.\ntraining.optimizer.lr: [0.001, 0.003, 0.01]\n"
              "training.batch_size: [32, 64]\nmodel.params.n_estimators: [100, 300]")


def parse_seeds(text: str) -> list[int]:
    """``"0, 1, 2"`` or ``"0-4"`` → seeds; raises ``ValueError`` in words."""
    seeds: list[int] = []
    for part in text.replace(";", ",").split(","):
        part = part.strip()
        if not part:
            continue
        span = _RANGE.match(part)
        if span:
            seeds += list(range(int(span[1]), int(span[2]) + 1))
        else:
            seeds.append(int(part))
    if len(seeds) < 2:
        raise ValueError("a seed study needs at least two seeds, e.g. 0-4")
    if len(set(seeds)) != len(seeds):
        raise ValueError("each seed once")
    return seeds


class StudySection(Card):
    def __init__(self, ctx: AppContext, current: Callable[[], ExperimentSpec | None],
                 load: Callable[[ExperimentSpec], None], parent: QWidget | None = None) -> None:
        super().__init__("Repeat and sweep", "several runs that answer one question", parent=parent)
        self.ctx, self.current, self.load = ctx, current, load
        rows = form(self)
        self.seeds = QLineEdit("0-4")
        self.seeds.setToolTip("Seeds as a list (0, 1, 2) or a range (0-4)")
        self.study_button = QPushButton(icon("sigma"), "Queue seed study")
        self.study_button.clicked.connect(lambda: self.queue_study())
        seeds_row = QHBoxLayout()
        seeds_row.addWidget(self.seeds, 1)
        seeds_row.addWidget(self.study_button)
        rows.addRow("Seeds", seeds_row)
        self.space = yaml_box(SPACE_HINT, height=92)
        rows.addRow("Sweep space", self.space)
        self.method = combo([("Every combination (grid)", "grid"), ("Random sample", "random")])
        self.count = spin(1, 500, 12)
        self.count.setToolTip("Runs in a random sweep")
        self.sweep_button = QPushButton(icon("sliders-horizontal"), "Queue sweep")
        self.sweep_button.clicked.connect(lambda: self.queue_sweep())
        sweep_row = QHBoxLayout()
        sweep_row.addWidget(self.method)
        sweep_row.addWidget(self.count)
        sweep_row.addWidget(self.sweep_button)
        sweep_row.addStretch(1)
        rows.addRow("Method", sweep_row)
        self.baseline_name = QLineEdit()
        self.baseline_name.setPlaceholderText("a name, e.g. resnet18-cifar10-reference")
        save = QPushButton(icon("save"), "Save as baseline")
        save.clicked.connect(lambda: self.save_baseline())
        save_row = QHBoxLayout()
        save_row.addWidget(self.baseline_name, 1)
        save_row.addWidget(save)
        rows.addRow("Baseline", save_row)
        self.baselines = QComboBox()
        self.baselines.setMinimumWidth(200)
        open_button = QPushButton(icon("folder-open"), "Load")
        open_button.clicked.connect(self._load_baseline)
        load_row = QHBoxLayout()
        load_row.addWidget(self.baselines, 1)
        load_row.addWidget(open_button)
        rows.addRow("Saved", load_row)
        self.note = label("Every run of a study is an ordinary run; compare configurations only on matched "
                          "compute budgets.", "CardCaption", wrap=True)
        self.add(self.note)
        self.refresh_baselines()

    def _spec(self) -> ExperimentSpec | None:
        spec = self.current()
        if spec is None:
            self.note.setText("Fix the experiment first (see Check and run).")
        return spec

    def queue_study(self) -> str | None:
        spec = self._spec()
        if spec is None:
            return None
        try:
            study_id = self.ctx.studies.launch_study(spec, parse_seeds(self.seeds.text()))
        except ValueError as exc:
            self.note.setText(str(exc))
            return None
        self.note.setText(f"Queued seed study {study_id}; it opens on Studies & Sweeps.")
        self._open(study_id)
        return study_id

    def queue_sweep(self) -> str | None:
        spec = self._spec()
        if spec is None:
            return None
        try:
            space = {key: list(value) if isinstance(value, list | tuple) else [value]
                     for key, value in yaml_mapping(self.space.toPlainText(), "Sweep space").items()}
            sweep_id = self.ctx.studies.launch_sweep(spec, space, method=self.method.currentData(),
                                                     count=self.count.value())
        except ValueError as exc:
            self.note.setText(str(exc))
            return None
        self.note.setText(f"Queued sweep {sweep_id}; it opens on Studies & Sweeps.")
        self._open(sweep_id)
        return sweep_id

    def _open(self, group_id: str) -> None:
        page = self.ctx.show_page("studies")
        if hasattr(page, "show_group"):
            page.show_group(group_id)

    def save_baseline(self) -> str | None:
        spec = self._spec()
        name = self.baseline_name.text().strip()
        if spec is None or not name:
            if spec is not None:
                self.note.setText("Name the baseline first.")
            return None
        path = studies.save_baseline(self.ctx.paths.configs, name, spec)
        self.refresh_baselines()
        self.baselines.setCurrentIndex(max(self.baselines.findData(path.stem), 0))
        self.note.setText(f"Saved as {path.name} in the workspace's configs/baselines/.")
        return path.stem

    def refresh_baselines(self) -> None:
        self.baselines.clear()
        for name, _path in studies.list_baselines(self.ctx.paths.configs):
            self.baselines.addItem(name, name)
        self.baselines.setEnabled(self.baselines.count() > 0)

    def _load_baseline(self) -> None:
        name = self.baselines.currentData()
        if name:
            self.load(studies.load_baseline(self.ctx.paths.configs, name))
            self.note.setText(f"Loaded baseline {name}.")
