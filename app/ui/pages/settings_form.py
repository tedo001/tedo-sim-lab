"""The Settings page's configuration form: every setting in settings.yaml with a control, saved
together (checked first; a wrong value is named and nothing is saved), applied at once where it
can be, otherwise from the next run, start or experiment, as the page says."""

from __future__ import annotations

import re

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QHBoxLayout,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QWidget,
)

from core.common import AppConfig, mlflow_tracking_uri
from core.common.config import ConfigError

from ...services.context import AppContext
from ...services.settings import LATER
from ..icons import icon
from ..widgets import Card, label

__all__ = ["SettingsForm"]

TITLES = {"log_level": "Log level", "mlflow_tracking": "MLflow tracking",
          "mlflow_tracking_uri": "MLflow URI", "terminal_shell": "Terminal shell",
          "max_concurrent_runs": "Concurrent runs", "default_device": "Default device",
          "pass_environment": "Passed to runs"}


class SettingsForm(Card):
    saved = Signal(object)  # the names that changed

    def __init__(self, ctx: AppContext, parent: QWidget | None = None) -> None:
        super().__init__("Configuration", "settings.yaml in the workspace; defaults where unset",
                         parent=parent)
        self.ctx = ctx
        form = QFormLayout()
        self.log_level = QComboBox()
        for name in ("DEBUG", "INFO", "WARNING", "ERROR"):
            self.log_level.addItem(name.capitalize(), name)
        self.mlflow_tracking = QCheckBox("Mirror every run to MLflow")
        self.mlflow_uri = QLineEdit()
        self.mlflow_uri.setPlaceholderText(f"{mlflow_tracking_uri(AppConfig(), ctx.paths)} (default)")
        self.mlflow_uri.setToolTip("Leave empty for the workspace's own SQLite store")
        self.terminal_shell = QComboBox()
        self.terminal_shell.setEditable(True)
        self.terminal_shell.setMinimumWidth(160)  # program paths can be long
        self.terminal_shell.setToolTip("auto = PowerShell on Windows, your $SHELL or bash elsewhere; or a "
                                       "program's name or path")
        self.concurrent = QSpinBox()
        self.concurrent.setRange(1, 8)
        self.concurrent.setToolTip("Runs that train at the same time (1 suits a single GPU)")
        self.device = QComboBox()
        self.device.setMinimumWidth(160)  # GPU names can be long
        self.pass_environment = QLineEdit()
        self.pass_environment.setPlaceholderText("none: runs never see your tokens or keys")
        self.pass_environment.setToolTip("Names of environment variables that runs and workers may see "
                                         "although they look like credentials (e.g. WANDB_API_KEY), "
                                         "separated by spaces or commas. Only names are saved.")
        for title, rows in (("Log level", self.log_level), ("MLflow tracking", self.mlflow_tracking),
                            ("MLflow URI", self.mlflow_uri), ("Terminal shell", self.terminal_shell),
                            ("Concurrent runs", self.concurrent), ("Default device", self.device),
                            ("Passed to runs", self.pass_environment)):
            form.addRow(title, rows)
        self.add(form)
        buttons = QHBoxLayout()
        self.save_button = QPushButton(icon("save"), "Save")
        self.save_button.setObjectName("Primary")
        self.save_button.clicked.connect(self.save)
        self.revert_button = QPushButton(icon("rotate-ccw"), "Revert")
        self.revert_button.clicked.connect(self.load)
        buttons.addWidget(self.save_button)
        buttons.addWidget(self.revert_button)
        buttons.addStretch(1)
        self.add(buttons)
        self.status = label("", "CardCaption", wrap=True)
        self.add(self.status)
        self.load()
        ctx.settings.changed.connect(lambda names: self.load(only=names))
        ctx.hardware.info_changed.connect(lambda _info: self._fill_devices())

    # Values ------------------------------------------------------------------------------------------
    def _fill_shells(self) -> None:
        self.terminal_shell.clear()
        self.terminal_shell.addItem("auto")
        for profile in self.ctx.terminals.profiles():
            if profile.shell and profile.available and profile.argv:
                self.terminal_shell.addItem(profile.argv[0])

    def _fill_devices(self) -> None:
        current = self.device.currentData() or self.ctx.config.default_device
        self.device.clear()
        self.device.addItem("Automatic (GPU when available)", "auto")
        self.device.addItem("CPU", "cpu")
        info = self.ctx.hardware.info
        if info is not None and info.cuda_available:
            for gpu in info.gpus:
                self.device.addItem(f"cuda:{gpu.index} · {gpu.name}", f"cuda:{gpu.index}")
        if info is not None and info.mps_available:
            self.device.addItem("Apple MPS", "mps")
        if self.device.findData(current) < 0:
            self.device.addItem(current, current)  # configured, but not seen on this machine (yet)
        self.device.setCurrentIndex(self.device.findData(current))

    def load(self, only: set[str] | None = None) -> None:
        """Show the saved values (``only`` = just these settings, keeping other edits)."""
        config = self.ctx.config

        def wanted(name: str) -> bool:
            return only is None or name in only

        if wanted("log_level"):
            self.log_level.setCurrentIndex(self.log_level.findData(config.log_level))
        if wanted("mlflow_tracking"):
            self.mlflow_tracking.setChecked(config.mlflow_tracking)
        if wanted("mlflow_tracking_uri"):
            self.mlflow_uri.setText(config.mlflow_tracking_uri)
        if wanted("terminal_shell"):
            self._fill_shells()
            self.terminal_shell.setCurrentText(config.terminal_shell)
        if wanted("max_concurrent_runs"):
            self.concurrent.setValue(config.max_concurrent_runs)
        if wanted("default_device"):
            self.device.clear()
            self._fill_devices()
        if wanted("pass_environment"):
            self.pass_environment.setText(" ".join(config.pass_environment))
        if only is None:
            self.status.setText("")

    def values(self) -> dict:
        return {"log_level": self.log_level.currentData(),
                "mlflow_tracking": self.mlflow_tracking.isChecked(),
                "mlflow_tracking_uri": self.mlflow_uri.text().strip(),
                "terminal_shell": self.terminal_shell.currentText().strip() or "auto",
                "max_concurrent_runs": self.concurrent.value(), "default_device": self.device.currentData(),
                "pass_environment": [n for n in re.split(r"[\s,]+", self.pass_environment.text()) if n]}

    def save(self) -> set[str]:
        try:
            changed = self.ctx.settings.update(**self.values())
        except ConfigError as exc:
            self.status.setText(f"Not saved: {exc}")
            return set()
        if not changed:
            self.status.setText("Nothing changed.")
            return changed
        later = [f"{TITLES[name]} {LATER[name]}" for name in sorted(changed) if name in LATER]
        now = [TITLES[name] for name in sorted(changed) if name not in LATER and name in TITLES]
        parts = [f"{', '.join(now)}: in effect now"] if now else []
        self.status.setText("Saved. " + "; ".join(parts + later) + ".")
        self.saved.emit(changed)
        return changed
