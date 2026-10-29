"""Settings' *Remote targets* card: the machines an experiment can run on (an SSH server, a cloud
instance), kept in the workspace's ``configs/remotes.yaml``. Only where and how to reach them is
stored, never a password or key: authentication is SSH's own (a key file, the agent,
``~/.ssh/config``). *Test* runs the target's Python once to show its version and host name."""

from __future__ import annotations

from typing import Any

from PySide6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QHBoxLayout,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QWidget,
)

from core.remote.targets import Target, TargetError, load_targets, save_targets
from core.remote.transport import transport_for

from ...services.context import AppContext
from ..icons import icon
from ..widgets import Card, DataTable, label

__all__ = ["RemoteTargetsCard"]

_PROBE = "import platform, sys; print(sys.version.split()[0], platform.node(), platform.system())"


class RemoteTargetsCard(Card):
    def __init__(self, ctx: AppContext, parent: QWidget | None = None) -> None:
        super().__init__("Remote targets", "machines experiments can run on, over SSH", parent=parent)
        self.ctx = ctx
        self._job: str | None = None
        self.table = DataTable(("Target", "Where"), mono_columns=(1,), stretch_column=1)
        self.table.itemSelectionChanged.connect(self._picked)
        self.add(self.table)
        rows = QFormLayout()
        rows.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        self.name = QLineEdit()
        self.name.setPlaceholderText("gpu-box")
        self.host = QLineEdit()
        self.host.setPlaceholderText("server.example.org or an ~/.ssh/config alias")
        self.user = QLineEdit()
        self.port = QSpinBox()
        self.port.setRange(1, 65535)
        self.port.setValue(22)
        self.python = QLineEdit("python3")
        self.folder = QLineEdit("~/tedo-lab-remote")
        self.key = QLineEdit()
        self.key.setPlaceholderText("optional: ~/.ssh/id_ed25519 (the file's path, never its contents)")
        self.transport = QComboBox()
        self.transport.addItem("SSH", "ssh")
        self.transport.addItem("A folder on this machine", "local")
        for title, widget in (("Name", self.name), ("Host", self.host), ("User", self.user),
                              ("Port", self.port), ("Python there", self.python),
                              ("Folder there", self.folder), ("Key file", self.key),
                              ("Transport", self.transport)):
            rows.addRow(title, widget)
        self.add(rows)
        buttons = QHBoxLayout()
        self.save_button = QPushButton(icon("save"), "Save target")
        self.save_button.clicked.connect(lambda: self.save())
        self.remove_button = QPushButton(icon("trash-2"), "Remove")
        self.remove_button.clicked.connect(lambda: self.remove())
        self.test_button = QPushButton(icon("refresh-cw"), "Test")
        self.test_button.clicked.connect(lambda: self.test())
        for button in (self.save_button, self.remove_button, self.test_button):
            buttons.addWidget(button)
        buttons.addStretch(1)
        self.add(buttons)
        self.note = label("Choose a target under Run on in the Experiment Builder; the code, the run and "
                          "(when its licence allows) the dataset go there, and the results come back here.",
                          "CardCaption", wrap=True)
        self.add(self.note)
        ctx.jobs.job_finished.connect(self._tested)
        self.refresh()

    def targets(self) -> dict[str, Target]:
        try:
            return load_targets(self.ctx.paths.configs)
        except TargetError as exc:
            self.note.setText(str(exc))
            return {}

    def refresh(self) -> None:
        self.names = sorted(self.targets())
        found = self.targets()
        self.table.clear_rows()
        for name in self.names:
            self.table.add_row((name, found[name].describe()))
        self.remove_button.setEnabled(False)
        self.test_button.setEnabled(False)

    def _picked(self) -> None:
        rows = self.table.selectionModel().selectedRows()
        if not rows:
            return
        target = self.targets()[self.names[rows[0].row()]]
        self.name.setText(target.name)
        self.host.setText(target.host)
        self.user.setText(target.user)
        self.port.setValue(target.port)
        self.python.setText(target.python)
        self.folder.setText(target.folder)
        self.key.setText(target.identity_file)
        self.transport.setCurrentIndex(max(self.transport.findData(target.transport), 0))
        self.remove_button.setEnabled(True)
        self.test_button.setEnabled(self._job is None)

    def save(self) -> Target | None:
        try:
            target = Target(self.name.text().strip(), self.host.text().strip(), self.user.text().strip(),
                            self.port.value(), self.python.text().strip() or "python3",
                            self.folder.text().strip() or "~/tedo-lab-remote", self.key.text().strip(),
                            self.transport.currentData())
        except TargetError as exc:
            self.note.setText(str(exc))
            return None
        targets = self.targets()
        targets[target.name] = target
        save_targets(self.ctx.paths.configs, targets)
        self.note.setText(f"Saved {target.name} in configs/remotes.yaml.")
        self.refresh()
        return target

    def remove(self) -> bool:
        targets = self.targets()
        name = self.name.text().strip()
        if targets.pop(name, None) is None:
            return False
        save_targets(self.ctx.paths.configs, targets)
        self.note.setText(f"Removed {name}.")
        self.refresh()
        return True

    def test(self) -> str | None:
        target = self.targets().get(self.name.text().strip())
        if target is None:
            return None

        def work(cancel: Any, progress: Any) -> str:
            done = transport_for(target).run([target.python, "-c", _PROBE], timeout=60)
            if done.returncode != 0:
                raise RuntimeError((done.stderr or done.stdout).strip()[-300:] or f"exit {done.returncode}")
            return done.stdout.strip()

        self.note.setText(f"Reaching {target.name}…")
        self._job = self.ctx.jobs.submit_task(work, title=f"Test {target.name}", quiet=True)
        self.test_button.setEnabled(False)
        return self._job

    def _tested(self, job_id: str, status: str) -> None:
        if job_id != self._job:
            return
        self._job = None
        job = self.ctx.jobs.job(job_id)
        self.note.setText(f"Reached it: Python {job.result}." if status == "completed"
                          else f"Could not reach it: {job.error}")
        self.test_button.setEnabled(True)
