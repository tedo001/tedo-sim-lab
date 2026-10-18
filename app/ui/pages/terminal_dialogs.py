"""Choices a terminal needs before it starts: Docker (a shell in a running container, or a new
container from an image, with the workspace mounted if wanted) and a command of your own."""

from __future__ import annotations

import shlex
import sys
from pathlib import Path

from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QInputDialog,
    QRadioButton,
    QVBoxLayout,
    QWidget,
)

from core.terminal.profiles import docker_command, docker_containers, docker_images

from ...services.context import AppContext
from ..widgets import label

__all__ = ["DockerDialog", "ask_command"]


class DockerDialog(QDialog):
    def __init__(self, ctx: AppContext, docker: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.ctx, self.docker = ctx, docker
        self.setWindowTitle("Docker terminal")
        self.setMinimumWidth(460)
        box = QVBoxLayout(self)
        self.exec_choice = QRadioButton("A shell in a running container")
        self.run_choice = QRadioButton("A new container from an image (removed when it exits)")
        group = QButtonGroup(self)
        for button in (self.exec_choice, self.run_choice):
            group.addButton(button)
            button.toggled.connect(self._update)
        form = QFormLayout()
        self.container = QComboBox()
        self.image = QComboBox()
        self.image.setEditable(True)
        self.image.setEditText("python:3.11-slim")
        self.shell = QComboBox()
        self.shell.setEditable(True)
        self.shell.addItems(["sh", "bash", "ash", "zsh", "python"])
        self.mount = QCheckBox(f"Mount the workspace at /workspace ({ctx.paths.workspace})")
        self.mount.setChecked(True)
        form.addRow(self.exec_choice)
        form.addRow("Container", self.container)
        form.addRow(self.run_choice)
        form.addRow("Image", self.image)
        form.addRow("", self.mount)
        form.addRow("Shell", self.shell)
        box.addLayout(form)
        self.status = label("Reading containers and images…", "CardCaption", wrap=True)
        box.addWidget(self.status)
        self.buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok |
                                        QDialogButtonBox.StandardButton.Cancel)
        self.buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Open terminal")
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        box.addWidget(self.buttons)
        self.run_choice.setChecked(True)
        self._job = ctx.jobs.submit_task(lambda cancel, progress: _read_docker(docker),
                                         title="Read Docker containers", quiet=True)
        ctx.jobs.job_finished.connect(self._read)

    def _read(self, job_id: str, status: str) -> None:
        if job_id != self._job:
            return
        try:
            self.ctx.jobs.job_finished.disconnect(self._read)
        except (RuntimeError, TypeError):
            pass
        job = self.ctx.jobs.job(job_id)
        if status != "completed":
            self.status.setText(f"Docker did not answer: {job.error or status}")
            return
        containers, images, problem = job.result
        for container in containers:
            self.container.addItem(f"{container.name} · {container.image} · {container.status}",
                                   container.id)
        current = self.image.currentText()
        self.image.addItems(images)
        self.image.setEditText(current)
        if containers:
            self.exec_choice.setChecked(True)
        self.status.setText(problem or f"{len(containers)} running container(s), {len(images)} image(s).")
        self._update()

    def _update(self) -> None:
        running = self.exec_choice.isChecked()
        self.container.setEnabled(running and self.container.count() > 0)
        self.image.setEnabled(not running)
        self.mount.setEnabled(not running)
        ok = self.buttons.button(QDialogButtonBox.StandardButton.Ok)
        ok.setEnabled(not running or self.container.count() > 0)

    def command(self) -> list[str]:
        shell = self.shell.currentText().strip() or "sh"
        if self.exec_choice.isChecked():
            return docker_command(self.docker, container=str(self.container.currentData()), shell=shell)
        mount = self.ctx.paths.workspace if self.mount.isChecked() else None
        return docker_command(self.docker, image=self.image.currentText().strip(), shell=shell, mount=mount)

    def title(self) -> str:
        if self.exec_choice.isChecked():
            return f"docker: {self.container.currentText().split(' · ')[0]}"
        return f"docker: {self.image.currentText().strip()}"


def _read_docker(docker: str):
    """Containers and images, or what went wrong (the daemon may not be running)."""
    try:
        return docker_containers(docker), docker_images(docker), ""
    except RuntimeError as exc:
        return [], [], f"Docker is installed but did not answer: {exc}"


def ask_command(parent: QWidget, folder: Path) -> list[str] | None:
    """A command line of the person's own (``ssh host``, ``htop``, ``python -m http.server``)."""
    text, ok = QInputDialog.getText(parent, "Run a command in a terminal",
                                    f"Command (runs in {folder}):")
    if not ok or not text.strip():
        return None
    try:
        return shlex.split(text, posix=sys.platform != "win32")
    except ValueError:
        return None
