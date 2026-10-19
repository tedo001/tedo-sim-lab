"""The Simulation Lab: experiments you run freely, for research or practice.

Quantum physics — circuits on a state-vector simulator (up to 10 qubits, textbook presets,
Bloch spheres, sampled shots, OpenQASM), wave mechanics (a wave packet tunnelling, reflecting
and oscillating; stationary states) and electrons in a crystal lattice (a self-consistent
Kohn–Sham calculation drawn in 3-D). Computer vision — an image-processing pipeline, synthetic
datasets with COCO annotations, and a pinhole camera with lens distortion.

Save keeps a simulation's settings and outputs in ``results/simulations``; Export script writes it
as Python that runs again with the lab's own modules, to change in the Code page.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QFileDialog,
    QInputDialog,
    QPushButton,
    QSizePolicy,
    QStackedWidget,
    QWidget,
)

from labs.simulation import records
from labs.simulation.scripts import script_for

from ....services.context import AppContext
from ...icons import icon
from ...widgets import Page, Segmented, label
from .camera import CameraLab
from .circuits import CircuitLab
from .images import ImageLab
from .lattice import LatticeLab
from .saved import SavedView
from .scenes import SceneLab
from .waves import WaveLab

__all__ = ["SimulationLabPage"]

TOOLS = (("circuit", "Circuits"), ("waves", "Waves"), ("lattice", "Lattice"), ("image", "Images"),
         ("scenes", "Scenes"), ("camera", "Camera"), ("saved", "Saved"))
#: The tabs' names in full (their tooltips).
TITLES = {"circuit": "Quantum circuits", "waves": "Wave mechanics",
          "lattice": "Electrons in a crystal lattice: the Kohn–Sham density in 3-D",
          "image": "Image processing",
          "scenes": "Synthetic scenes for detection and segmentation", "camera": "Camera model",
          "saved": "Saved simulations"}


class SimulationLabPage(Page):
    def __init__(self, ctx: AppContext, parent: QWidget | None = None) -> None:
        super().__init__("Simulation Lab", "quantum physics and computer vision, to experiment freely",
                         parent)
        self.ctx = ctx
        self.save_button = QPushButton(icon("save"), "Save")
        self.save_button.setToolTip("Keep this simulation's settings and outputs (results/simulations)")
        self.save_button.clicked.connect(lambda: self.save())
        self.script_button = QPushButton(icon("file-code"), "Export script…")
        self.script_button.setToolTip("This simulation as Python that runs it again, to change freely")
        self.script_button.clicked.connect(lambda: self.export_script())
        folder = QPushButton(icon("folder-open"), "Open folder")
        folder.clicked.connect(self.open_folder)
        for widget in (self.save_button, self.script_button, folder):
            self.head.add_action(widget)
        self.switch = Segmented(TOOLS)
        for key, button in self.switch.buttons.items():
            button.setToolTip(TITLES[key])
        self.body.addWidget(self.switch)
        self.message = label("", "CardCaption", wrap=True)
        self.message.hide()
        self.body.addWidget(self.message)
        self.stack = QStackedWidget()
        self.tools = {"circuit": CircuitLab(), "waves": WaveLab(), "lattice": LatticeLab(ctx.jobs),
                      "image": ImageLab(ctx.jobs), "scenes": SceneLab(ctx), "camera": CameraLab()}
        self.saved = SavedView(ctx)
        self.saved.open_requested.connect(self.open_saved)
        for widget in (*self.tools.values(), self.saved):
            widget.message.connect(self.say)
            self.stack.addWidget(widget)
        self.body.addWidget(self.stack, 1)
        self.switch.changed.connect(self.show_tool)
        self.show_tool("circuit")

    def show_tool(self, key: str) -> QWidget:
        widget = self.tools.get(key, self.saved)
        if self.switch.current != key:
            self.switch.buttons[key].setChecked(True)
        for other in (*self.tools.values(), self.saved):  # the stack follows the tool shown
            policy = QSizePolicy.Policy.Preferred if other is widget else QSizePolicy.Policy.Ignored
            other.setSizePolicy(policy, policy)
        self.stack.setCurrentWidget(widget)
        self.stack.adjustSize()
        is_tool = key in self.tools
        self.save_button.setEnabled(is_tool)
        self.script_button.setEnabled(is_tool)
        if key == "saved":
            self.saved.refresh()
        return widget

    def current_tool(self):
        return self.tools.get(self.switch.current)

    def say(self, text: str) -> None:
        self.message.setText(text)
        self.message.setVisible(bool(text))

    # Saving ---------------------------------------------------------------------------------------------
    def save(self, name: str | None = None) -> Path | None:
        tool = self.current_tool()
        if tool is None:
            return None
        if name is None:
            name, ok = QInputDialog.getText(self, "Save the simulation", "Name:", text=tool.default_name())
            if not ok:
                return None
        try:
            folder = records.save(self.ctx.paths.results, tool.kind, name, tool.settings(), tool.outputs())
        except (OSError, ValueError, TypeError) as exc:
            self.say(f"Not saved: {exc}")
            return None
        self.say(f"Saved in {folder}")
        return folder

    def export_script(self, path: Path | None = None) -> Path | None:
        tool = self.current_tool()
        if tool is None:
            return None
        if path is None:
            start = (self.ctx.terminals.default_folder() / f"{tool.kind}_simulation.py")
            chosen, _ = QFileDialog.getSaveFileName(self, "Export as a Python script", str(start),
                                                    "Python (*.py)")
            if not chosen:
                return None
            path = Path(chosen)
        try:
            path.write_text(script_for(tool.kind, tool.settings()), encoding="utf-8")
        except OSError as exc:
            self.say(f"Not written: {exc}")
            return None
        self.say(f"Wrote {path} — open it in the Code page and run it (F5).")
        return path

    def open_saved(self, kind: str, settings: dict) -> None:
        tool = self.tools.get(kind)
        if tool is None:
            return
        self.show_tool(kind)
        try:
            tool.load(settings)
        except (KeyError, TypeError, ValueError) as exc:
            self.say(f"This saved simulation could not be opened: {exc}")
            return
        self.say(f"Opened a saved {records.KINDS[kind].lower()}.")

    def open_folder(self) -> None:
        folder = records.simulations_folder(self.ctx.paths.results)
        folder.mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder)))
