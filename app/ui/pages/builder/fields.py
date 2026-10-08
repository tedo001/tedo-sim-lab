"""Small control factories shared by the Experiment Builder's sections."""

from __future__ import annotations

from typing import Any

import yaml
from PySide6.QtWidgets import QComboBox, QDoubleSpinBox, QFormLayout, QPlainTextEdit, QSpinBox

from ...widgets import Card

__all__ = ["combo", "double", "form", "spin", "yaml_box", "yaml_mapping", "yaml_text"]


def spin(low: int, high: int, value: int, *, special: str = "") -> QSpinBox:
    box = QSpinBox()
    box.setRange(low, high)
    box.setValue(value)
    if special:
        box.setSpecialValueText(special)
    box.setMaximumWidth(160)
    return box


def double(low: float, high: float, value: float, decimals: int, step: float) -> QDoubleSpinBox:
    box = QDoubleSpinBox()
    box.setRange(low, high)
    box.setDecimals(decimals)
    box.setSingleStep(step)
    box.setValue(value)
    box.setMaximumWidth(160)
    return box


def combo(items: list[tuple[str, object]]) -> QComboBox:
    box = QComboBox()
    for title, data in items:
        box.addItem(title, data)
    box.setMaximumWidth(320)
    return box


def form(card: Card) -> QFormLayout:
    layout = QFormLayout()
    layout.setHorizontalSpacing(14)
    layout.setVerticalSpacing(8)
    card.add(layout)
    return layout


def yaml_box(placeholder: str, height: int = 72) -> QPlainTextEdit:
    """A small monospace editor for a YAML mapping (model options, a search space)."""
    box = QPlainTextEdit()
    box.setObjectName("Code")
    box.setPlaceholderText(placeholder)
    box.setFixedHeight(height)
    box.setTabChangesFocus(True)
    return box


def yaml_mapping(text: str, what: str) -> dict[str, Any]:
    """``text`` as a mapping; raises ``ValueError`` naming ``what`` when it is not one."""
    if not text.strip():
        return {}
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ValueError(f"{what}: not valid YAML ({str(exc).splitlines()[0]})") from None
    if not isinstance(data, dict):
        raise ValueError(f"{what}: write one 'name: value' per line")
    return {str(key): value for key, value in data.items()}


def yaml_text(data: dict[str, Any]) -> str:
    if not data:
        return ""
    data = {key: list(value) if isinstance(value, tuple) else value for key, value in data.items()}
    return yaml.safe_dump(data, sort_keys=False, default_flow_style=None, allow_unicode=True).strip()
