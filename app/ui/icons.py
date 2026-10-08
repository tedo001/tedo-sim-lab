"""Line icons from Lucide (ISC licence, see resources/icons/LICENSE-lucide.txt),
recoloured at render time: the SVGs draw with ``currentColor``, which Qt's SVG
renderer does not resolve, so the colour is substituted into the source.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from PyQt6.QtCore import QByteArray, Qt
from PyQt6.QtGui import QIcon, QPainter, QPixmap
from PyQt6.QtSvg import QSvgRenderer

from .theme.tokens import COLORS, SIZES

__all__ = ["ICONS_DIR", "available_icons", "icon", "pixmap"]

ICONS_DIR = Path(__file__).resolve().parents[1] / "resources" / "icons"
_SCALE = 2  # render at 2x so icons stay crisp on high-DPI screens


def available_icons() -> set[str]:
    return {file.stem for file in ICONS_DIR.glob("*.svg")}


@lru_cache(maxsize=256)
def pixmap(name: str, colour: str = COLORS["text_dim"], size: int = SIZES["icon"]) -> QPixmap:
    """``name`` drawn in ``colour`` at ``size`` logical pixels."""
    source = (ICONS_DIR / f"{name}.svg").read_text(encoding="utf-8")
    renderer = QSvgRenderer(QByteArray(source.replace("currentColor", colour).encode()))
    image = QPixmap(size * _SCALE, size * _SCALE)
    image.fill(Qt.GlobalColor.transparent)
    painter = QPainter(image)
    renderer.render(painter)
    painter.end()
    image.setDevicePixelRatio(_SCALE)
    return image


def icon(name: str, colour: str = COLORS["text_dim"], size: int = SIZES["icon"]) -> QIcon:
    return QIcon(pixmap(name, colour, size))
