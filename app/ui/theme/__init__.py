"""Apply the lab's look: bundled fonts, a dark Fusion palette and the style sheet."""

from __future__ import annotations

import tempfile
from pathlib import Path
from string import Template

from PySide6.QtGui import QColor, QFont, QFontDatabase, QPalette
from PySide6.QtWidgets import QApplication

from .tokens import COLORS, FONT_FAMILY, MONO_FAMILY, SIZES, TONES, rgba

__all__ = ["COLORS", "FONT_FAMILY", "MONO_FAMILY", "SIZES", "TONES", "apply_theme",
           "load_fonts", "mono_font", "stylesheet"]

_HERE = Path(__file__).resolve().parent
FONTS_DIR = _HERE.parents[1] / "resources" / "fonts"
ICONS_DIR = _HERE.parents[1] / "resources" / "icons"

_fonts_loaded: list[str] | None = None


def load_fonts() -> list[str]:
    """Register the bundled Inter and JetBrains Mono faces; return the families."""
    global _fonts_loaded
    if _fonts_loaded is None:
        families: set[str] = set()
        for file in sorted(FONTS_DIR.glob("*.ttf")):
            font_id = QFontDatabase.addApplicationFont(str(file))
            if font_id >= 0:
                families.update(QFontDatabase.applicationFontFamilies(font_id))
        _fonts_loaded = sorted(families)
    return _fonts_loaded


def mono_font(pixel_size: int = SIZES["font_mono"]) -> QFont:
    font = QFont(MONO_FAMILY)
    font.setStyleHint(QFont.StyleHint.Monospace)
    font.setPixelSize(pixel_size)
    return font


def _pill_rules() -> str:
    return "\n".join(
        f'QLabel#Pill[tone="{tone}"] {{ color: {colour}; background: {rgba(colour, 0.12)}; '
        f'border: 1px solid {rgba(colour, 0.32)}; }}'
        for tone, colour in TONES.items())


def coloured_asset(name: str, colour: str) -> str:
    """A copy of icon ``name`` drawn in ``colour``, for QSS ``url()``s (which cannot recolour).

    Written to the temporary folder, never next to the code: the installed app's
    folder is read-only.
    """
    folder = Path(tempfile.gettempdir()) / "tedo-ai-lab-theme"
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / f"{name}-{colour.lstrip('#').lower()}.svg"
    if not target.exists():
        source = (ICONS_DIR / f"{name}.svg").read_text(encoding="utf-8")
        target.write_text(source.replace("currentColor", colour), encoding="utf-8")
    return target.as_posix()


def stylesheet() -> str:
    """The rendered style sheet: the QSS template with every token filled in."""
    template = Template((_HERE / "style.qss").read_text(encoding="utf-8"))
    values = dict(COLORS, font=FONT_FAMILY, mono=MONO_FAMILY, radius=SIZES["radius"],
                  chevron=coloured_asset("chevron-down", COLORS["text_dim"]),
                  font_px=SIZES["font"], mono_px=SIZES["font_mono"],
                  page_title_px=SIZES["page_title"], card_title_px=SIZES["card_title"])
    return template.substitute(values) + "\n" + _pill_rules() + "\n"


def _palette() -> QPalette:
    """Native dialogs, menus and tooltips follow the same colours as the QSS."""
    c = {name: QColor(value) for name, value in COLORS.items()}
    palette = QPalette()
    role = QPalette.ColorRole
    for r, colour in ((role.Window, c["bg"]), (role.WindowText, c["text"]),
                      (role.Base, c["raised"]), (role.AlternateBase, c["surface"]),
                      (role.Text, c["text"]), (role.Button, c["raised"]),
                      (role.ButtonText, c["text"]), (role.Highlight, c["accent"]),
                      (role.HighlightedText, QColor("#FFFFFF")), (role.ToolTipBase, c["raised"]),
                      (role.ToolTipText, c["text"]), (role.PlaceholderText, c["text_faint"]),
                      (role.Link, c["accent_hover"]), (role.Mid, c["border"])):
        palette.setColor(r, colour)
    palette.setColor(QPalette.ColorGroup.Disabled, role.Text, c["text_faint"])
    palette.setColor(QPalette.ColorGroup.Disabled, role.ButtonText, c["text_faint"])
    return palette


def apply_theme(app: QApplication) -> None:
    """Fonts, Fusion, palette and style sheet. Idempotent: re-applying an unchanged style
    sheet would re-polish every widget in the application."""
    load_fonts()
    sheet = stylesheet()
    if app.styleSheet() == sheet:
        return
    app.setStyle("Fusion")
    app.setPalette(_palette())
    font = QFont(FONT_FAMILY)
    font.setPixelSize(SIZES["font"])
    app.setFont(font)
    app.setStyleSheet(sheet)
