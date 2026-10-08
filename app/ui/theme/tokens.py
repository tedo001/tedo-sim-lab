"""Design tokens: every colour, face and size the interface uses.

Dark, restrained, scientific: near-black neutral surfaces, one muted blue for
anything you can press, desaturated status colours, and no gradients or glow.
Widgets that paint themselves (charts, icons) read these same values, so a
retune happens here and nowhere else.
"""

from __future__ import annotations

__all__ = ["COLORS", "FONT_FAMILY", "MONO_FAMILY", "SIZES", "TONES", "rgba"]

COLORS: dict[str, str] = {
    # Surfaces, darkest first.
    "header": "#0A0B0D",
    "bg": "#0F1114",
    "sidebar": "#111317",
    "surface": "#16191D",
    "raised": "#1C2026",
    "selected": "#192231",
    "border": "#262A31",
    "border_soft": "#1E2228",
    # Text.
    "text": "#E3E6EA",
    "text_dim": "#9AA2AD",
    "text_faint": "#6B737E",
    # The one interactive colour.
    "accent": "#4C7BD0",
    "accent_hover": "#5A88DA",
    "accent_pressed": "#3F6BBA",
    # Status.
    "ok": "#4FA874",
    "warn": "#C9973B",
    "fail": "#D2594F",
    "info": "#5B8CDB",
    "violet": "#9584D4",
    "grey": "#8A929C",
}

#: Pill tones → the status colour they are drawn in.
TONES: dict[str, str] = {
    "ok": COLORS["ok"],             # Ready, Connected, open-source
    "warn": COLORS["warn"],         # Not installed, research-only, non-commercial
    "fail": COLORS["fail"],         # Failed, gated, proprietary
    "info": COLORS["info"],         # Running, commercial
    "experimental": COLORS["violet"],
    "planned": COLORS["grey"],      # Planned, not connected, unspecified
}

FONT_FAMILY = "Inter"
MONO_FAMILY = "JetBrains Mono"

SIZES: dict[str, int] = {
    "font": 13,
    "font_small": 11,
    "font_mono": 12,
    "page_title": 20,
    "card_title": 13,
    "radius": 4,
    "page_margin": 20,
    "gap": 12,
    "sidebar_width": 224,
    "topbar_height": 48,
    "icon": 16,
}


def rgba(hex_colour: str, alpha: float) -> str:
    """``#RRGGBB`` + alpha → a QSS ``rgba(...)`` string."""
    value = hex_colour.lstrip("#")
    red, green, blue = (int(value[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({red}, {green}, {blue}, {alpha:.2f})"
