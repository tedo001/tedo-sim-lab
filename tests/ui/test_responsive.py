"""Pages stay usable at half width: rows stack, stat tiles wrap into balanced rows."""

from __future__ import annotations

from PySide6.QtWidgets import QLabel, QScrollArea

from app.ui.widgets import ResponsiveRow, StatStrip, StatTile


def test_row_stacks_below_its_breakpoint(qtbot) -> None:
    row = ResponsiveRow([(QLabel("a"), 2), (QLabel("b"), 1)], breakpoint=600)
    qtbot.addWidget(row)
    row.resize(900, 100)
    row.show()
    qtbot.waitUntil(lambda: not row.stacked())
    row.resize(400, 100)
    qtbot.waitUntil(row.stacked)
    row.resize(800, 100)
    qtbot.waitUntil(lambda: not row.stacked())


def test_stat_tiles_wrap_into_balanced_rows(qapp, qtbot) -> None:
    from app.ui.theme import apply_theme
    apply_theme(qapp)  # tile sizes depend on the theme's fonts and margins
    strip = StatStrip({str(i): StatTile(f"T{i}") for i in range(4)})
    qtbot.addWidget(strip)
    strip.show()
    expected = ((1200, 4), (560, 2), (150, 1))  # 560 has room for three: 2 × 2, not 3 + 1
    for width, columns in expected:
        strip.resize(width, 80)
        qtbot.wait(60)  # let Qt apply the new minimum size before measuring
        assert strip.columns() == columns, (width, strip.width())


def test_home_and_hardware_fit_half_width(window, qtbot) -> None:
    window.resize(1440, 900)
    window.set_split_open(True, "hardware")
    window.navigate("home")
    qtbot.wait(50)
    scroll = window.page_widget("home").findChild(QScrollArea)
    assert scroll is not None
    assert scroll.horizontalScrollBar().maximum() == 0  # no sideways scrolling
