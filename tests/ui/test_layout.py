"""Collapsible sidebar (Ctrl+B) and split view (Ctrl+\\), remembered between sessions."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel

from app.main_window import MainWindow
from app.services.ui_state import UiState
from app.ui.theme.tokens import SIZES


def visible_labels(item) -> list[str]:
    return [label.text() for label in item.findChildren(QLabel) if label.isVisible() and label.text()]


def test_sidebar_collapses_to_an_icon_rail(window, qtbot) -> None:
    qtbot.mouseClick(window.top_bar.sidebar_button, Qt.MouseButton.LeftButton)
    assert window.sidebar_collapsed()
    assert window.sidebar.width() == SIZES["sidebar_rail"]
    training = window.sidebar.items["training"]
    assert visible_labels(training) == []                      # icon only
    assert training.toolTip().startswith("Training:")          # the name is in the tooltip
    assert "Expand" in window.top_bar.sidebar_button.toolTip()
    qtbot.mouseClick(training, Qt.MouseButton.LeftButton)      # still navigates
    assert window.current_page_id() == "training"
    window.toggle_sidebar()
    assert window.sidebar.width() == SIZES["sidebar_width"] and "Training" in visible_labels(training)


def test_ctrl_b_toggles_the_sidebar(window, qtbot) -> None:
    window.activateWindow()
    qtbot.keyClick(window, Qt.Key.Key_B, Qt.KeyboardModifier.ControlModifier)
    assert window.sidebar_collapsed()
    qtbot.keyClick(window, Qt.Key.Key_B, Qt.KeyboardModifier.ControlModifier)
    assert not window.sidebar_collapsed()


def test_split_view_shows_a_second_page(window, qtbot) -> None:
    assert not window.split_open()
    qtbot.mouseClick(window.top_bar.split_button, Qt.MouseButton.LeftButton)
    assert window.split_open() and window.top_bar.split_button.isChecked()
    window.set_split_open(True, "documentation")
    pane = window.split_pane
    assert pane.current_page_id() == "documentation"
    shown = pane.stack.currentWidget()
    assert shown.property("pageId") == "documentation"
    assert shown is not window.page_widget("documentation")    # its own instance
    window.navigate("settings")                                 # the main side moves alone
    assert pane.current_page_id() == "documentation"


def test_split_page_chooser_and_close(window, qtbot) -> None:
    window.set_split_open(True, "home")
    pane = window.split_pane
    pane.chooser.setCurrentIndex(pane.chooser.findData("settings"))
    assert pane.current_page_id() == "settings"
    qtbot.mouseClick(pane.close_button, Qt.MouseButton.LeftButton)
    assert not window.split_open() and not window.top_bar.split_button.isChecked()


def test_ctrl_backslash_toggles_split_view(window, qtbot) -> None:
    qtbot.keyClick(window, Qt.Key.Key_Backslash, Qt.KeyboardModifier.ControlModifier)
    assert window.split_open()
    qtbot.keyClick(window, Qt.Key.Key_Backslash, Qt.KeyboardModifier.ControlModifier)
    assert not window.split_open()


def test_layout_is_remembered(ctx, qtbot, tmp_path) -> None:
    state = UiState.in_file(tmp_path / "layout.ini")
    first = MainWindow(ctx, state)
    qtbot.addWidget(first)
    first.set_sidebar_collapsed(True)
    first.set_split_open(True, "documentation")
    first.close()
    second = MainWindow(ctx, UiState.in_file(tmp_path / "layout.ini"))
    qtbot.addWidget(second)
    assert second.sidebar_collapsed()
    assert second.split_open() and second.split_pane.current_page_id() == "documentation"


def test_a_stale_remembered_page_falls_back(ctx, qtbot, tmp_path) -> None:
    state = UiState.in_file(tmp_path / "layout.ini")
    state.split_open = True
    state.split_page = "a_page_from_an_old_build"
    window = MainWindow(ctx, state)
    qtbot.addWidget(window)
    assert window.split_open() and window.split_pane.current_page_id() == "hardware"
