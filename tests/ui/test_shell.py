"""The window shell: sidebar, search, navigation and placeholders."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QAbstractButton

from app.navigation import NAV, page
from app.ui.pages.placeholder import PlaceholderPage, status_text


def test_window_opens_on_home(window) -> None:
    assert window.current_page_id() == "home"
    assert window.sidebar.items["home"].is_active()


def test_every_page_opens(window, qtbot) -> None:
    for spec in NAV:
        window.navigate(spec.id)
        assert window.current_page_id() == spec.id
        assert window.stack.currentWidget().property("pageId") == spec.id
        if spec.in_sidebar:
            assert window.sidebar.items[spec.id].is_active()
            assert sum(item.is_active() for item in window.sidebar.items.values()) == 1


def test_pages_are_built_once(window) -> None:
    first = window.page_widget("training")
    window.navigate("home")
    window.navigate("training")
    assert window.stack.currentWidget() is first


def test_sidebar_click_navigates(window, qtbot) -> None:
    qtbot.mouseClick(window.sidebar.items["dataset_hub"], Qt.MouseButton.LeftButton)
    assert window.current_page_id() == "dataset_hub"


def test_settings_is_reached_from_the_gear(window, qtbot) -> None:
    assert "settings" not in window.sidebar.items
    qtbot.mouseClick(window.top_bar.settings_button, Qt.MouseButton.LeftButton)
    assert window.current_page_id() == "settings"


def test_search_goes_to_a_page(window, qtbot) -> None:
    window.top_bar.search.setText("hardware")
    qtbot.keyClick(window.top_bar.search, Qt.Key.Key_Return)
    assert window.current_page_id() == "hardware"
    assert window.top_bar.search.text() == ""


def test_ambiguous_search_stays_put(window, qtbot) -> None:
    window.top_bar.search.setText("model")  # Model Zoo and Model Registry
    qtbot.keyClick(window.top_bar.search, Qt.Key.Key_Return)
    assert window.current_page_id() == "home"


def test_placeholders_are_honest(window) -> None:
    for page_id in ("plugin_store", "audio", "simulation_lab"):
        window.navigate(page_id)
        widget = window.stack.currentWidget()
        assert isinstance(widget, PlaceholderPage)
        assert widget.pill.text() == status_text(page(page_id))
        assert not widget.findChildren(QAbstractButton)  # nothing to press that would do nothing


def test_status_text() -> None:
    assert status_text(page("plugin_store")) == "Planned for v0.1 · build phase 9"
    assert status_text(page("audio")) == "Planned for v0.2"
    assert status_text(page("quantum_ml")) == "Planned · scope not decided"
    assert status_text(page("home")) == "Available"


def test_future_releases_are_tagged_in_the_sidebar(window) -> None:
    from PySide6.QtWidgets import QLabel
    tags = {item_id: [w.text() for w in item.findChildren(QLabel) if w.objectName() == "NavTag"]
            for item_id, item in window.sidebar.items.items()}
    assert tags["audio"] == ["v0.2"]
    assert tags["simulation_lab"] == ["TBD"]
    assert tags["training"] == []
