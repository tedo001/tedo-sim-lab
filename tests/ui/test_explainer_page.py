"""CNN Explainer page: every view opens, follows the input, and the numbers it shows add up."""

from __future__ import annotations

import numpy as np
import pytest
from PySide6.QtCore import QPoint, Qt
from PySide6.QtGui import QColor, QImage
from PySide6.QtWidgets import QScrollArea

from app.ui.pages.explainer import ExplainerPage
from app.ui.pages.explainer.colours import diverging_lut, lightness
from app.ui.pages.explainer.detail_conv import ConvDetail
from app.ui.pages.explainer.detail_layers import InputDetail, PoolDetail, ReluDetail
from app.ui.pages.explainer.detail_output import OutputDetail
from app.ui.pages.explainer.overview import INPUT
from app.ui.pages.explainer.playground import Playground
from app.ui.theme.tokens import COLORS
from labs.computer_vision.explainer import conv_step


@pytest.fixture
def page(ctx, qtbot):
    widget = ExplainerPage(ctx)
    qtbot.addWidget(widget)
    widget.resize(1200, 900)
    return widget


def test_page_explains_an_untrained_mnist_network(page) -> None:
    architecture = page.net.architecture
    assert architecture.class_names == tuple("0123456789")  # from the MNIST dataset card
    assert page.weights_pill.text() == "Untrained" and "build phase 4" in page.weights_note.text()
    assert len(page.overview.columns) == 12  # input, 10 map layers, output
    assert page.prediction.text().startswith("top score: ")
    assert isinstance(page.detail, ConvDetail) and page.detail_card.title.text() == "Convolution"
    first = "Digit 0" if page.preset.samples()[0].label == 0 else "Vertical edge"
    assert page.input_line.text().startswith(f"input: {first}")


def test_clicking_the_overview_opens_each_view(page, qtbot) -> None:
    expected = {INPUT: InputDetail, 1: ReluDetail, 2: ConvDetail, 4: PoolDetail, 11: OutputDetail}
    for layer, kind in expected.items():
        node = 0 if layer == INPUT else 1
        centre = page.overview.node_rect(layer, node).center().toPoint()
        qtbot.mouseClick(page.overview, Qt.MouseButton.LeftButton, pos=centre)
        assert isinstance(page.detail, kind), layer
        assert page.selection == (layer, node)
    page.detail.class_clicked.emit(7)
    assert page.selection == (11, 7) and page.detail.table.selected == 7


def test_hover_names_the_map(page, qtbot) -> None:
    page.show()
    centre = page.overview.node_rect(2, 3).center().toPoint()
    qtbot.mouseMove(page.overview, centre)
    assert page.hover_line.text().startswith("conv_1_2 · channel 4 of 10")


def test_convolution_arithmetic_and_animation(page, qtbot) -> None:
    page.select(2, 4)
    detail = page.detail
    detail.go_to(5, 9)
    step = conv_step(page.trace, 2, 4, 5, 9)
    assert detail.sum_text.text().endswith(f"= {step.value:.3g}".replace("-", "−"))
    assert detail.input_column.view._window == (5, 9, 3, 3)
    detail.set_source(6)
    assert detail.thumbs[6]._selected and "channel 7" in detail.input_column.caption.text()
    detail.hover(0, 0)
    assert not detail.playing and detail.position == (0, 0)
    detail.play()
    qtbot.waitUntil(lambda: detail.position != (0, 0), timeout=2000)
    detail.pause()
    detail.go_to(detail.rows - 1, detail.columns - 1)
    detail.advance()
    assert detail.position == (0, 0)  # wraps around


def test_relu_and_pool_formulas(page) -> None:
    page.select(1, 0)
    page.detail.go_to(3, 4)
    before = page.trace.outputs[0][0, 3, 4]
    assert page.detail.formula.text().endswith(f"= {max(before, 0):.3g}".replace("-", "−"))
    page.select(4, 2)
    page.detail.go_to(1, 1)
    assert page.detail.left.view._window == (2, 2, 2, 2)
    assert page.detail.formula.text().startswith("max(")


def test_controls_change_the_network(page) -> None:
    first = page.net.weight("conv_1_1").copy()
    page.reseed()
    assert not np.array_equal(first, page.net.weight("conv_1_1")) and "seed 1" in page.weights_note.text()
    page.scale_combo.setCurrentIndex(2)  # whole network
    assert len(set(page.limits.values())) == 1
    page.network_combo.setCurrentIndex(2)  # CIFAR-10, colour
    assert page.trace.input.shape == (3, 32, 32) and not page.draw_panel.isVisibleTo(page)
    assert page.net.architecture.class_names[0] == "airplane"
    page.select(INPUT, 2)
    assert "blue channel" in page.detail_facts.text()


def test_drawing_feeds_the_network(page, qtbot) -> None:
    pad = page.pad
    qtbot.mousePress(pad, Qt.MouseButton.LeftButton, pos=QPoint(84, 30))
    qtbot.mouseMove(pad, QPoint(84, 140))
    qtbot.mouseRelease(pad, Qt.MouseButton.LeftButton, pos=QPoint(84, 140))
    assert page.input_line.text() == "input: your drawing"
    assert page.trace.input[0, :, 13:16].max() > 0.9 and page.trace.input[0, :, :5].max() == 0
    pad.clear()
    assert page.trace.input.max() == 0


def test_opening_an_image(page, tmp_path) -> None:
    image = QImage(100, 60, QImage.Format.Format_RGB32)
    image.fill(QColor("#FF0000"))
    path = tmp_path / "red.png"
    image.save(str(path))
    page.network_combo.setCurrentIndex(2)
    assert page.open_image_file(path)
    assert page.trace.input.shape == (3, 32, 32) and page.trace.input[0].min() > 0.99
    assert page.trace.input[2].max() < 0.01 and page.input_line.text() == "input: red.png"
    (tmp_path / "broken.png").write_text("not an image")
    assert not page.open_image_file(tmp_path / "broken.png")
    assert page.error_line.isVisibleTo(page) and "broken.png" in page.error_line.text()


def test_playground(qtbot) -> None:
    playground = Playground()
    qtbot.addWidget(playground)
    playground.set_values(input=6, kernel=3, padding=0, stride=2)
    assert playground.formula.text().endswith("= 2") and "last 1 row" in playground.note.text()
    playground.set_values(input=7)
    assert playground.formula.text().endswith("= 3") and playground.note.text() == ""
    playground.spins["kernel"].setValue(1)
    assert playground.spins["padding"].value() == 0  # padding must stay below the kernel size
    playground.play_button.setChecked(True)
    qtbot.waitUntil(lambda: playground.position != (0, 0), timeout=3000)
    playground._hovered(0, 1)
    assert not playground.play_button.isChecked() and playground.position == (0, 1)


def test_diverging_scale_is_symmetric_and_monotone() -> None:
    lut = diverging_lut()
    lum = [lightness(f"#{int(value) & 0xFFFFFF:06X}") for value in lut]
    assert all(a >= b - 1e-3 for a, b in zip(lum[:128], lum[1:129], strict=True))  # darker towards 0
    assert all(a <= b + 1e-3 for a, b in zip(lum[128:-1], lum[129:], strict=True))  # lighter towards +
    assert abs(lightness(COLORS["heat_neg"]) - lightness(COLORS["heat_pos"])) < 0.01


def test_explainer_fits_half_width(window, qtbot) -> None:
    window.resize(1440, 900)
    window.set_split_open(True, "hardware")
    window.navigate("cnn_explainer")
    qtbot.wait(80)
    scroll = window.page_widget("cnn_explainer").findChild(QScrollArea)
    assert scroll.horizontalScrollBar().maximum() == 0
