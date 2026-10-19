"""The 3-D views: the surface and point-cloud widgets, the Dataset Hub's explorer (tables, image
sets, the catalogue's button, the Classical ML page's button) and clusters in 3-D in the results."""

from __future__ import annotations

import gzip

import numpy as np
import pytest
from PySide6.QtCore import QPointF

from app.ui.pages.sklearn_results import SklearnResults
from app.ui.widgets.scatter3d import ScatterView, nice_ticks
from app.ui.widgets.surface3d import SurfaceView


def _wait(qtbot, view) -> None:
    qtbot.waitUntil(lambda: view.table is not None and view._job is None, timeout=60_000)


def test_the_surface_view_draws_turns_and_reads_out(qtbot) -> None:
    view = SurfaceView("A bowl")
    qtbot.addWidget(view)
    view.resize(420, 360)
    view.show()
    x = np.linspace(0, 6, 24, endpoint=False)
    xx, yy = np.meshgrid(x, x)
    height = np.cos(xx) * np.cos(yy)
    view.set_surface(x, x, height, colour=height ** 2, scale="density", labels=("V (Ha)", "n"))
    view.set_points(np.array([[3.0, 3.0], [1.0, 5.0]]), [((1.0, 1.0), (3.0, 3.0), 1.0)])
    assert not view.grab().isNull() and view._screen is not None
    rows, columns = view._screen.shape[:2]
    assert (rows, columns) == (25, 25)  # periodic: closed with the first row and column again
    middle = QPointF(*view._screen[12, 12])
    view.hover_at(middle)
    assert view._hover is not None
    hidden = view.hidden(np.array([[0.0, 0.0, -1.0], [0.0, 0.0, 2.0]]))
    assert hidden.tolist() == [True, False]  # under the surface is hidden, far above it is not
    turned = []
    view.camera_changed.connect(lambda *camera: turned.append(camera))
    view.set_camera(400.0, 120.0, 9.0, notify=True)
    assert turned == [(40.0, 89.0, 4.0)]  # wrapped and clamped
    view.mouseDoubleClickEvent(None)
    assert (view.azimuth, view.elevation) == (325.0, 30.0)


def test_the_scatter_view_groups_folds_and_colours_by_number(qtbot) -> None:
    view = ScatterView("Cells")
    qtbot.addWidget(view)
    view.resize(460, 420)
    view.show()
    rng = np.random.default_rng(0)
    points = rng.normal(size=(300, 3))
    view.set_points(points, groups=np.arange(300) % 3, names=["a", "b", "c"], axes=("CD3", "CD5", "CD19"))
    assert not view.folded() and view.colour_of(1).name().lower() == "#d95926"
    assert not view.grab().isNull() and view._screen.shape == (300, 2)
    view.hover_at(QPointF(*view._screen[7]))
    assert view._hover is not None
    view.set_points(points, groups=np.arange(300) % 10, names=[str(i) for i in range(10)], highlight=4)
    assert view.folded() and view.colour_of(4).name().lower() == "#3987e5"
    assert view.colour_of(5).name().lower() == "#4a5059"  # the other groups fold into one grey
    view.set_highlight(5)
    assert view.highlight == 5
    view.set_points(np.vstack([points, [[np.nan, 0, 0]]]), values=np.arange(301.0), value_label="age")
    assert len(view._points) == 300 and view._values is not None and not view.grab().isNull()
    assert list(nice_ticks(0.13, 0.91)) == pytest.approx([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])


def test_the_dataset_hub_explores_tables_in_3d(window, qtbot) -> None:
    window.navigate("dataset_hub")
    hub = window.page_widget("dataset_hub")
    assert hub.explore("iris") and hub.switch.current == "explore"
    view = hub.views["explore"]
    _wait(qtbot, view)
    assert view.axes_box.currentData() == "components"  # four columns: components first
    assert view.view.names == ["setosa", "versicolor", "virginica"] and view.view.axes[0].startswith("PC 1")
    assert view.components.rows()[0][1] > 0.5 and "99%" in view.components_note.text()
    view.axes_box.setCurrentIndex(view.axes_box.findData("columns"))
    view.columns[2].setCurrentIndex(3)
    assert view.view.axes == ("sepal length (cm)", "sepal width (cm)", "petal width (cm)")
    view.colour.setCurrentIndex(view.colour.findData("none"))
    assert view.view._groups is None
    view.select("digits")
    _wait(qtbot, view)
    assert view.highlight.isEnabled() and view.view.folded()
    view.highlight.setCurrentIndex(7)
    assert view.view.highlight == 7
    view.select("diabetes")
    _wait(qtbot, view)
    assert view.colour.currentData() == "numbers" and view.view._values is not None
    hub.show_view("catalogue")
    catalogue = hub.views["catalogue"]
    catalogue.select("wine")
    assert catalogue.explore_button.isVisible() and catalogue.explore_button.isEnabled()
    catalogue.explore_button.click()
    assert hub.switch.current == "explore" and view.dataset.currentData() == "wine"
    catalogue.select("cifar10")
    assert catalogue.explore_button.isHidden()  # pickled batches are not read in the window


def test_mnist_style_images_are_read_from_idx_files(window, ctx, qtbot) -> None:
    raw = ctx.paths.datasets / "mnist" / "MNIST" / "raw"
    raw.mkdir(parents=True)
    rng = np.random.default_rng(1)
    images = rng.integers(0, 256, (40, 28, 28), dtype=np.uint8)
    labels = (np.arange(40) % 10).astype(np.uint8)
    for prefix in ("train", "t10k"):
        (raw / f"{prefix}-images-idx3-ubyte").write_bytes(b"\x00\x00\x08\x03" + b"".join(
            n.to_bytes(4, "big") for n in images.shape) + images.tobytes())
        (raw / f"{prefix}-labels-idx1-ubyte").write_bytes(gzip.compress(  # gzipped works too
            b"\x00\x00\x08\x01" + len(labels).to_bytes(4, "big") + labels.tobytes()))
        labels_file = raw / f"{prefix}-labels-idx1-ubyte"
        labels_file.with_name(labels_file.name + ".gz").write_bytes(labels_file.read_bytes())
    window.navigate("dataset_hub")
    hub = window.page_widget("dataset_hub")
    view = hub.show_view("explore")
    view.refresh()
    assert hub.explore("mnist")
    _wait(qtbot, view)
    assert view.table.rows == 40 and view.table.values.shape == (40, 784)
    assert view.axes_box.currentData() == "components" and not view.columns[0].isEnabled()
    assert "40 of the 40 test images" in view.status.text()


def test_the_classical_ml_page_opens_the_explorer(window, qtbot) -> None:
    window.navigate("classical_ml")
    page = window.page_widget("classical_ml")
    page.datasets.selectRow(page.dataset_ids.index("wine"))
    page.explore()
    assert window.current_page_id() == "dataset_hub"
    view = window.page_widget("dataset_hub").views["explore"]
    assert view.dataset.currentData() == "wine"
    _wait(qtbot, view)


def test_clusters_show_in_3d_when_the_run_kept_three_directions(qtbot) -> None:
    results = SklearnResults()
    qtbot.addWidget(results)
    results.resize(700, 900)
    results.show()
    rng = np.random.default_rng(2)
    cloud = [[*rng.normal(size=3).tolist(), int(i % 3)] for i in range(60)]
    found = {"task": "clustering", "metrics": {"clusters": 3}, "groups": ["0", "1", "2"],
             "points": [[p[0], p[1], p[3]] for p in cloud], "points3d": cloud}
    results.show_results(found)
    assert not results.dimensions.isHidden() and results.scatter.isVisible() and results.scatter3d.isHidden()
    results.dimensions.select("3d")
    assert results.scatter3d.isVisible() and results.scatter.isHidden()
    assert len(results.scatter3d._points) == 60 and results.scatter3d.names == ["0", "1", "2"]
    results.show_results({**found, "points3d": None})
    assert results.dimensions.isHidden() and results.scatter.isVisible()
