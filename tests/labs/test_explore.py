"""Datasets as points: tables (classes, a numeric target, text columns, sampling), IDX image files,
and principal components."""

from __future__ import annotations

import gzip

import numpy as np
import pytest

from core.catalog import load_catalog
from core.common import CredentialStore
from labs.classical_ml.datasets import import_csv
from labs.common.explore import explorable, principal_components, read_idx, tabular_points


class NoKeyring:
    def get_password(self, *args):
        return None


@pytest.fixture
def registry(workspace):
    return load_catalog(workspace, CredentialStore(env={}, backend=NoKeyring())).datasets


def test_tables_keep_their_classes_or_a_number(registry) -> None:
    iris = tabular_points(registry.adapter("iris"), registry.root)
    assert iris.rows == 150 and len(iris.columns) == 4
    assert iris.group_names == ["setosa", "versicolor", "virginica"]
    assert np.bincount(iris.groups).tolist() == [50, 50, 50] and iris.numbers is None and iris.note == ""
    diabetes = tabular_points(registry.adapter("diabetes"), registry.root)
    assert diabetes.groups is None and diabetes.numbers is not None and diabetes.number_name == "target"
    digits = tabular_points(registry.adapter("digits"), registry.root, limit=500)
    assert digits.rows == 500 and digits.total == 1797 and "500 of 1,797 rows" in digits.note
    assert explorable(registry.adapter("iris")) and not explorable(registry.adapter("cifar10"))
    assert explorable(registry.adapter("mnist"))


def test_text_columns_are_left_out_of_an_imported_csv(workspace, registry, tmp_path) -> None:
    source = tmp_path / "plants.csv"
    rows = ["height,colour,leaves,kind"] + [f"{10 + i},{'red' if i % 2 else 'green'},{i % 7},{'ab'[i % 2]}"
                                             for i in range(30)]
    source.write_text("\n".join(rows) + "\n", encoding="utf-8")
    card_id = import_csv(source, registry.root, name="Plants", target="kind", task="tabular_classification")
    reloaded = load_catalog(workspace, CredentialStore(env={}, backend=NoKeyring())).datasets
    table = tabular_points(reloaded.adapter(card_id), registry.root)
    assert table.columns == ["height", "leaves"] and table.group_names == ["a", "b"]
    assert "1 text column left out (colour)" in table.note


def test_idx_files_are_read_plain_or_gzipped(tmp_path) -> None:
    data = np.arange(24, dtype=np.uint8).reshape(2, 3, 4)
    raw = b"\x00\x00\x08\x03" + b"".join(n.to_bytes(4, "big") for n in data.shape) + data.tobytes()
    (tmp_path / "plain").write_bytes(raw)
    (tmp_path / "zipped.gz").write_bytes(gzip.compress(raw))
    assert np.array_equal(read_idx(tmp_path / "plain"), data)
    assert np.array_equal(read_idx(tmp_path / "zipped.gz"), data)
    floats = np.array([1.5, -2.0], dtype=">f4")
    (tmp_path / "floats").write_bytes(b"\x00\x00\x0d\x01" + (2).to_bytes(4, "big") + floats.tobytes())
    assert read_idx(tmp_path / "floats").tolist() == [1.5, -2.0]
    broken = (("text", b"hello there", "not an IDX"),
              ("kind", b"\x00\x00\x07\x01" + bytes(8), "unknown IDX type"),
              ("short", raw[:-5], "cut short"))
    for name, content, problem in broken:
        (tmp_path / name).write_bytes(content)
        with pytest.raises(ValueError, match=problem):
            read_idx(tmp_path / name)


def test_principal_components_order_the_variance() -> None:
    rng = np.random.default_rng(0)
    base = rng.normal(size=(400, 2)) * [5.0, 1.0]
    echo = base[:, 0] * 0.5 + rng.normal(0, 0.1, 400)
    values = np.column_stack([base[:, 0], base[:, 1], echo, np.full(400, 3.0)])
    scores, share = principal_components(values, standardize=False)
    assert scores.shape == (400, 3) and share.sum() == pytest.approx(1.0) and np.all(np.diff(share) <= 1e-12)
    assert share[0] > 0.9  # the long direction, shared by columns 0 and 2
    centred = values - values.mean(axis=0)
    assert np.allclose(np.abs(scores[:, 0]), np.abs(np.linalg.svd(centred, full_matrices=False)[0][:, 0]
                                                    * np.linalg.svd(centred, compute_uv=False)[0]))
    scaled, share_scaled = principal_components(values)  # unit variance: the constant column stays 0
    assert np.isfinite(scaled).all() and share_scaled[-1] == pytest.approx(0.0, abs=1e-12)
    with_gap = values.copy()
    with_gap[3, 1] = np.nan
    assert np.isfinite(principal_components(with_gap)[0]).all()  # a gap takes the column's mean
    narrow, _ = principal_components(values[:, :2])
    assert narrow.shape == (400, 3) and np.all(narrow[:, 2] == 0)
