"""Vision datasets on disk, the downloader, splits and metrics: no PyTorch needed."""

from __future__ import annotations

import gzip
import hashlib
from pathlib import Path

import numpy as np
import pytest

from core.catalog import load_cards
from core.common.cancel import Cancelled, CancelToken
from core.common.paths import AppPaths
from core.dataset_registry.adapter import AcknowledgementRequired
from labs.common.download import DownloadError, download_file
from labs.computer_vision.training.data import split_indices
from labs.computer_vision.training.metrics import confusion_matrix, summary
from tests.fixtures.fake_vision import write_fake_mnist


@pytest.fixture
def cards(tmp_path):
    paths = AppPaths.resolve(tmp_path / "ws").ensure()
    datasets, _models = load_cards(paths)
    return paths, datasets


def test_vision_cards_are_stable_with_class_names(cards) -> None:
    _paths, datasets = cards
    for card_id in ("mnist", "fashion_mnist", "cifar10"):
        card = datasets.get(card_id)
        assert card.maturity == "stable" and len(card.class_names) == card.num_classes == 10
        assert datasets.status(card_id) == "not_downloaded"
    assert datasets.adapter("cifar10").classes[0] == "airplane"


def test_cifar100_reads_its_class_names_from_the_download(cards, monkeypatch) -> None:
    import pickle

    paths, datasets = cards
    monkeypatch.setenv("TEDO_LAB_WORKSPACE", str(paths.workspace))
    adapter = datasets.adapter("cifar100")
    assert datasets.status("cifar100") == "not_downloaded" and adapter.classes[:2] == ("0", "1")
    folder = paths.datasets / "cifar100" / "cifar-100-python"
    folder.mkdir(parents=True)
    for name in ("train", "test"):
        (folder / name).write_bytes(b"")
    names = [f"class {i}" for i in range(100)]
    (folder / "meta").write_bytes(pickle.dumps({"fine_label_names": names, "coarse_label_names": []}))
    assert datasets.status("cifar100") == "ready" and adapter.classes[99] == "class 99"
    assert adapter.url.endswith("cifar-100-python.tar.gz")


def test_status_follows_the_files(cards) -> None:
    paths, datasets = cards
    adapter = datasets.adapter("mnist")
    assert adapter.local_status(paths.datasets) == "absent"
    raw = write_fake_mnist(paths.datasets)
    assert adapter.local_status(paths.datasets) == "present" and datasets.status("mnist") == "ready"
    (raw / "t10k-labels-idx1-ubyte").unlink()
    assert adapter.local_status(paths.datasets) == "partial"
    assert adapter.describe(paths.datasets).extra["mean"] == (0.1307,)


def test_downloads_need_acknowledgement_when_the_licence_is_unclear(cards) -> None:
    paths, datasets = cards
    with pytest.raises(AcknowledgementRequired):  # MNIST publishes no licence
        datasets.adapter("mnist").prepare(paths.datasets, lambda *_: None, CancelToken())


def _served(tmp_path: Path, content: bytes) -> tuple[str, str]:
    source = tmp_path / "source.bin"
    source.write_bytes(content)
    return source.as_uri(), hashlib.md5(content, usedforsecurity=False).hexdigest()


def test_downloader_checks_falls_back_and_cancels(tmp_path) -> None:
    url, md5 = _served(tmp_path, b"x" * 200_000)
    seen = []
    target = download_file(["file:///nowhere/missing.bin", url], tmp_path / "out" / "data.bin", md5=md5,
                           progress=lambda f, m: seen.append(f), cancel=CancelToken())
    assert target.read_bytes() == b"x" * 200_000 and seen[-1] == pytest.approx(1.0)
    again = download_file(["file:///nowhere"], target, md5=md5, progress=lambda *_: None,
                          cancel=CancelToken())
    assert again == target  # already there: no download at all
    with pytest.raises(DownloadError, match="checksum"):
        download_file([url], tmp_path / "bad.bin", md5="0" * 32, progress=lambda *_: None,
                      cancel=CancelToken())
    assert not (tmp_path / "bad.bin").exists() and not list(tmp_path.glob("*.part"))
    token = CancelToken()
    token.cancel()
    with pytest.raises(Cancelled):
        download_file([url], tmp_path / "cancelled.bin", progress=lambda *_: None, cancel=token)
    assert not (tmp_path / "cancelled.bin").exists()


def test_mnist_download_unpacks(cards, tmp_path, monkeypatch) -> None:
    paths, datasets = cards
    adapter = datasets.adapter("fashion_mnist")
    mirror = tmp_path / "mirror"
    mirror.mkdir()
    resources = []
    for name, _md5 in adapter.resources:
        data = gzip.compress(name.encode())
        (mirror / name).write_bytes(data)
        resources.append((name, hashlib.md5(data, usedforsecurity=False).hexdigest()))
    monkeypatch.setattr(type(adapter), "mirrors", (mirror.as_uri() + "/",))
    monkeypatch.setattr(type(adapter), "resources", tuple(resources))
    adapter.prepare(paths.datasets, lambda *_: None, CancelToken())  # MIT: no acknowledgement needed
    raw = paths.datasets / "fashion_mnist" / "FashionMNIST" / "raw"
    assert (raw / "t10k-labels-idx1-ubyte").read_bytes() == b"t10k-labels-idx1-ubyte.gz"
    assert adapter.local_status(paths.datasets) == "present"


def test_stratified_split_keeps_the_class_mix() -> None:
    targets = np.repeat(np.arange(4), [100, 50, 30, 20])
    train, val = split_indices(targets, 0.2, stratify=True, seed=0)
    assert len(set(train) | set(val)) == 200 and not set(train) & set(val)
    assert np.bincount(targets[val]).tolist() == [20, 10, 6, 4]
    again, _ = split_indices(targets, 0.2, stratify=True, seed=0)
    assert np.array_equal(train, again)
    everything, none = split_indices(targets, 0.0, stratify=True, seed=0)
    assert len(everything) == 200 and len(none) == 0


def test_metrics_match_scikit_learn() -> None:
    rng = np.random.default_rng(3)
    truth, guess = rng.integers(0, 4, 300), rng.integers(0, 4, 300)
    guess[:5] = 0
    matrix = confusion_matrix(truth, guess, 5)  # class 4 never appears: counts as 0
    assert matrix.sum() == 300 and matrix[4].sum() == 0
    result = summary(matrix)
    sklearn = pytest.importorskip("sklearn.metrics")
    assert result["acc"] == pytest.approx(sklearn.accuracy_score(truth, guess))
    for key, function in (("precision", sklearn.precision_score), ("recall", sklearn.recall_score),
                          ("f1", sklearn.f1_score)):
        expected = function(truth, guess, labels=range(5), average="macro", zero_division=0)
        assert result[key] == pytest.approx(expected)
