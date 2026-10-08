"""Card loading, search and status, and the shipped catalogue itself."""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import pytest

from core.common.licensing import LicenseCategory, download_policy
from core.common.paths import CODE_ROOT
from core.common.vocab import Framework, Task
from core.dataset_registry import DatasetRegistry
from core.model_registry import ModelRegistry

CONFIGS = CODE_ROOT / "configs"

GOOD = dedent("""\
    cards:
      - id: alpha
        name: Alpha Digits
        tasks: [image_classification]
        modality: image
        source_url: https://example.org/alpha
        license: {category: open-source, name: MIT}
        size: 1 MB
        tags: [digits]
      - id: beta
        name: Beta Scenes
        tasks: [object_detection]
        modality: image
        source_url: https://example.org/beta
        license: {category: gated, name: Beta terms}
        size: 9 GB
""")


def write(tmp_path: Path, name: str, text: str) -> Path:
    file = tmp_path / name
    file.write_text(text)
    return file


@pytest.fixture
def registry(tmp_path: Path) -> DatasetRegistry:
    registry = DatasetRegistry(tmp_path / "data")
    assert registry.load_file(write(tmp_path, "good.yaml", GOOD)) == []
    return registry


def test_load_and_get(registry: DatasetRegistry) -> None:
    assert len(registry) == 2
    assert registry.get("alpha").name == "Alpha Digits"
    assert "beta" in registry


def test_unknown_id_suggests_close_matches(registry: DatasetRegistry) -> None:
    with pytest.raises(KeyError, match="did you mean alpha"):
        registry.get("alpah")


def test_search(registry: DatasetRegistry) -> None:
    assert [c.id for c in registry.search("digits")] == ["alpha"]
    assert [c.id for c in registry.search("alpha digits")] == ["alpha"]
    assert registry.search("alpha scenes") == []
    assert [c.id for c in registry.search(task=Task.OBJECT_DETECTION)] == ["beta"]
    assert [c.id for c in registry.search(license={LicenseCategory.GATED})] == ["beta"]
    assert [c.id for c in registry.search(status="catalog_only")] == ["alpha", "beta"]


@pytest.mark.parametrize("text, message", [
    ("cards:\n  - {id: x, name: X}\n", "Field required"),
    (GOOD.replace("size: 1 MB", "size: 1 MB\n    colour: red"), "Extra inputs"),
    (GOOD.replace("category: open-source", "category: free"), "license.category"),
    (GOOD.replace("id: alpha", "id: Alpha Caps"), "String should match"),
    ("cards: [unclosed\n", "unreadable"),
    ("42\n", "unreadable"),
])
def test_bad_cards_are_reported_not_raised(tmp_path: Path, text: str, message: str) -> None:
    registry = DatasetRegistry(tmp_path)
    errors = registry.load_file(write(tmp_path, "bad.yaml", text))
    assert errors and message in str(errors[0])
    assert registry.errors == errors


def test_duplicates_keep_the_first(tmp_path: Path, registry: DatasetRegistry) -> None:
    errors = registry.load_file(write(tmp_path, "again.yaml", GOOD))
    assert len(errors) == 2 and "duplicate id" in errors[0].message
    assert registry.source("alpha").name == "good.yaml"


def test_adapter_reference_must_be_well_formed(tmp_path: Path) -> None:
    registry = DatasetRegistry(tmp_path)
    errors = registry.load_file(write(tmp_path, "x.yaml", GOOD.replace(
        "size: 1 MB", "size: 1 MB\n    adapter: not a reference")))
    assert "adapter must look like" in str(errors[0])


def test_catalogue_only_and_planned_datasets_explain_themselves(tmp_path: Path) -> None:
    registry = DatasetRegistry(tmp_path)
    registry.load_file(write(tmp_path, "x.yaml", GOOD.replace(
        "size: 1 MB", "size: 1 MB\n    adapter: labs.nowhere:Loader\n    maturity: planned")))
    assert registry.status_detail("alpha")[0] == "planned"
    with pytest.raises(NotImplementedError, match="planned"):
        registry.adapter("alpha")
    with pytest.raises(NotImplementedError, match="catalogue entry"):
        registry.adapter("beta")


def test_stable_card_with_missing_loader_is_not_installed(tmp_path: Path) -> None:
    registry = DatasetRegistry(tmp_path)
    registry.load_file(write(tmp_path, "x.yaml", GOOD.replace(
        "size: 1 MB", "size: 1 MB\n    adapter: labs.nowhere:Loader\n    maturity: stable")))
    status, reason = registry.status_detail("alpha")
    assert status == "not_installed" and "labs.nowhere" in reason


# ── the shipped catalogue ──────────────────────────────────────────────────
@pytest.fixture(scope="module")
def datasets() -> DatasetRegistry:
    registry = DatasetRegistry(Path("/nonexistent"))
    assert registry.load_dir(CONFIGS / "datasets") == []
    return registry


@pytest.fixture(scope="module")
def models() -> ModelRegistry:
    registry = ModelRegistry()
    assert registry.load_dir(CONFIGS / "models") == []
    return registry


def test_shipped_catalogue_has_what_v01_needs(datasets, models) -> None:
    for card_id in ("mnist", "fashion_mnist", "cifar10", "imagenet", "coco2017", "iris", "wine",
                    "breast_cancer", "digits"):
        assert card_id in datasets
    for card_id in ("simple_cnn", "lenet5", "resnet18", "xgboost", "random_forest", "kmeans",
                    "yolov8"):
        assert card_id in models


def test_restricted_datasets_are_never_downloadable(datasets) -> None:
    assert not download_policy(datasets.get("imagenet")).allowed
    assert datasets.get("imagenet").license.category == LicenseCategory.GATED
    assert not download_policy(datasets.get("coco2017")).allowed  # never bulk-downloaded


def test_v01_datasets_may_be_downloaded_on_request(datasets) -> None:
    for card_id in ("mnist", "fashion_mnist", "cifar10"):
        assert download_policy(datasets.get(card_id)).allowed, card_id


def test_licences_without_a_published_licence_say_so(datasets) -> None:
    for card_id in ("mnist", "cifar10", "cifar100"):
        assert datasets.get(card_id).license.category == LicenseCategory.UNSPECIFIED


def test_ultralytics_is_flagged_copyleft(models) -> None:
    yolo = models.get("yolov8")
    assert yolo.license.copyleft and "AGPL" in yolo.license.name


def test_pretrained_weights_carry_their_own_licence(models) -> None:
    weights = models.get("resnet18").weights_by_id("imagenet1k_v1")
    assert weights.license.category == LicenseCategory.RESEARCH_ONLY
    assert models.get("resnet18").license.category == LicenseCategory.OPEN_SOURCE


def test_every_card_has_a_licence_and_source(datasets, models) -> None:
    for card in [*datasets.all(), *models.all()]:
        assert card.license.name and card.source_url.startswith("http"), card.id


def test_planned_models_cannot_be_built(models) -> None:
    assert models.status("resnet18") == "planned"
    with pytest.raises(NotImplementedError, match="v0.1"):
        models.builder("resnet18")
    assert [c.id for c in models.search(framework=Framework.XGBOOST)] == ["xgboost"]


def test_stable_cards_resolve(datasets, models) -> None:
    """Once a loader or builder is marked stable, its code must exist."""
    for card in datasets.all():
        if card.maturity == "stable" and card.adapter:
            assert datasets.status(card.id) in ("ready", "not_downloaded"), card.id
    for card in models.all():
        if card.maturity == "stable":
            assert models.status(card.id) in ("ready", "not_installed"), card.id
