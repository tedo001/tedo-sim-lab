"""The ImageNet class list, WordNet through nltk, and the COCO annotation reader."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from labs.computer_vision.coco import CocoFile
from labs.ontology.imagenet import imagenet_classes, search_classes
from labs.ontology.wordnet import FOLDER, WordNet, WordNetAdapter
from tests.fixtures.fake_coco import write_fake_coco


def test_imagenet_class_list_is_the_public_1000() -> None:
    classes = imagenet_classes()
    assert len(classes) == 1000 and len({c.wnid for c in classes}) == 1000
    assert [c.index for c in classes] == list(range(1000))
    assert classes[0].wnid == "n01440764" and classes[0].name == "tench"
    assert classes[1].name == "goldfish" and "orange-red" in classes[1].gloss
    assert search_classes("goldfish")[0].index == 1
    assert search_classes("n01440764")[0].index == 0 and search_classes("999")[0].index == 999
    names = [c.name for c in search_classes("retriever")]
    assert "golden retriever" in names and len(search_classes("")) == 200


def _installed_wordnet() -> Path | None:
    """nltk's own WordNet download (~/nltk_data and friends), when this machine has one."""
    nltk = pytest.importorskip("nltk")

    for folder in nltk.data.path:
        zipped = Path(folder) / "corpora" / "wordnet.zip"
        if zipped.is_file():
            return zipped
    return None


def test_wordnet_adapter_and_reader(tmp_path) -> None:
    from core.dataset_registry.cards import DatasetCard

    card = DatasetCard.model_validate({"id": "wordnet", "name": "WordNet", "tasks": [], "modality": "text",
                                       "source_url": "https://wordnet.princeton.edu/", "size": "10 MB",
                                       "license": {"category": "open-source", "name": "WordNet 3.0 License"}})
    adapter = WordNetAdapter(card)
    assert adapter.local_status(tmp_path) == "absent"
    pytest.importorskip("nltk")
    with pytest.raises(LookupError, match="not downloaded"):
        WordNet(tmp_path)
    source = _installed_wordnet()
    if source is None:
        pytest.skip("WordNet is not installed for nltk on this machine")
    (tmp_path / FOLDER / "corpora").mkdir(parents=True)
    shutil.copy2(source, tmp_path / FOLDER / "corpora" / "wordnet.zip")
    assert adapter.local_status(tmp_path) == "present"
    wordnet = WordNet(tmp_path)
    dog = wordnet.info("n02084071")
    assert dog.node.name == "dog.n.01" and "domestic dog" in dog.lemmas
    assert dog.paths[0][0].lemma == "entity" and dog.paths[0][-1].wnid == "n02084071"
    assert len(dog.imagenet) == 118 and any(c.name == "golden retriever" for c in dog.imagenet)
    assert wordnet.info("dog.n.01").node.wnid == "n02084071"
    assert any(node.name == "crane.n.05" for node in wordnet.lookup("crane"))
    with pytest.raises(LookupError):
        wordnet.synset("n99999999")


def test_coco_file_counts_licences_and_annotations(tmp_path) -> None:
    coco = CocoFile.load(write_fake_coco(tmp_path))
    assert len(coco.images) == 2 and coco.annotation_count == 3 and len(coco.categories) == 3
    assert coco.category_counts() == [("person", "person", 2, 2), ("car", "vehicle", 1, 1),
                                      ("dog", "animal", 0, 0)]
    assert [(name, count) for name, _url, count in coco.licence_counts()] == [
        ("Attribution License", 1), ("Attribution-NonCommercial License", 1)]
    assert coco.image_ids() == [10, 11] and coco.image_ids("car") == [10] and coco.image_ids("dog") == []
    image = coco.image(10)
    assert image.licence == "Attribution License"
    assert [a.category for a in image.annotations] == ["person", "car"]
    assert image.annotations[0].polygons[0][:4] == (4.0, 4.0, 24.0, 4.0)
    crowd = coco.image(11).annotations[0]
    assert crowd.crowd and crowd.rle is not None and not crowd.polygons
    pytest.importorskip("pycocotools")
    mask = CocoFile.decode_mask(crowd, 32, 32)
    assert mask.shape == (32, 32) and int(mask.sum()) == 64 and mask[:8, :8].all()
    assert CocoFile.decode_mask(image.annotations[0], 48, 64) is None


def test_not_a_coco_file(tmp_path) -> None:
    path = tmp_path / "other.json"
    path.write_text('{"hello": 1}', encoding="utf-8")
    with pytest.raises(ValueError, match="not a COCO annotation file"):
        CocoFile.load(path)
