"""COCO detection and segmentation scores, checked against pycocotools' own COCOeval, and the
checks of a COCO annotation file."""

from __future__ import annotations

import contextlib
import io
import json

import numpy as np
import pytest

from labs.computer_vision.detection_metrics import box_iou, check_annotations, coco_map, mask_iou


def _square(x: float, y: float, w: float, h: float) -> list[list[float]]:
    return [[x, y, x + w, y, x + w, y + h, x, y + h]]


def random_coco(seed: int, images: int = 12, classes: int = 3) -> tuple[dict, list[dict]]:
    """Ground truth (some crowd regions) and noisy, partly wrong detections of it."""
    rng = np.random.default_rng(seed)
    gt_images, annotations, predictions = [], [], []
    for image_id in range(1, images + 1):
        gt_images.append({"id": image_id, "width": 64, "height": 48, "file_name": f"{image_id}.png",
                          "license": 1})
        for _ in range(rng.integers(0, 5)):
            x, y = rng.uniform(0, 40), rng.uniform(0, 30)
            w, h = min(rng.uniform(4, 22), 63 - x), min(rng.uniform(4, 16), 47 - y)
            category = int(rng.integers(1, classes + 1))
            annotations.append({"id": len(annotations) + 1, "image_id": image_id, "category_id": category,
                                "bbox": [x, y, w, h], "area": w * h, "iscrowd": int(rng.random() < 0.1),
                                "segmentation": _square(x, y, w, h)})
            if rng.random() < 0.8:
                box = [x + rng.normal(0, 2), y + rng.normal(0, 2), w * rng.uniform(.8, 1.2),
                       h * rng.uniform(.8, 1.2)]
                guess = category if rng.random() < .85 else int(rng.integers(1, classes + 1))
                predictions.append({"image_id": image_id, "category_id": guess, "bbox": box,
                                    "score": float(rng.random()), "segmentation": _square(*box)})
        for _ in range(rng.integers(0, 3)):  # false alarms
            box = [rng.uniform(0, 40), rng.uniform(0, 30), rng.uniform(3, 15), rng.uniform(3, 12)]
            predictions.append({"image_id": image_id, "category_id": int(rng.integers(1, classes + 1)),
                                "bbox": box, "score": float(rng.random()), "segmentation": _square(*box)})
    categories = [{"id": c, "name": f"class {c}"} for c in range(1, classes + 1)]
    return {"images": gt_images, "annotations": annotations, "categories": categories,
            "licenses": [{"id": 1, "name": "CC BY 4.0"}]}, predictions


def test_box_and_mask_iou():
    assert box_iou([[0, 0, 10, 10]], [[5, 0, 10, 10]])[0, 0] == pytest.approx(50 / 150)
    assert box_iou([[0, 0, 10, 10]], [[0, 0, 20, 20]], crowd=[True])[0, 0] == pytest.approx(1.0)
    a, b = np.zeros((4, 4), bool), np.zeros((4, 4), bool)
    a[:2], b[1:3] = True, True
    assert mask_iou([a], [b])[0, 0] == pytest.approx(4 / 12)
    assert mask_iou([], [b]).shape == (0, 1)


def test_perfect_detections_score_one():
    truth, _ = random_coco(3)
    perfect = [{"image_id": a["image_id"], "category_id": a["category_id"], "bbox": a["bbox"], "score": 1.0}
               for a in truth["annotations"] if not a["iscrowd"]]
    scores = coco_map(truth, perfect)
    assert scores["map"] == pytest.approx(1.0) and scores["mean_iou_matched"] == pytest.approx(1.0)


@pytest.mark.parametrize("kind", ["bbox", "segm"])
def test_matches_pycocotools(kind):
    pytest.importorskip("pycocotools")
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    for seed in range(6):
        truth, predictions = random_coco(seed)
        with contextlib.redirect_stdout(io.StringIO()):
            gt = COCO()
            gt.dataset = json.loads(json.dumps(truth))
            gt.createIndex()
            results = [dict(p) for p in predictions]
            if kind == "bbox":
                for p in results:
                    p.pop("segmentation")
            evaluation = COCOeval(gt, gt.loadRes(results), kind)
            evaluation.evaluate()
            evaluation.accumulate()
            evaluation.summarize()
        mine = coco_map(truth, predictions, kind=kind)
        expected = list(evaluation.stats[:3])
        assert [mine["map"], mine["map50"], mine["map75"]] == pytest.approx(expected, abs=1e-9)
        assert set(mine["per_class"]) <= {"class 1", "class 2", "class 3"}


def test_annotation_checks_find_a_broken_file():
    truth, _ = random_coco(1)
    assert all(check["status"] == "ok" for check in check_annotations(truth)
               if check["id"] != "coco:unannotated")
    broken = json.loads(json.dumps(truth))
    broken["annotations"].append({"id": 900, "image_id": 999, "category_id": 1, "bbox": [0, 0, 5, 5]})
    broken["annotations"].append({"id": 901, "image_id": 1, "category_id": 42, "bbox": [0, 0, 5, 5]})
    broken["annotations"].append({"id": 902, "image_id": 1, "category_id": 1, "bbox": [60, 40, 30, 30]})
    broken["images"].append(dict(broken["images"][0]))
    broken["images"].append({"id": 77, "width": 10, "height": 10, "file_name": "77.png"})
    found = {check["id"]: check for check in check_annotations(broken)}
    assert found["coco:orphans"]["status"] == "error" and found["coco:orphans"]["examples"] == [900]
    assert found["coco:categories"]["examples"] == [901]
    assert 902 in found["coco:boxes"]["examples"]
    assert found["coco:repeats"]["status"] == "error"
    assert found["coco:licences"]["status"] == "warn" and found["coco:unannotated"]["status"] == "warn"
