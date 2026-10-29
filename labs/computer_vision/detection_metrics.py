"""Detection and segmentation scores against COCO ground truth: box IoU, mask IoU and COCO mean
average precision (AP over IoU 0.50:0.05:0.95, AP50, AP75, per class), computed the way the COCO
evaluation does (score-ordered greedy matching, crowd regions ignored, at most 100 detections per
image and class, precision interpolated at 101 recall points), in numpy.

``check_annotations`` looks for what makes a COCO file unfit to score against: images without
annotations, annotations on unknown images or categories, empty or out-of-image boxes, repeated
image ids or file names, images without a licence."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np

__all__ = ["IOU_THRESHOLDS", "box_iou", "check_annotations", "coco_map", "mask_iou"]

IOU_THRESHOLDS = np.linspace(0.5, 0.95, 10)
_RECALLS = np.linspace(0.0, 1.0, 101)
MAX_DETECTIONS = 100


def box_iou(detections: Any, truths: Any, crowd: Sequence[bool] | None = None) -> np.ndarray:
    """IoU of every detection with every ground-truth box, boxes as ``[x, y, width, height]``. For a
    crowd region the overlap is divided by the detection's own area (as COCO does)."""
    d, g = np.asarray(detections, float).reshape(-1, 4), np.asarray(truths, float).reshape(-1, 4)
    x0 = np.maximum(d[:, None, 0], g[None, :, 0])
    y0 = np.maximum(d[:, None, 1], g[None, :, 1])
    x1 = np.minimum(d[:, None, 0] + d[:, None, 2], g[None, :, 0] + g[None, :, 2])
    y1 = np.minimum(d[:, None, 1] + d[:, None, 3], g[None, :, 1] + g[None, :, 3])
    inter = np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None)
    area_d, area_g = d[:, 2] * d[:, 3], g[:, 2] * g[:, 3]
    union = area_d[:, None] + area_g[None, :] - inter
    if crowd is not None:
        union = np.where(np.asarray(crowd, bool)[None, :], area_d[:, None], union)
    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)


def mask_iou(detections: Sequence[np.ndarray], truths: Sequence[np.ndarray],
             crowd: Sequence[bool] | None = None) -> np.ndarray:
    """IoU of every predicted mask with every ground-truth mask (boolean ``height × width``)."""
    if not len(detections) or not len(truths):
        return np.zeros((len(detections), len(truths)))
    d = np.stack([np.asarray(m, bool).ravel() for m in detections])
    g = np.stack([np.asarray(m, bool).ravel() for m in truths])
    inter = d.astype(np.float64) @ g.T.astype(np.float64)
    area_d, area_g = d.sum(1).astype(float), g.sum(1).astype(float)
    union = area_d[:, None] + area_g[None, :] - inter
    if crowd is not None:
        union = np.where(np.asarray(crowd, bool)[None, :], area_d[:, None], union)
    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)


def _masks(items: Iterable[dict[str, Any]], height: int, width: int) -> list[np.ndarray]:
    from pycocotools import mask as mask_utils

    out = []
    for item in items:
        segmentation = item.get("segmentation")
        if isinstance(segmentation, list):
            rle = mask_utils.merge(mask_utils.frPyObjects(segmentation, height, width))
        elif isinstance(segmentation, dict) and isinstance(segmentation.get("counts"), list):
            rle = mask_utils.frPyObjects(segmentation, height, width)
        else:
            rle = segmentation
        out.append(mask_utils.decode(rle).astype(bool))
    return out


def _match(ious: np.ndarray, crowd: np.ndarray, threshold: float) -> np.ndarray:
    """For each detection (score order) the ground truth it matches at ``threshold``, or -1. A
    crowd region can take any number of detections; they count neither for nor against."""
    matched_gt = np.full(ious.shape[1], -1)
    matched = np.full(ious.shape[0], -1)
    for di in range(ious.shape[0]):
        best, m = min(threshold, 1 - 1e-10), -1
        for gi in range(ious.shape[1]):
            if matched_gt[gi] >= 0 and not crowd[gi]:
                continue
            if m > -1 and not crowd[m] and crowd[gi]:
                break  # ground truth is ordered regular first: never trade a real match for a crowd
            if ious[di, gi] < best:
                continue
            best, m = ious[di, gi], gi
        if m >= 0:
            matched[di], matched_gt[m] = m, di
    return matched


def coco_map(truth: dict[str, Any], predictions: Sequence[dict[str, Any]], *,
             kind: str = "bbox") -> dict[str, Any]:
    """COCO AP of ``predictions`` (COCO results: ``image_id``, ``category_id``, ``bbox`` or
    ``segmentation``, ``score``) against the annotation file ``truth`` (its parsed JSON)."""
    images = {image["id"]: image for image in truth.get("images", [])}
    categories = {c["id"]: c.get("name", str(c["id"])) for c in truth.get("categories", [])}
    gts: dict[tuple[int, int], list[dict]] = defaultdict(list)
    dts: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for item in truth.get("annotations", []):
        gts[item["image_id"], item["category_id"]].append(item)
    for item in predictions:
        if item["image_id"] in images:
            dts[item["image_id"], item["category_id"]].append(item)
    per_class: dict[int, dict[str, Any]] = {}
    matched_ious: list[float] = []
    for cid in categories:
        scores, hits, ignored, regular = [], [[] for _ in IOU_THRESHOLDS], [[] for _ in IOU_THRESHOLDS], 0
        for image_id in sorted(images):
            image = images[image_id]
            g = sorted(gts.get((image_id, cid), []), key=lambda a: bool(a.get("iscrowd", 0)))
            d = sorted(dts.get((image_id, cid), []), key=lambda a: -a["score"])[:MAX_DETECTIONS]
            crowd = np.array([bool(a.get("iscrowd", 0)) for a in g], bool)
            regular += int((~crowd).sum())
            if not d:
                continue
            if kind == "segm":
                h, w = int(image["height"]), int(image["width"])
                ious = mask_iou(_masks(d, h, w), _masks(g, h, w), crowd) if g else np.zeros((len(d), 0))
            else:
                ious = box_iou([a["bbox"] for a in d], [a["bbox"] for a in g], crowd) if g \
                    else np.zeros((len(d), 0))
            scores += [a["score"] for a in d]
            for ti, threshold in enumerate(IOU_THRESHOLDS):
                matched = _match(ious, crowd, threshold)
                hits[ti] += [m >= 0 and not crowd[m] for m in matched]
                ignored[ti] += [m >= 0 and bool(crowd[m]) for m in matched]
                if ti == 0:
                    matched_ious += [float(ious[i, m]) for i, m in enumerate(matched)
                                     if m >= 0 and not crowd[m]]
        if regular == 0:
            continue  # no ground truth of this class: not part of the mean (COCO's -1)
        order = np.argsort(-np.asarray(scores), kind="mergesort")
        precisions = []
        for ti in range(len(IOU_THRESHOLDS)):
            tp = np.asarray(hits[ti], bool)[order]
            keep = ~np.asarray(ignored[ti], bool)[order]
            tp_sum = np.cumsum(tp & keep).astype(float)
            fp_sum = np.cumsum(~tp & keep).astype(float)
            recall = tp_sum / regular
            precision = tp_sum / np.maximum(tp_sum + fp_sum, np.spacing(1))
            precision = np.maximum.accumulate(precision[::-1])[::-1] if len(precision) else precision
            interpolated = np.zeros(len(_RECALLS))
            index = np.searchsorted(recall, _RECALLS, side="left")
            valid = index < len(precision)
            interpolated[valid] = precision[index[valid]]
            precisions.append(interpolated)
        precisions = np.asarray(precisions)
        per_class[cid] = {"name": categories[cid], "ap": float(precisions.mean()),
                          "ap50": float(precisions[0].mean()), "ap75": float(precisions[5].mean()),
                          "truths": regular, "detections": len(scores)}
    aps = [v["ap"] for v in per_class.values()]
    return {"kind": kind, "map": float(np.mean(aps)) if aps else 0.0,
            "map50": float(np.mean([v["ap50"] for v in per_class.values()])) if aps else 0.0,
            "map75": float(np.mean([v["ap75"] for v in per_class.values()])) if aps else 0.0,
            "mean_iou_matched": float(np.mean(matched_ious)) if matched_ious else 0.0,
            "per_class": {v["name"]: v for v in per_class.values()},
            "images": len(images), "predictions": len(predictions),
            "method": "COCO AP: IoU 0.50:0.05:0.95, 101 recall points, max 100 detections, crowd ignored"}


def _check(check_id: str, title: str, status: str, detail: str, **extra: Any) -> dict[str, Any]:
    return {"id": check_id, "title": title, "status": status, "detail": detail, **extra}


def check_annotations(truth: dict[str, Any]) -> list[dict[str, Any]]:
    """Checks of a COCO annotation file (its parsed JSON), in the data-check format."""
    images = truth.get("images", [])
    ids = Counter(image.get("id") for image in images)
    names = Counter(image.get("file_name") for image in images)
    by_id = {image.get("id"): image for image in images}
    categories = {c.get("id") for c in truth.get("categories", [])}
    annotations = truth.get("annotations", [])
    annotated = {a.get("image_id") for a in annotations}
    orphans = [a.get("id") for a in annotations if a.get("image_id") not in by_id]
    unknown = [a.get("id") for a in annotations if a.get("category_id") not in categories]
    bad_boxes = []
    for a in annotations:
        image = by_id.get(a.get("image_id"))
        x, y, w, h = (list(a.get("bbox") or [0, 0, 0, 0]) + [0, 0, 0, 0])[:4]
        outside = image is not None and (x < -1 or y < -1 or x + w > image.get("width", 1e9) + 1
                                         or y + h > image.get("height", 1e9) + 1)
        if w <= 0 or h <= 0 or outside:
            bad_boxes.append(a.get("id"))
    unlabelled = [i for i in by_id if i not in annotated]
    no_licence = [i for i, image in by_id.items() if image.get("license") is None]
    checks = [
        _check("coco:unannotated", "Images without annotations", "warn" if unlabelled else "ok",
               f"{len(unlabelled):,} of {len(by_id):,} images have no annotation (background, or missed?)"
               if unlabelled else "every image is annotated", count=len(unlabelled), examples=unlabelled[:5]),
        _check("coco:orphans", "Annotations on unknown images", "error" if orphans else "ok",
               f"{len(orphans):,} annotations name an image id the file does not list" if orphans
               else "every annotation belongs to a listed image", count=len(orphans), examples=orphans[:5]),
        _check("coco:categories", "Unknown categories", "error" if unknown else "ok",
               f"{len(unknown):,} annotations name a category the file does not list" if unknown
               else "every annotation's category is listed", count=len(unknown), examples=unknown[:5]),
        _check("coco:boxes", "Empty or out-of-image boxes", "error" if bad_boxes else "ok",
               f"{len(bad_boxes):,} boxes have no area or reach outside their image" if bad_boxes
               else "every box has an area inside its image", count=len(bad_boxes), examples=bad_boxes[:5]),
    ]
    repeated = [i for i, n in ids.items() if n > 1] + [n for n, c in names.items() if c > 1 and n]
    checks.append(_check("coco:repeats", "Repeated images", "error" if repeated else "ok",
                         f"{len(repeated):,} image ids or file names appear twice" if repeated
                         else "image ids and file names are unique", count=len(repeated),
                         examples=[str(r) for r in repeated[:5]]))
    checks.append(_check("coco:licences", "Per-image licences", "warn" if no_licence else "ok",
                         f"{len(no_licence):,} images state no licence" if no_licence
                         else "every image states its licence", count=len(no_licence)))
    return checks
