"""Reading a COCO-format annotation file a person already has (``instances_val2017.json`` or
their own export): categories, per-image licences, and each image's annotations.

The lab never downloads COCO in bulk; the inspector only opens local files. Polygons are kept
as given; RLE masks (crowd regions) are decoded with pycocotools when it is installed.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = ["Annotation", "CocoFile", "CocoImage"]


@dataclass(frozen=True)
class Annotation:
    id: int
    category: str
    #: x, y, width, height in pixels.
    bbox: tuple[float, float, float, float]
    area: float
    crowd: bool
    #: Polygons as flat [x1, y1, x2, y2, ...] lists; empty for RLE masks.
    polygons: tuple[tuple[float, ...], ...] = ()
    rle: Any = None
    keypoints: tuple[float, ...] = ()


@dataclass(frozen=True)
class CocoImage:
    id: int
    file_name: str
    width: int
    height: int
    licence: str
    annotations: tuple[Annotation, ...] = ()


@dataclass
class CocoFile:
    path: Path
    info: dict[str, Any]
    categories: dict[int, dict[str, Any]]
    licences: dict[int, dict[str, Any]]
    images: dict[int, dict[str, Any]]
    by_image: dict[int, list[dict[str, Any]]] = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path) -> CocoFile:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(data, dict) or "images" not in data:
            raise ValueError(f"{Path(path).name} is not a COCO annotation file (no 'images' list)")
        by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for annotation in data.get("annotations", []):
            by_image[annotation["image_id"]].append(annotation)
        return cls(Path(path), data.get("info", {}) or {},
                   {c["id"]: c for c in data.get("categories", [])},
                   {lic["id"]: lic for lic in data.get("licenses", [])},
                   {image["id"]: image for image in data["images"]}, dict(by_image))

    # Summary ------------------------------------------------------------------------
    @property
    def annotation_count(self) -> int:
        return sum(len(items) for items in self.by_image.values())

    def category_counts(self) -> list[tuple[str, str, int, int]]:
        """(name, supercategory, annotations, images), most annotated first."""
        annotations, images = Counter(), defaultdict(set)
        for image_id, items in self.by_image.items():
            for item in items:
                annotations[item["category_id"]] += 1
                images[item["category_id"]].add(image_id)
        rows = [(c.get("name", str(cid)), c.get("supercategory", ""), annotations[cid], len(images[cid]))
                for cid, c in self.categories.items()]
        return sorted(rows, key=lambda row: (-row[2], row[0]))

    def licence_counts(self) -> list[tuple[str, str, int]]:
        """(licence name, url, images): what the images may be used for, per the file itself."""
        counts = Counter(image.get("license") for image in self.images.values())
        rows = []
        for licence_id, count in counts.most_common():
            licence = self.licences.get(licence_id, {})
            fallback = "not stated" if licence_id is None else f"licence {licence_id}"
            rows.append((licence.get("name") or fallback, licence.get("url", ""), count))
        return rows

    # Images -------------------------------------------------------------------------
    def image_ids(self, category: str | None = None, limit: int | None = None) -> list[int]:
        """Image ids in file order, optionally only those with an annotation of ``category``."""
        ids = list(self.images)
        if category is not None:
            wanted = {cid for cid, c in self.categories.items() if c.get("name") == category}
            ids = [i for i in ids if any(a["category_id"] in wanted for a in self.by_image.get(i, ()))]
        return ids[:limit] if limit else ids

    def image(self, image_id: int) -> CocoImage:
        raw = self.images[image_id]
        licence = self.licences.get(raw.get("license"), {}).get("name", "not stated")
        annotations = []
        for item in self.by_image.get(image_id, ()):
            segmentation = item.get("segmentation")
            polygons = tuple(tuple(map(float, p)) for p in segmentation) \
                if isinstance(segmentation, list) else ()
            category = self.categories.get(item["category_id"], {}).get("name", str(item["category_id"]))
            annotations.append(Annotation(
                item["id"], category,
                tuple(map(float, item.get("bbox", (0, 0, 0, 0)))), float(item.get("area", 0.0)),
                bool(item.get("iscrowd", 0)), polygons,
                segmentation if isinstance(segmentation, dict) else None,
                tuple(map(float, item.get("keypoints", ())))))
        return CocoImage(image_id, raw.get("file_name", ""), int(raw.get("width", 0)),
                         int(raw.get("height", 0)), licence, tuple(annotations))

    @staticmethod
    def decode_mask(annotation: Annotation, height: int, width: int) -> Any:
        """The RLE mask as a ``height × width`` uint8 array, or ``None`` (no RLE, or no pycocotools)."""
        if annotation.rle is None:
            return None
        try:
            from pycocotools import mask as mask_utils
        except ImportError:
            return None
        rle = annotation.rle
        if isinstance(rle.get("counts"), list):  # uncompressed RLE
            rle = mask_utils.frPyObjects(rle, height, width)
        return mask_utils.decode(rle)
