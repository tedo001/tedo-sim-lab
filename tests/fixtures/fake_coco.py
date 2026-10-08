"""A tiny COCO-format annotation file (two images, three categories, two licences, polygons
and one RLE crowd mask) for the COCO inspector tests."""

from __future__ import annotations

import json
from pathlib import Path

__all__ = ["write_fake_coco"]


def write_fake_coco(folder: Path, *, with_images: bool = False) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    data = {
        "info": {"description": "fake COCO for tests"},
        "licenses": [{"id": 1, "name": "Attribution License",
                      "url": "http://creativecommons.org/licenses/by/2.0/"},
                     {"id": 2, "name": "Attribution-NonCommercial License",
                      "url": "http://creativecommons.org/licenses/by-nc/2.0/"}],
        "categories": [{"id": 1, "name": "person", "supercategory": "person"},
                       {"id": 3, "name": "car", "supercategory": "vehicle"},
                       {"id": 18, "name": "dog", "supercategory": "animal"}],
        "images": [{"id": 10, "file_name": "a.png", "width": 64, "height": 48, "license": 1},
                   {"id": 11, "file_name": "b.png", "width": 32, "height": 32, "license": 2}],
        "annotations": [
            {"id": 100, "image_id": 10, "category_id": 1, "bbox": [4, 4, 20, 30], "area": 400.0, "iscrowd": 0,
             "segmentation": [[4, 4, 24, 4, 24, 34, 4, 34]]},
            {"id": 101, "image_id": 10, "category_id": 3, "bbox": [30, 10, 20, 10], "area": 150.0,
             "iscrowd": 0, "segmentation": [[30, 10, 50, 10, 50, 20, 30, 20]]},
            {"id": 102, "image_id": 11, "category_id": 1, "bbox": [0, 0, 8, 8], "area": 64.0, "iscrowd": 1,
             "segmentation": {"size": [32, 32], "counts": [0, 8, 24, 8, 24, 8, 24, 8, 24, 8, 24, 8, 24, 8, 24,
                                                            8, 792]}},
        ],
    }
    path = folder / "instances_fake.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    if with_images:
        from PySide6.QtGui import QColor, QImage

        for name, (width, height) in (("a.png", (64, 48)), ("b.png", (32, 32))):
            image = QImage(width, height, QImage.Format.Format_RGB32)
            image.fill(QColor("#336699"))
            image.save(str(folder / name))
    return path
