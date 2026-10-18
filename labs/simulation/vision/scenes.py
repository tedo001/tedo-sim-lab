"""Synthetic scenes for detection and segmentation experiments: circles, squares and triangles of
random size, colour and place on a textured background, with exact boxes and outlines. A dataset
is written as PNG images plus a COCO ``annotations.json`` the Dataset Hub's COCO inspector reads.

Shapes drawn later hide what is behind them. Annotations describe what is visible: an object's
box and area come from its visible pixels, its segmentation is the outline polygon when nothing
covers it and an (uncompressed) RLE mask of the visible part when something does; an object that
is hidden entirely is left out. With ``overlap`` off, shapes are placed apart instead.

Everything follows from the seed, so a dataset can be made again bit for bit.
"""

from __future__ import annotations

import json
import math
import shutil
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from core.common.cancel import CancelToken

from .png import write_png

__all__ = ["CATEGORIES", "SceneSpec", "generate_dataset", "render_scene", "rle_counts"]

CATEGORIES = ("circle", "square", "triangle")


@dataclass
class SceneSpec:
    width: int = 160
    height: int = 120
    min_objects: int = 1
    max_objects: int = 4
    min_size: int = 14
    max_size: int = 44
    shapes: tuple[str, ...] = CATEGORIES
    noise: float = 0.04
    background: str = "gradient"  # gradient, plain, texture
    seed: int = 0
    #: Shapes may cover each other (annotations then describe the visible parts).
    overlap: bool = True
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        data = asdict(self)
        data["shapes"] = list(self.shapes)
        return data

    @classmethod
    def from_dict(cls, data: dict) -> SceneSpec:
        known = {key: data[key] for key in cls.__dataclass_fields__ if key in data}
        if "shapes" in known:
            known["shapes"] = tuple(known["shapes"])
        return cls(**known)

    def check(self) -> None:
        if not (16 <= self.width <= 2048 and 16 <= self.height <= 2048):
            raise ValueError("width and height are 16 to 2048 pixels")
        if not 0 <= self.min_objects <= self.max_objects <= 50:
            raise ValueError("objects: 0 ≤ min ≤ max ≤ 50")
        if not 4 <= self.min_size <= self.max_size <= min(self.width, self.height):
            raise ValueError("sizes: 4 ≤ min ≤ max ≤ the image's shorter side")
        if not self.shapes or any(s not in CATEGORIES for s in self.shapes):
            raise ValueError(f"shapes are some of {', '.join(CATEGORIES)}")


def _background(spec: SceneSpec, rng: np.random.Generator) -> np.ndarray:
    y, x = np.mgrid[0:spec.height, 0:spec.width].astype(np.float32)
    x, y = x / max(spec.width - 1, 1), y / max(spec.height - 1, 1)
    base = rng.uniform(0.1, 0.5, 3).astype(np.float32)
    if spec.background == "plain":
        image = np.broadcast_to(base, (spec.height, spec.width, 3)).copy()
    elif spec.background == "texture":
        image = base + 0.08 * np.sin(2 * np.pi * (rng.uniform(4, 12) * x + rng.uniform(4, 12) * y))[..., None]
    else:
        tilt = rng.uniform(-0.25, 0.25, 3).astype(np.float32)
        image = base + x[..., None] * tilt + y[..., None] * tilt[::-1]
    return np.clip(image, 0, 1).astype(np.float32)


def _shape_mask(kind: str, cx: float, cy: float, size: float, angle: float, shape: tuple[int, int]):
    """(mask, polygon as [x0, y0, x1, y1, …])."""
    height, width = shape
    y, x = np.mgrid[0:height, 0:width].astype(np.float32) + 0.5
    if kind == "circle":
        radius = size / 2
        mask = (x - cx) ** 2 + (y - cy) ** 2 <= radius ** 2
        points = [(cx + radius * math.cos(t), cy + radius * math.sin(t))
                  for t in np.linspace(0, 2 * math.pi, 24, endpoint=False)]
        return mask, points
    count = 4 if kind == "square" else 3
    radius = size / 2 * (math.sqrt(2) if kind == "square" else 1.15)
    start = angle + (math.pi / 4 if kind == "square" else -math.pi / 2)
    points = [(cx + radius * math.cos(start + 2 * math.pi * i / count),
               cy + radius * math.sin(start + 2 * math.pi * i / count)) for i in range(count)]
    mask = np.ones(shape, dtype=bool)
    for (x1, y1), (x2, y2) in zip(points, points[1:] + points[:1], strict=True):
        mask &= (x2 - x1) * (y - y1) - (y2 - y1) * (x - x1) >= 0  # inside every edge (clockwise)
    return mask, points


def rle_counts(mask: np.ndarray) -> dict:
    """COCO's uncompressed RLE of a boolean mask: run lengths down the columns, zeros first."""
    flat = np.asarray(mask, dtype=bool).ravel(order="F")
    edges = np.flatnonzero(flat[1:] != flat[:-1]) + 1
    counts = np.diff(np.concatenate([[0], edges, [flat.size]])).tolist()
    return {"counts": ([0] if flat[0] else []) + counts, "size": [int(mask.shape[0]), int(mask.shape[1])]}


def render_scene(spec: SceneSpec, rng: np.random.Generator) -> tuple[np.ndarray, list[dict]]:
    """One image (H, W, 3) and its visible objects: category, bbox [x, y, w, h], area, segmentation."""
    spec.check()
    image = _background(spec, rng)
    owner = np.full(image.shape[:2], -1, dtype=np.int32)  # which shape each pixel shows
    drawn: list[tuple[str, np.ndarray, list]] = []
    for _ in range(int(rng.integers(spec.min_objects, spec.max_objects + 1))):
        kind = str(rng.choice(spec.shapes))
        size = float(rng.uniform(spec.min_size, spec.max_size))
        for _attempt in range(1 if spec.overlap else 30):
            cx = float(rng.uniform(size / 2, spec.width - size / 2))
            cy = float(rng.uniform(size / 2, spec.height - size / 2))
            angle = float(rng.uniform(0, 2 * math.pi))
            mask, points = _shape_mask(kind, cx, cy, size, angle, image.shape[:2])
            if spec.overlap or not (mask & (owner >= 0)).any():
                break
        else:
            continue  # no free place left for it
        if not mask.any():
            continue
        colour = rng.uniform(0.55, 1.0, 3) * rng.permutation([1.0, 0.6, 0.25])
        image[mask] = colour.astype(np.float32)
        owner[mask] = len(drawn)
        drawn.append((kind, mask, points))
    objects = []
    for index, (kind, mask, points) in enumerate(drawn):
        visible = owner == index
        if not visible.any():
            continue  # covered entirely by later shapes
        ys, xs = np.nonzero(visible)
        x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
        area = int(visible.sum())
        if area == int(mask.sum()):
            segmentation = [[round(min(max(v, 0.0), limit), 2) for px, py in points
                             for v, limit in ((px, spec.width), (py, spec.height))]]
        else:
            segmentation = rle_counts(visible)
        objects.append({"category": kind, "bbox": [x0, y0, x1 - x0, y1 - y0], "area": area,
                        "segmentation": segmentation})
    if spec.noise > 0:
        image = image + rng.normal(0, spec.noise, image.shape).astype(np.float32)
    return np.clip(image, 0, 1).astype(np.float32), objects


def generate_dataset(spec: SceneSpec, count: int, folder: Path, name: str = "synthetic shapes",
                     progress=None, cancel: CancelToken | None = None) -> Path:
    """``count`` images in ``folder/images`` and ``folder/annotations.json`` (COCO); returns the
    annotations file. The dataset is written beside ``folder`` and moved there when complete, so
    a cancelled or failed run leaves nothing behind."""
    if not 1 <= count <= 100_000:
        raise ValueError("count is 1 to 100 000")
    spec.check()
    folder = Path(folder)
    if folder.exists() and any(folder.iterdir()):
        raise ValueError(f"{folder} already holds files: choose another name")
    partial = folder.with_name(f".{folder.name}.partial")
    shutil.rmtree(partial, ignore_errors=True)
    try:
        _write_dataset(spec, count, partial, name, progress, cancel)
        if folder.exists():
            folder.rmdir()  # empty, checked above
        partial.rename(folder)
    except BaseException:
        shutil.rmtree(partial, ignore_errors=True)
        raise
    return folder / "annotations.json"


def _write_dataset(spec: SceneSpec, count: int, folder: Path, name: str, progress,
                   cancel: CancelToken | None) -> None:
    rng = np.random.default_rng(spec.seed)
    images_dir = folder / "images"
    categories = [{"id": i + 1, "name": c, "supercategory": "shape"} for i, c in enumerate(CATEGORIES)]
    ids = {c["name"]: c["id"] for c in categories}
    coco = {"info": {"description": name, "version": "1.0", "year": datetime.now(UTC).year,
                     "contributor": "TEDO AI Research Lab (generated)",
                     "date_created": datetime.now(UTC).isoformat(timespec="seconds"),
                     "settings": spec.to_dict()},
            "licenses": [{"id": 1, "name": "Generated by you with the lab's scene generator", "url": ""}],
            "images": [], "annotations": [], "categories": categories}
    annotation_id = 1
    for index in range(1, count + 1):
        if cancel is not None:
            cancel.raise_if_cancelled()
        image, objects = render_scene(spec, rng)
        file_name = f"{index:06d}.png"
        write_png(images_dir / file_name, image)
        coco["images"].append({"id": index, "file_name": f"images/{file_name}", "width": spec.width,
                               "height": spec.height, "license": 1})
        for item in objects:
            coco["annotations"].append({"id": annotation_id, "image_id": index,
                                        "category_id": ids[item["category"]], "bbox": item["bbox"],
                                        "area": item["area"], "segmentation": item["segmentation"],
                                        "iscrowd": 0})
            annotation_id += 1
        if progress is not None:
            progress(index / count, f"{index} of {count} images")
    images_dir.mkdir(parents=True, exist_ok=True)
    (folder / "annotations.json").write_text(json.dumps(coco, indent=1), encoding="utf-8")
