"""Multimodal lab: images and words in one space. CLIP labels an image with words you type (zero-shot)
and finds the images in a folder that best match a sentence, drawn as a 3-D map of the folder with
the sentence among them; BLIP writes a caption. Models are downloaded on request with their licences;
everything runs in the inference worker."""

from __future__ import annotations

from typing import Any

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QFileDialog, QHBoxLayout, QLabel, QLineEdit, QPushButton, QVBoxLayout, QWidget

from core.common.vocab import Modality, Task
from labs.common.explore import principal_components

from ...services.context import AppContext
from ..icons import icon
from ..widgets import Card, FlowGrid, Page, PathLabel, label
from ..widgets.plots import BarList
from ..widgets.scatter3d import ScatterView
from .lab_parts import DownloadedModels, LabDatasets, PretrainedModels
from .testing.inputs import IMAGE_FILTER

__all__ = ["MultimodalPage"]

THUMB = 120


def _thumb(path: str, caption: str) -> QWidget:
    tile = QWidget()
    column = QVBoxLayout(tile)
    column.setContentsMargins(0, 0, 0, 0)
    column.setSpacing(2)
    picture = QLabel()
    picture.setPixmap(QPixmap(path).scaled(THUMB, THUMB, Qt.AspectRatioMode.KeepAspectRatio,
                                           Qt.TransformationMode.SmoothTransformation))
    picture.setAlignment(Qt.AlignmentFlag.AlignCenter)
    picture.setToolTip(path)
    column.addWidget(picture)
    column.addWidget(label(caption, "Mono"), 0, Qt.AlignmentFlag.AlignHCenter)
    return tile


class _ImagePick(QWidget):
    def __init__(self, ctx: AppContext) -> None:
        super().__init__()
        self.ctx, self.path = ctx, ""
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        button = QPushButton(icon("image"), "Open image…")
        button.clicked.connect(self.choose)
        self.preview = QLabel()
        self.shown = PathLabel("")
        row.addWidget(button)
        row.addWidget(self.preview)
        row.addWidget(self.shown, 1)

    def choose(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Open an image", str(self.ctx.paths.workspace),
                                              IMAGE_FILTER)
        if path:
            self.set_path(path)

    def set_path(self, path: str) -> None:
        self.path = path
        self.shown.setText(path)
        self.preview.setPixmap(QPixmap(path).scaled(56, 56, Qt.AspectRatioMode.KeepAspectRatio,
                                                    Qt.TransformationMode.SmoothTransformation))


class MultimodalPage(Page):
    def __init__(self, ctx: AppContext, parent: QWidget | None = None) -> None:
        super().__init__("Multimodal", "zero-shot labels, image search by sentence, captions", parent)
        self.ctx = ctx
        self.search_folder = ""

        zero = Card("Zero-shot classification",
                    "CLIP compares the image with “a photo of a {label}” for each label")
        self.clip = DownloadedModels(ctx, {Task.ZERO_SHOT_CLASSIFICATION})
        self.zero_image = _ImagePick(ctx)
        self.labels = QLineEdit("cat, dog, car, bird, flower, a handwritten digit")
        self.labels.setPlaceholderText("labels, separated by commas")
        go = QPushButton(icon("play"), "Classify")
        go.setObjectName("Primary")
        go.clicked.connect(self.classify)
        zero.add(self.clip)
        zero.add(self.zero_image)
        zero.add(self.labels)
        zero.add(go)
        self.zero_result = BarList()
        self.zero_note = label("", "CardCaption", wrap=True)
        zero.add(self.zero_note)
        zero.add(self.zero_result)
        self.body.addWidget(zero)

        search = Card("Image search", "CLIP ranks a folder's images by how well they match a sentence")
        row = QHBoxLayout()
        folder = QPushButton(icon("folder-open"), "Choose folder…")
        folder.clicked.connect(self.choose_folder)
        self.folder_label = PathLabel("")
        row.addWidget(folder)
        row.addWidget(self.folder_label, 1)
        search.add(row)
        self.query = QLineEdit("a dog playing outside")
        go = QPushButton(icon("search"), "Search")
        go.setObjectName("Primary")
        go.clicked.connect(self.search)
        query_row = QHBoxLayout()
        query_row.addWidget(self.query, 1)
        query_row.addWidget(go)
        search.add(query_row)
        self.search_note = label("", "CardCaption", wrap=True)
        search.add(self.search_note)
        self.matches = QWidget()
        self.matches_layout = QVBoxLayout(self.matches)
        self.matches_layout.setContentsMargins(0, 0, 0, 0)
        search.add(self.matches)
        self.map = ScatterView("the folder's images by their CLIP vectors; the sentence is the brightest "
                               "point")
        self.map.setMinimumHeight(280)
        self.map.hide()
        search.add(self.map)
        self.body.addWidget(search)

        caption = Card("Captions", "BLIP describes the image in a sentence; start it yourself to steer it")
        self.blip = DownloadedModels(ctx, {Task.IMAGE_CAPTIONING})
        self.caption_image = _ImagePick(ctx)
        self.prompt = QLineEdit()
        self.prompt.setPlaceholderText("Optional start, e.g. “a picture of”")
        go = QPushButton(icon("captions"), "Describe")
        go.setObjectName("Primary")
        go.clicked.connect(self.describe_image)
        self.caption = label("", "BodyStrong", wrap=True, selectable=True)
        for widget in (self.blip, self.caption_image, self.prompt, go, self.caption):
            caption.add(widget)
        self.body.addWidget(caption)

        self.models = PretrainedModels(ctx, {Task.ZERO_SHOT_CLASSIFICATION, Task.IMAGE_CAPTIONING})
        self.models.changed.connect(self.refresh)
        self.body.addWidget(self.models)
        self.body.addWidget(LabDatasets(ctx, lambda card: card.modality == Modality.MULTIMODAL,
                                        "image–text datasets, listed with their terms; the tools work on "
                                        "your images"))
        self.body.addStretch(1)

    def refresh(self) -> None:
        self.clip.refresh()
        self.blip.refresh()

    def _need(self, chooser: DownloadedModels, note: Any) -> bool:
        if chooser.repo is None:
            note.setText("Download the model below first.")
            return False
        return True

    def classify(self) -> None:
        if not self._need(self.clip, self.zero_note):
            return
        if not self.zero_image.path:
            self.zero_note.setText("Open an image first.")
            return
        labels = [part.strip() for part in self.labels.text().split(",") if part.strip()]
        self.zero_note.setText("Working…")

        def done(result: dict[str, Any]) -> None:
            self.zero_note.setText(f"Best match: {result['labels'][0]} ({result['probabilities'][0]:.0%})")
            self.zero_result.set_rows([[name, p] for name, p in zip(result["labels"], result["probabilities"],
                                                                   strict=True)])

        self.ctx.inference.request("zero_shot", {"model": self.clip.repo, "image": self.zero_image.path,
                                                 "labels": labels}, on_done=done,
                                   on_error=lambda m: self.zero_note.setText(f"Failed: {m}"))

    def choose_folder(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Choose a folder of images",
                                                str(self.ctx.paths.workspace))
        if path:
            self.search_folder = path
            self.folder_label.setText(path)

    def search(self) -> None:
        if not self._need(self.clip, self.search_note):
            return
        if not self.search_folder:
            self.search_note.setText("Choose a folder of images first.")
            return
        self.search_note.setText("Reading the images (once; later searches are instant)…")

        def partial(data: dict[str, Any]) -> None:
            self.search_note.setText(f"{data.get('done', 0)} of {data.get('total', 0)} images read…")

        self.ctx.inference.request("image_search", {"model": self.clip.repo, "folder": self.search_folder,
                                                    "query": self.query.text()},
                                   on_partial=partial, on_done=self._found,
                                   on_error=lambda m: self.search_note.setText(f"Failed: {m}"))

    def _found(self, result: dict[str, Any]) -> None:
        while self.matches_layout.count():
            widget = self.matches_layout.takeAt(0).widget()
            if widget is not None:
                widget.deleteLater()
        files, scores = result["files"], result["scores"]
        if not files:
            self.search_note.setText("No images (PNG, JPEG, BMP, WebP, GIF) in that folder.")
            self.map.hide()
            return
        self.search_note.setText(f"The best {min(12, len(files))} of {len(files)} images; cosine similarity "
                                 "under each.")
        self.matches_layout.addWidget(FlowGrid([_thumb(path, f"{score:.3f}") for path, score in
                                                zip(files[:12], scores[:12],
                                                    strict=True)], THUMB + 8, fewest=2))
        vectors = np.asarray(result["vectors"] + [result["query_vector"]], dtype=float)
        if len(vectors) >= 4:
            points, _share = principal_components(vectors, 3, standardize=False)
            similarity = list(vectors[:-1] @ vectors[-1]) + [1.0]
            self.map.set_points(points, values=similarity, value_label="similarity to the sentence",
                                axes=("PC 1", "PC 2", "PC 3"))
            self.map.show()

    def describe_image(self) -> None:
        if not self._need(self.blip, self.caption):
            return
        if not self.caption_image.path:
            self.caption.setText("Open an image first.")
            return
        self.caption.setText("Looking…")
        self.ctx.inference.request("caption", {"model": self.blip.repo, "image": self.caption_image.path,
                                               "prompt": self.prompt.text()},
                                   on_done=lambda r: self.caption.setText(r["caption"] or "(no caption)"),
                                   on_error=lambda m: self.caption.setText(f"Failed: {m}"))
