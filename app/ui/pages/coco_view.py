"""COCO annotation inspector: open a COCO-format annotation file you already have (and, if you
like, the folder of its images) to see its categories, the licences its images carry, and any
image's boxes, polygons and masks. Reading a large file runs in the background; the lab never
downloads COCO itself."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtGui import QImage
from PySide6.QtWidgets import QComboBox, QFileDialog, QHBoxLayout, QPushButton, QVBoxLayout, QWidget

from labs.computer_vision.coco import CocoFile

from ...services.context import AppContext
from ..icons import icon
from ..widgets import Card, DataTable, ResponsiveRow, StatStrip, StatTile, label
from ..widgets.annotated import AnnotatedImage
from ..widgets.plots import BarList

__all__ = ["CocoView"]

#: Image rows listed at once; the category filter narrows big files.
IMAGE_ROWS = 500


class CocoView(QWidget):
    def __init__(self, ctx: AppContext, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.ctx = ctx
        self.coco: CocoFile | None = None
        self.images_dir: Path | None = None
        self.image_ids: list[int] = []
        self._job: str | None = None
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        intro = Card("COCO annotation inspector", "local files only; the lab never downloads COCO in bulk")
        row = QHBoxLayout()
        open_button = QPushButton(icon("file-json"), "Open annotation file…")
        open_button.clicked.connect(self._ask_file)
        self.folder_button = QPushButton(icon("folder-open"), "Choose images folder…")
        self.folder_button.clicked.connect(self._ask_folder)
        row.addWidget(open_button)
        row.addWidget(self.folder_button)
        row.addStretch(1)
        intro.add(row)
        self.state = label("Open a COCO-format JSON file, for example annotations/instances_val2017.json "
                           "from cocodataset.org, or your own export. The images folder is optional: without "
                           "it the annotations are drawn on a blank canvas.", "CardCaption", wrap=True)
        intro.add(self.state)
        layout.addWidget(intro)

        titles = (("images", "Images"), ("annotations", "Annotations"), ("categories", "Categories"),
                  ("licences", "Image licences"))
        self.tiles = {key: StatTile(title) for key, title in titles}
        self.stats = StatStrip(self.tiles)
        layout.addWidget(self.stats)
        categories = Card("Categories", "annotations per category (top 15)")
        self.category_bars = BarList()
        categories.add(self.category_bars)
        licences = Card("Image licences", "as listed in the file: check them before any commercial use",
                        padded=False)
        self.licences = DataTable(("Licence", "Images"), mono_columns=(1,), stretch_column=0)
        licences.add(self.licences)
        self.summary = ResponsiveRow([(categories, 1), (licences, 1)], breakpoint=900)
        layout.addWidget(self.summary)

        browse = Card("Images", padded=False)
        self.category = QComboBox()
        self.category.setMaximumWidth(240)
        self.category.currentIndexChanged.connect(lambda _: self._fill_images())
        browse.add_head_widget(self.category)
        self.images = DataTable(("File", "Size", "Objects"), mono_columns=(0, 1, 2), stretch_column=0)
        self.images.itemSelectionChanged.connect(self._image_chosen)
        browse.add(self.images)
        viewer = Card("Image")
        self.canvas = AnnotatedImage()
        self.canvas.setMinimumHeight(360)
        viewer.add(self.canvas)
        self.image_note = label("", "CardCaption", wrap=True)
        viewer.add(self.image_note)
        self.browser = ResponsiveRow([(browse, 2), (viewer, 3)], breakpoint=900)
        layout.addWidget(self.browser)
        ctx.jobs.job_finished.connect(self._loaded)
        self._show_loaded(False)

    def _show_loaded(self, loaded: bool) -> None:
        for widget in (self.stats, self.summary, self.browser):
            widget.setVisible(loaded)
        self.folder_button.setEnabled(loaded)

    # Files ---------------------------------------------------------------------------
    def _ask_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Open COCO annotations", str(self.ctx.paths.datasets),
                                              "COCO annotations (*.json)")
        if path:
            self.open_file(Path(path))

    def _ask_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Images folder", str(self.ctx.paths.datasets))
        if folder:
            self.set_images_dir(Path(folder))

    def open_file(self, path: Path) -> str:
        """Read ``path`` in the background; returns the job id."""
        self.state.setText(f"Reading {path.name}…")
        self._job = self.ctx.jobs.submit_task(lambda cancel, progress: CocoFile.load(path),
                                              title=f"Read {path.name}")
        return self._job

    def set_images_dir(self, folder: Path) -> None:
        self.images_dir = folder
        self._image_chosen()

    def _loaded(self, job_id: str, status: str) -> None:
        if job_id != self._job:
            return
        self._job = None
        job = self.ctx.jobs.job(job_id)
        if status != "completed":
            self.state.setText(f"Could not read it: {job.error}")
            return
        self.coco = coco = job.result
        description = coco.info.get("description") or coco.path.name
        self.state.setText(f"{coco.path} · {description}")
        self.tiles["images"].set(f"{len(coco.images):,}")
        self.tiles["annotations"].set(f"{coco.annotation_count:,}")
        self.tiles["categories"].set(f"{len(coco.categories):,}")
        licence_rows = coco.licence_counts()
        self.tiles["licences"].set(f"{len(licence_rows):,}")
        counts = coco.category_counts()
        self.category_bars.set_rows([(name, count) for name, _super, count, _images in counts])
        self.licences.clear_rows()
        for name, _url, count in licence_rows:
            self.licences.add_row((name, f"{count:,}"))
        self.category.blockSignals(True)
        self.category.clear()
        self.category.addItem("All categories", None)
        for name, _super, count, _images in counts:
            if count:
                self.category.addItem(f"{name} ({count:,})", name)
        self.category.blockSignals(False)
        self._show_loaded(True)
        self._fill_images()

    # Browsing ------------------------------------------------------------------------
    def _fill_images(self) -> None:
        if self.coco is None:
            return
        self.image_ids = self.coco.image_ids(self.category.currentData(), limit=IMAGE_ROWS)
        self.images.blockSignals(True)
        self.images.clear_rows()
        for image_id in self.image_ids:
            raw = self.coco.images[image_id]
            size = f"{raw.get('width')}×{raw.get('height')}"
            self.images.add_row((raw.get("file_name", str(image_id)), size,
                                 str(len(self.coco.by_image.get(image_id, ())))))
        self.images.blockSignals(False)
        if self.image_ids:
            self.images.selectRow(0)

    def _image_chosen(self) -> None:
        rows = self.images.selectionModel().selectedRows() if self.coco else []
        if not rows:
            return
        image = self.coco.image(self.image_ids[rows[0].row()])
        picture = None
        note = f"{image.file_name} · {len(image.annotations)} objects · licence: {image.licence}"
        if self.images_dir is not None:
            path = self.images_dir / image.file_name
            picture = QImage(str(path)) if path.is_file() else None
            if picture is None:
                note += f" · {image.file_name} is not in {self.images_dir}"
        else:
            note += " · choose the images folder to see the picture"
        masks = {}
        for index, annotation in enumerate(image.annotations):
            mask = CocoFile.decode_mask(annotation, image.height, image.width)
            if mask is not None:
                masks[index] = mask
        self.canvas.set_content(picture, image.width, image.height, image.annotations, masks)
        self.image_note.setText(note)
