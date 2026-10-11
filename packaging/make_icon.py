"""Draw the app icon (packaging/tedo-lab.ico): a rounded square in the interface's accent blue
with a white "T". Run from the repository root: ``python packaging/make_icon.py``."""

import struct
import tempfile
from pathlib import Path

from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QColor, QFont, QGuiApplication, QImage, QPainter

from app.ui.theme import load_fonts
from app.ui.theme.tokens import COLORS

SIZES = (16, 24, 32, 48, 64, 128, 256)


def draw(size: int) -> QImage:
    image = QImage(size, size, QImage.Format.Format_ARGB32)
    image.fill(Qt.GlobalColor.transparent)
    painter = QPainter(image)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.setRenderHint(QPainter.RenderHint.TextAntialiasing)
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(QColor(COLORS["accent"]))
    margin = size * 0.04
    side = size - 2 * margin
    painter.drawRoundedRect(QRectF(margin, margin, side, side), size * 0.22, size * 0.22)
    font = QFont("Inter")
    font.setPixelSize(round(size * 0.66))
    font.setWeight(QFont.Weight.Bold)
    painter.setFont(font)
    painter.setPen(QColor("#FFFFFF"))
    painter.drawText(QRectF(0, -size * 0.02, size, size), Qt.AlignmentFlag.AlignCenter, "T")
    painter.end()
    return image


def main() -> None:
    app = QGuiApplication([])  # noqa: F841 (fonts and painting need it)
    load_fonts()
    out = Path(__file__).resolve().parent
    largest = draw(256)
    largest.save(str(out / "tedo-lab.png"))
    # Qt's ICO writer stores one image per file; write each size and join them into one .ico.
    images = []
    with tempfile.TemporaryDirectory() as folder:
        for size in SIZES:
            path = Path(folder) / f"{size}.png"
            draw(size).save(str(path), "PNG")
            images.append((size, path.read_bytes()))
    header = struct.pack("<HHH", 0, 1, len(images))
    offset = 6 + 16 * len(images)
    entries, blobs = b"", b""
    for size, data in images:
        entries += struct.pack("<BBBBHHII", size % 256, size % 256, 0, 0, 1, 32, len(data),
                               offset + len(blobs))
        blobs += data
    (out / "tedo-lab.ico").write_bytes(header + entries + blobs)
    print(f"wrote {out / 'tedo-lab.ico'} ({', '.join(map(str, SIZES))} px)")


if __name__ == "__main__":
    main()
