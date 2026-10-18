"""PNG files with the standard library (zlib): write 8-bit grey, RGB or RGBA images, and read the
ones this module and most tools write (8-bit, any filter, no interlacing). Enough to save
synthetic datasets and results without an imaging library in the window."""

from __future__ import annotations

import struct
import zlib
from pathlib import Path

import numpy as np

__all__ = ["encode_png", "read_png", "to_uint8", "write_png"]

_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_COLOUR_TYPES = {1: 0, 3: 2, 4: 6}   # channels → PNG colour type
_CHANNELS = {0: 1, 2: 3, 4: 2, 6: 4}


def to_uint8(image: np.ndarray) -> np.ndarray:
    if image.dtype == np.uint8:
        return image
    return (np.clip(image, 0.0, 1.0) * 255 + 0.5).astype(np.uint8)


def _chunk(kind: bytes, data: bytes) -> bytes:
    checksum = zlib.crc32(kind + data) & 0xFFFFFFFF
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", checksum)


def encode_png(image: np.ndarray) -> bytes:
    """The PNG file's bytes for an image (float in [0, 1] or uint8; grey, RGB or RGBA)."""
    pixels = to_uint8(image)
    if pixels.ndim == 2:
        pixels = pixels[..., None]
    height, width, channels = pixels.shape
    if channels not in _COLOUR_TYPES:
        raise ValueError("an image is grey, RGB or RGBA")
    rows = np.concatenate([np.zeros((height, 1), dtype=np.uint8), pixels.reshape(height, -1)], axis=1)
    header = struct.pack(">IIBBBBB", width, height, 8, _COLOUR_TYPES[channels], 0, 0, 0)
    return _SIGNATURE + _chunk(b"IHDR", header) + _chunk(b"IDAT", zlib.compress(rows.tobytes(), 6)) + \
        _chunk(b"IEND", b"")


def write_png(path: Path, image: np.ndarray) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encode_png(image))
    return path


def read_png(path: Path) -> np.ndarray:
    """uint8 pixels, (H, W) or (H, W, channels)."""
    data = Path(path).read_bytes()
    if not data.startswith(_SIGNATURE):
        raise ValueError(f"{path} is not a PNG file")
    position, idat, header = 8, b"", None
    while position < len(data):
        length, kind = struct.unpack(">I4s", data[position:position + 8])
        body = data[position + 8:position + 8 + length]
        if kind == b"IHDR":
            header = struct.unpack(">IIBBBBB", body)
        elif kind == b"IDAT":
            idat += body
        elif kind == b"IEND":
            break
        position += 12 + length
    if header is None:
        raise ValueError(f"{path} has no header")
    width, height, depth, colour, _, _, interlace = header
    if depth != 8 or interlace or colour not in _CHANNELS:
        raise ValueError("only 8-bit, non-interlaced grey/RGB/RGBA PNG files are read here")
    channels = _CHANNELS[colour]
    stride = width * channels
    raw = np.frombuffer(zlib.decompress(idat), dtype=np.uint8).reshape(height, stride + 1)
    out = np.zeros((height, stride), dtype=np.int32)
    for y in range(height):
        kind, line = raw[y, 0], raw[y, 1:].astype(np.int32)
        previous = out[y - 1] if y else np.zeros(stride, dtype=np.int32)
        if kind == 0:
            out[y] = line
        elif kind == 2:
            out[y] = (line + previous) & 255
        else:
            row = np.zeros(stride, dtype=np.int32)
            for x in range(stride):
                left = row[x - channels] if x >= channels else 0
                up = previous[x]
                corner = previous[x - channels] if x >= channels else 0
                if kind == 1:
                    predictor = left
                elif kind == 3:
                    predictor = (left + up) // 2
                else:  # Paeth
                    p = left + up - corner
                    pa, pb, pc = abs(p - left), abs(p - up), abs(p - corner)
                    predictor = left if pa <= pb and pa <= pc else up if pb <= pc else corner
                row[x] = (line[x] + predictor) & 255
            out[y] = row
    pixels = out.astype(np.uint8).reshape(height, width, channels)
    return pixels[..., 0] if channels == 1 else pixels
