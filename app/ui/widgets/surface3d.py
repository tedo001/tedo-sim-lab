"""A height field in 3-D (see :mod:`.view3d` for the camera): shaded quads far to near with a mesh
on them, spheres standing on the surface (drawn faint where the surface hides them), arrows arcing
from one point to another, a colour scale and a read-out of the point under the pointer.

The surface is painted once per view into an image; moving spheres only redraw themselves on top.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QImage, QPainter, QPainterPath, QPen, QPolygonF
from PySide6.QtWidgets import QWidget

from ..theme.ramps import colour_ramp
from ..theme.tokens import COLORS, FONT_FAMILY, MONO_FAMILY
from .view3d import OrbitView, font

__all__ = ["SCALES", "SurfaceView"]

RELIEF = 0.55        # the surface's height relative to its half width
SPHERE = 0.045       # sphere radius, in scene units
MAX_QUADS = 64       # quads along the longer side
SCALES = {"height": ("surface_low", "surface_high"), "density": ("density_low", "density_high")}


def _fmt(value: float) -> str:
    return f"{value:.3g}" if 1e-3 <= abs(value) < 1e4 or value == 0 else f"{value:.2e}"


class SurfaceView(OrbitView):
    def __init__(self, title: str = "", parent: QWidget | None = None) -> None:
        super().__init__(title, parent)
        self.mesh = True
        self.scale = "height"
        self.labels = ("", "")
        self.points = np.zeros((0, 2))
        self.arcs: list[tuple[tuple[float, float], tuple[float, float], float]] = []
        self._data: dict[str, np.ndarray] | None = None
        self._cache: QImage | None = None
        self._screen: np.ndarray | None = None  # vertices on screen (rows × columns × 2)
        self._depth: np.ndarray | None = None
        self._hover: tuple[int, int] | None = None

    # Data -------------------------------------------------------------------------------------------
    def set_surface(self, x: np.ndarray, y: np.ndarray, height: np.ndarray, *,
                    colour: np.ndarray | None = None, scale: str = "height",
                    labels: tuple[str, str] = ("", ""), periodic: bool = True) -> None:
        """``height`` (rows along ``y``) as the surface; ``colour`` (same shape) paints it on
        ``scale`` ("height" or "density"), else the height does. ``labels`` name both for the
        read-out and the legend, units included."""
        x, y, height = np.asarray(x, float), np.asarray(y, float), np.asarray(height, float)
        paint = height if colour is None else np.asarray(colour, float)
        if periodic:  # close the surface: the first row and column again, one step on
            x = np.append(x, x[-1] + (x[-1] - x[-2] if len(x) > 1 else 1.0))
            y = np.append(y, y[-1] + (y[-1] - y[-2] if len(y) > 1 else 1.0))
            height = np.pad(height, ((0, 1), (0, 1)), mode="wrap")
            paint = np.pad(paint, ((0, 1), (0, 1)), mode="wrap")
        stride = max(1, math.ceil((max(len(x), len(y)) - 1) / MAX_QUADS))
        columns = np.unique(np.append(np.arange(0, len(x), stride), len(x) - 1))
        rows = np.unique(np.append(np.arange(0, len(y), stride), len(y) - 1))
        x, y = x[columns], y[rows]
        height, paint = height[np.ix_(rows, columns)], paint[np.ix_(rows, columns)]
        low, high = float(height.min()), float(height.max())
        level = (height - low) / (high - low) if high > low else np.full_like(height, 0.5)
        half = max(x[-1] - x[0], y[-1] - y[0]) / 2 or 1.0
        p_low = min(0.0, float(paint.min())) if scale == "density" else float(paint.min())
        p_high = float(paint.max())
        self._data = {"x": x, "y": y, "height": height, "paint": paint,
                      "u": (x - (x[0] + x[-1]) / 2) / half, "v": (y - (y[0] + y[-1]) / 2) / half,
                      "w": (level - 0.5) * RELIEF, "shade_of": (paint - p_low) / ((p_high - p_low) or 1.0),
                      "half": np.array(half), "range": np.array([p_low, p_high])}
        self.scale, self.labels = scale, labels
        self._hover = None
        self.refit()

    def set_points(self, points: np.ndarray, arcs: Sequence = ()) -> None:
        """Spheres at ``points`` (N × 2, data units) and arrows ((x, y) → (x, y), strength 0..1)."""
        self.points = np.asarray(points, float).reshape(-1, 2)
        self.arcs = list(arcs)
        self.update()

    def set_mesh(self, on: bool) -> None:
        self.mesh = on
        self.changed()

    def changed(self) -> None:
        self._cache = None
        self.update()

    def corners(self) -> np.ndarray:
        if self._data is None:
            return np.zeros((0, 3))
        u, v = self._data["u"], self._data["v"]
        return np.array([(a, b, c) for a in (u[0], u[-1]) for b in (v[0], v[-1])
                         for c in (-RELIEF / 2, RELIEF / 2)])

    # Geometry -------------------------------------------------------------------------------------------
    def surface_at(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """The surface's height w at scene points (bilinear; NaN outside)."""
        data = self._data
        us, vs, w = data["u"], data["v"], data["w"]
        fu = np.interp(u, us, np.arange(len(us)), left=np.nan, right=np.nan)
        fv = np.interp(v, vs, np.arange(len(vs)), left=np.nan, right=np.nan)
        inside = ~(np.isnan(fu) | np.isnan(fv))
        out = np.full(np.shape(u), np.nan)
        if inside.any():
            i0 = np.minimum(np.floor(fu[inside]).astype(int), len(us) - 2)
            j0 = np.minimum(np.floor(fv[inside]).astype(int), len(vs) - 2)
            tu, tv = fu[inside] - i0, fv[inside] - j0
            out[inside] = ((1 - tu) * (1 - tv) * w[j0, i0] + tu * (1 - tv) * w[j0, i0 + 1]
                           + (1 - tu) * tv * w[j0 + 1, i0] + tu * tv * w[j0 + 1, i0 + 1])
        return out

    def _scene(self, points: np.ndarray, lift: float) -> np.ndarray:
        data = self._data
        half = float(data["half"])
        u = (points[:, 0] - (data["x"][0] + data["x"][-1]) / 2) / half
        v = (points[:, 1] - (data["y"][0] + data["y"][-1]) / 2) / half
        w = np.nan_to_num(self.surface_at(u, v), nan=-RELIEF / 2) + lift
        return np.stack([u, v, w], axis=1)

    def hidden(self, points: np.ndarray) -> np.ndarray:
        """Whether the surface stands between the viewer and each scene point."""
        t = np.linspace(0.06, 0.95, 28)[None, :, None]
        ray = points[:, None, :] + t * (self.eye()[None, None, :] - points[:, None, :])
        ground = self.surface_at(ray[..., 0], ray[..., 1])
        return np.any(ground > ray[..., 2] + 0.004, axis=1)

    # Painting -----------------------------------------------------------------------------------------------
    def _render(self) -> QImage:
        ratio = self.devicePixelRatioF()
        image = QImage(max(1, int(self.width() * ratio)), max(1, int(self.height() * ratio)),
                       QImage.Format.Format_ARGB32_Premultiplied)
        image.setDevicePixelRatio(ratio)
        image.fill(Qt.GlobalColor.transparent)
        data = self._data
        u, v = np.meshgrid(data["u"], data["v"])
        screen, depth = self.project(u, v, data["w"])
        self._screen, self._depth = screen, depth
        scene = np.stack([u, v, data["w"]], axis=-1)
        normal = np.cross(scene[1:, 1:] - scene[:-1, :-1], scene[:-1, 1:] - scene[1:, :-1])
        normal /= np.linalg.norm(normal, axis=-1, keepdims=True)
        normal *= np.sign(normal[..., 2:3]) + (normal[..., 2:3] == 0)
        a = math.radians(self.azimuth)  # light from the viewer's upper left, turning with the view
        light = np.array([-0.45 * math.cos(a) - 0.55 * math.sin(a),
                          0.45 * math.sin(a) - 0.55 * math.cos(a), 0.7])
        shading = 0.45 + 0.55 * np.clip(normal @ (light / np.linalg.norm(light)), 0, 1)
        level = data["shade_of"]
        level = (level[1:, 1:] + level[:-1, :-1] + level[1:, :-1] + level[:-1, 1:]) / 4
        ramp = colour_ramp(*(COLORS[k] for k in SCALES[self.scale]))
        rgb = ramp[np.clip((level * 255).round().astype(int), 0, 255)] * shading[..., None]
        if self.mesh:  # each quad's outline is the mesh: its colour moved toward the mesh colour
            outline = rgb + (np.array(QColor(COLORS["surface_mesh"]).getRgbF()[:3]) - rgb) * 0.32
        else:
            outline = rgb
        fill_rgb = (rgb * 255).round().astype(int).tolist()
        line_rgb = (outline * 255).round().astype(int).tolist()
        quad_depth = (depth[1:, 1:] + depth[:-1, :-1] + depth[1:, :-1] + depth[:-1, 1:]) / 4
        columns = quad_depth.shape[1]
        corners = screen.tolist()
        painter = QPainter(image)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        pen = QPen()
        pen.setWidthF(1.0)
        for flat in np.argsort(-quad_depth, axis=None).tolist():
            j, i = divmod(flat, columns)
            a, b, c, d = corners[j][i], corners[j][i + 1], corners[j + 1][i + 1], corners[j + 1][i]
            pen.setColor(QColor(*line_rgb[j][i]))
            painter.setPen(pen)
            painter.setBrush(QColor(*fill_rgb[j][i]))
            painter.drawPolygon(QPolygonF([QPointF(a[0], a[1]), QPointF(b[0], b[1]), QPointF(c[0], c[1]),
                                           QPointF(d[0], d[1])]))
        painter.end()
        return image

    def paintEvent(self, event) -> None:  # noqa: N802 (Qt API)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setFont(font(FONT_FAMILY, 12))
        painter.setPen(QColor(COLORS["text_dim"]))
        painter.drawText(QRectF(4, 0, self.width() - 8, 18), int(Qt.AlignmentFlag.AlignLeft), self.title)
        if self._data is None:
            painter.setPen(QColor(COLORS["text_faint"]))
            painter.drawText(self.rect(), int(Qt.AlignmentFlag.AlignCenter), "Nothing to show yet")
            return
        if self._cache is None or self._cache.deviceIndependentSize().toSize() != self.size():
            self._cache = self._render()
        painter.drawImage(0, 0, self._cache)
        self._paint_spheres(painter)
        self._paint_arcs(painter)
        self._paint_hover(painter)
        self._paint_legend(painter)

    def _paint_spheres(self, painter: QPainter) -> None:
        if not len(self.points):
            return
        scene = self._scene(self.points, SPHERE)
        screen, depth = self.project(scene[:, 0], scene[:, 1], scene[:, 2])
        hidden = self.hidden(scene)
        radius = SPHERE * self.pixels(depth)
        body, rim = QColor(COLORS["electron"]), QColor(COLORS["electron"]).darker(170)
        shine = QColor(COLORS["electron"]).lighter(150)
        for index in np.argsort(-depth):
            centre, r = QPointF(*screen[index]), max(float(radius[index]), 2.5)
            painter.setOpacity(0.28 if hidden[index] else 1.0)
            painter.setPen(QPen(rim, 1.2))
            painter.setBrush(body)
            painter.drawEllipse(centre, r, r)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(shine)
            painter.drawEllipse(QPointF(centre.x() - r * 0.35, centre.y() - r * 0.35), r * 0.32, r * 0.32)
        painter.setOpacity(1.0)

    def _paint_arcs(self, painter: QPainter) -> None:
        for start, end, strength in self.arcs:
            ends = self._scene(np.array([start, end], float), SPHERE * 1.5)
            middle = ends.mean(axis=0)
            middle[2] += 0.12 + 0.25 * float(np.linalg.norm(ends[1, :2] - ends[0, :2]))
            points = np.vstack([ends[0], middle, ends[1]])
            screen, _depth = self.project(points[:, 0], points[:, 1], points[:, 2])
            colour = QColor(COLORS["text"])
            colour.setAlphaF(max(0.0, min(1.0, float(strength))) * 0.9)
            path = QPainterPath(QPointF(*screen[0]))
            path.quadTo(QPointF(*screen[1]), QPointF(*screen[2]))
            pen = QPen(colour, 2)
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            painter.setPen(pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawPath(path)
            direction = screen[2] - screen[1]
            length = float(np.hypot(*direction))
            if length > 1e-6:
                d = direction / length
                tip, side = screen[2], np.array([-d[1], d[0]])
                head = QPolygonF([QPointF(*tip), QPointF(*(tip - 8 * d + 4 * side)),
                                  QPointF(*(tip - 8 * d - 4 * side))])
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(colour)
                painter.drawPolygon(head)

    def _paint_hover(self, painter: QPainter) -> None:
        if self._hover is None or self._screen is None:
            return
        j, i = self._hover
        data = self._data
        painter.setPen(QPen(QColor(COLORS["text"]), 1.5))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawEllipse(QPointF(*self._screen[j, i]), 5, 5)
        height_label, paint_label = self.labels
        parts = [f"x {data['x'][i]:.2f} · y {data['y'][j]:.2f} bohr"]
        if height_label:
            parts.append(f"{height_label} {_fmt(float(data['height'][j, i]))}")
        if paint_label and paint_label != height_label:
            parts.append(f"{paint_label} {_fmt(float(data['paint'][j, i]))}")
        painter.setFont(font(MONO_FAMILY, 11))
        painter.setPen(QColor(COLORS["text"]))
        painter.drawText(QRectF(4, 18, self.width() - 8, 16), int(Qt.AlignmentFlag.AlignLeft),
                         " · ".join(parts))

    def _paint_legend(self, painter: QPainter) -> None:
        data = self._data
        steps = colour_ramp(*(COLORS[k] for k in SCALES[self.scale]), steps=8)
        width = 14
        left, top = self.width() - 8 * width - 8, self.height() - 14
        for index, rgb in enumerate(steps):
            painter.fillRect(QRectF(left + index * width, top, width - 1, 8), QColor.fromRgbF(*rgb))
        low, high = (float(value) for value in data["range"])
        painter.setFont(font(MONO_FAMILY, 10))
        painter.setPen(QColor(COLORS["text_faint"]))
        name = self.labels[1] or self.labels[0]
        painter.drawText(QRectF(4, top - 4, left - 12, 14), int(Qt.AlignmentFlag.AlignRight),
                         f"{name}  {_fmt(low)}")
        painter.drawText(QRectF(left, top - 16, 8 * width, 14), int(Qt.AlignmentFlag.AlignRight), _fmt(high))

    # Hover --------------------------------------------------------------------------------------------------
    def hover_at(self, position: QPointF | None) -> None:
        """The vertex under the pointer: of those within reach, the nearest to the viewer."""
        found = None
        if position is not None and self._screen is not None:
            distance = np.hypot(self._screen[..., 0] - position.x(), self._screen[..., 1] - position.y())
            close = distance < 12
            if close.any():
                depth = np.where(close, self._depth, np.inf)
                j, i = np.unravel_index(int(np.argmin(depth)), depth.shape)
                found = (int(j), int(i))
        if found != self._hover:
            self._hover = found
            self.update()
