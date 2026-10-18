"""A pinhole camera with radial lens distortion: how 3-D points land on the image.

x_c = R·X + t (the world in camera coordinates), then x = x_c/z_c, y = y_c/z_c, distorted by
d = 1 + k1·r² + k2·r⁴, and u = f·x·d + c_x, v = f·y·d + c_y. The scene objects are wireframes
(a cube, a ground grid, the world axes) whose edges are sampled densely, so distortion bends them.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

__all__ = ["OBJECTS", "Camera", "intrinsics", "project", "projection_matrix", "rotation", "wireframe"]


@dataclass
class Camera:
    focal: float = 520.0
    cx: float = 320.0
    cy: float = 240.0
    width: int = 640
    height: int = 480
    #: Rotation in degrees: yaw about y, pitch about x, roll about z (applied yaw, pitch, roll).
    yaw: float = 25.0
    pitch: float = -20.0
    roll: float = 0.0
    #: The world origin in camera coordinates.
    tx: float = 0.0
    ty: float = 0.0
    tz: float = 6.0
    k1: float = 0.0
    k2: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> Camera:
        return cls(**{key: data[key] for key in cls.__dataclass_fields__ if key in data})


def rotation(yaw: float, pitch: float, roll: float) -> np.ndarray:
    a, b, c = np.deg2rad([yaw, pitch, roll])
    ry = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
    rx = np.array([[1, 0, 0], [0, np.cos(b), -np.sin(b)], [0, np.sin(b), np.cos(b)]])
    rz = np.array([[np.cos(c), -np.sin(c), 0], [np.sin(c), np.cos(c), 0], [0, 0, 1]])
    return rz @ rx @ ry


def intrinsics(camera: Camera) -> np.ndarray:
    return np.array([[camera.focal, 0, camera.cx], [0, camera.focal, camera.cy], [0, 0, 1]])


def projection_matrix(camera: Camera) -> np.ndarray:
    """P = K [R | t] (3 × 4), the distortion-free projection."""
    extrinsic = np.hstack([rotation(camera.yaw, camera.pitch, camera.roll),
                           np.array([[camera.tx], [camera.ty], [camera.tz]])])
    return intrinsics(camera) @ extrinsic


def project(camera: Camera, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(N × 2 pixel coordinates, N visible flags: in front of the camera and inside the image)."""
    world = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    turn = rotation(camera.yaw, camera.pitch, camera.roll)
    cam = world @ turn.T + np.array([camera.tx, camera.ty, camera.tz])
    z = cam[:, 2]
    in_front = z > 1e-6
    safe = np.where(in_front, z, 1.0)
    x, y = cam[:, 0] / safe, cam[:, 1] / safe
    r2 = x ** 2 + y ** 2
    d = 1 + camera.k1 * r2 + camera.k2 * r2 ** 2
    uv = np.stack([camera.focal * x * d + camera.cx, camera.focal * y * d + camera.cy], axis=1)
    inside = (uv[:, 0] >= 0) & (uv[:, 0] < camera.width) & (uv[:, 1] >= 0) & (uv[:, 1] < camera.height)
    return uv, in_front & inside


def _segments_cube() -> list[tuple[np.ndarray, np.ndarray]]:
    corners = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], dtype=float)
    return [(corners[i], corners[j]) for i in range(8) for j in range(i + 1, 8)
            if np.sum(np.abs(corners[i] - corners[j])) == 2]


def _segments_grid(half: int = 4) -> list[tuple[np.ndarray, np.ndarray]]:
    lines = []
    for value in range(-half, half + 1):
        lines.append((np.array([value, 1.0, -half]), np.array([value, 1.0, half])))
        lines.append((np.array([-half, 1.0, value]), np.array([half, 1.0, value])))
    return lines


def _segments_axes() -> list[tuple[np.ndarray, np.ndarray]]:
    origin = np.zeros(3)
    return [(origin, np.eye(3)[i] * 1.5) for i in range(3)]


OBJECTS = {"Cube": _segments_cube, "Ground grid": _segments_grid, "Axes": _segments_axes}


def wireframe(camera: Camera, objects: tuple[str, ...] = ("Cube", "Ground grid", "Axes"),
              samples: int = 24) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """(object name, N × 2 points along one edge, visible flags) for every edge."""
    out = []
    for name in objects:
        for start, end in OBJECTS[name]():
            t = np.linspace(0.0, 1.0, samples)[:, None]
            uv, visible = project(camera, start + t * (end - start))
            out.append((name, uv, visible))
    return out
