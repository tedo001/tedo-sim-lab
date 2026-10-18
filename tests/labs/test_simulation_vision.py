"""Vision engines (image operations, PNG files, synthetic scenes, the camera), saved simulations,
and the exported scripts, which must run on their own."""

from __future__ import annotations

import json
import os
import subprocess
import sys

import numpy as np
import pytest

from core.common.cancel import Cancelled, CancelToken
from core.common.paths import CODE_ROOT
from labs.simulation import records
from labs.simulation.quantum import preset
from labs.simulation.quantum.waves import WaveSetup
from labs.simulation.scripts import script_for
from labs.simulation.vision import (
    OPERATIONS,
    SAMPLES,
    Camera,
    SceneSpec,
    Step,
    generate_dataset,
    project,
    projection_matrix,
    render_scene,
    run_pipeline,
    sample,
)
from labs.simulation.vision import filters as f
from labs.simulation.vision.ops import Param
from labs.simulation.vision.png import read_png, write_png
from labs.simulation.vision.scenes import rle_counts


@pytest.mark.parametrize("op", sorted(OPERATIONS))
def test_every_operation_returns_a_valid_image(op: str) -> None:
    for image in (sample("Shapes", 64), sample("Checkerboard", 64)):
        out = OPERATIONS[op].run(image, OPERATIONS[op].defaults())
        assert out.dtype == np.float32 and out.ndim in (2, 3) and 0 <= out.min() and out.max() <= 1


def test_classic_operations_behave() -> None:
    bimodal = np.concatenate([np.full(500, 0.2), np.full(500, 0.8)]).reshape(25, 40).astype(np.float32)
    assert 0.2 < f.otsu_level(bimodal) < 0.8
    square = np.zeros((40, 40), dtype=np.float32)
    square[10:30, 10:30] = 1.0
    edges = f.canny(square, sigma=1.0)
    assert edges[20, 20] == 0 and edges[2, 2] == 0 and edges[8:13, 20].max() == 1  # on the border only
    assert len({tuple(c) for c in f.kmeans_colours(sample("Shapes", 64), k=3).reshape(-1, 3)}) <= 3
    noisy = f.noise_gaussian(square, 0.2, seed=4)
    assert np.array_equal(noisy, f.noise_gaussian(square, 0.2, seed=4))
    zone = sample("Zone plate", 128)
    high = lambda image: np.abs(np.fft.fft2(image))[32:96, 32:96].sum()  # noqa: E731
    assert high(f.frequency_filter(zone, "low", 0.05)) < 0.2 * high(zone)
    assert f.threshold(bimodal, method="otsu").mean() == pytest.approx(0.5)


def test_parameters_are_cleaned_and_pipelines_are_data() -> None:
    sigma = Param("sigma", "Sigma", "float", 1.0, 0.1, 5.0)
    assert sigma.clean(99) == 5.0 and sigma.clean("x") == 1.0
    assert Param("m", "M", "choice", "a", choices=("a", "b")).clean("z") == "a"
    steps = [Step("gaussian", {"sigma": 2.0}), Step("canny", enabled=False),
             Step("threshold", {"method": "otsu"})]
    result, timings = run_pipeline(sample("Shapes", 64), steps)
    assert result.shape == (64, 64) and len(timings) == 3
    assert [Step.from_dict(s.to_dict()) for s in steps] == steps
    with pytest.raises(ValueError):
        Step.from_dict({"op": "nope"})


def test_png_files_round_trip_and_read_files_from_other_writers(tmp_path, qapp) -> None:
    from PySide6.QtGui import QImage

    for name in SAMPLES:
        image = sample(name, 48)
        assert np.array_equal(read_png(write_png(tmp_path / f"{name}.png", image)),
                              (np.clip(image, 0, 1) * 255 + 0.5).astype(np.uint8))
    qt = QImage(str(tmp_path / "Shapes.png"))
    assert qt.save(str(tmp_path / "qt.png"))  # Qt's writer uses PNG's row filters
    assert np.array_equal(read_png(tmp_path / "qt.png")[..., :3], read_png(tmp_path / "Shapes.png"))


def test_synthetic_scenes_are_reproducible_and_valid_coco(tmp_path) -> None:
    from labs.computer_vision.coco import CocoFile

    spec = SceneSpec(width=96, height=64, min_objects=2, max_objects=5, seed=11)
    first, objects = render_scene(spec, np.random.default_rng(11))
    again, _ = render_scene(spec, np.random.default_rng(11))
    assert np.array_equal(first, again) and 2 <= len(objects) <= 5
    for item in objects:
        x, y, w, h = item["bbox"]
        assert 0 <= x and 0 <= y and x + w <= 96 and y + h <= 64 and item["area"] > 0
    annotations = generate_dataset(spec, 6, tmp_path / "shapes", name="test shapes")
    coco = json.loads(annotations.read_text())
    assert len(coco["images"]) == 6
    assert {c["name"] for c in coco["categories"]} == {"circle", "square", "triangle"}
    assert all((tmp_path / "shapes" / image["file_name"]).is_file() for image in coco["images"])
    loaded = CocoFile.load(annotations)
    assert len(loaded.categories) == 3
    with pytest.raises(ValueError):
        SceneSpec(min_size=1).check()
    with pytest.raises(ValueError, match="already holds files"):
        generate_dataset(spec, 2, tmp_path / "shapes")


def test_annotations_describe_what_is_visible() -> None:
    from labs.computer_vision.coco import Annotation, CocoFile

    pytest.importorskip("pycocotools")
    crowded = SceneSpec(width=64, height=64, min_objects=6, max_objects=6, min_size=24, max_size=40, seed=3)
    rng = np.random.default_rng(3)
    covered = []
    for _ in range(10):
        _image, objects = render_scene(crowded, rng)
        covered += [o for o in objects if isinstance(o["segmentation"], dict)]
    assert covered, "six large shapes on 64 × 64 pixels always cover one another"
    for item in covered:
        annotation = Annotation(1, item["category"], tuple(item["bbox"]), item["area"], False,
                                rle=item["segmentation"])
        mask = CocoFile.decode_mask(annotation, 64, 64)
        ys, xs = np.nonzero(mask)
        assert mask.sum() == item["area"]
        assert item["bbox"] == [xs.min(), ys.min(), xs.max() - xs.min() + 1, ys.max() - ys.min() + 1]
    apart = SceneSpec(width=128, height=128, min_objects=4, max_objects=4, min_size=10, max_size=20,
                      overlap=False, seed=3)
    rng = np.random.default_rng(3)
    for _ in range(10):
        _image, objects = render_scene(apart, rng)
        assert all(isinstance(o["segmentation"], list) for o in objects)
    assert rle_counts(np.array([[1, 0], [1, 1]], dtype=bool)) == {"counts": [0, 2, 1, 1], "size": [2, 2]}


def test_a_cancelled_dataset_leaves_nothing_behind(tmp_path) -> None:
    token = CancelToken()
    seen = []

    def progress(fraction: float, _text: str) -> None:
        seen.append(fraction)
        if len(seen) == 3:
            token.cancel()

    with pytest.raises(Cancelled):
        generate_dataset(SceneSpec(width=32, height=32, max_size=16), 50, tmp_path / "set", progress=progress,
                         cancel=token)
    assert list(tmp_path.iterdir()) == [] and len(seen) == 3


def test_the_camera_projects_like_its_matrix() -> None:
    camera = Camera()
    points = np.array([[1.0, -1.0, 1.0], [0.0, 0.0, 0.0], [0.5, 0.2, -0.7]])
    uv, visible = project(camera, points)
    homogeneous = (projection_matrix(camera) @ np.c_[points, np.ones(3)].T).T
    assert np.allclose(uv, homogeneous[:, :2] / homogeneous[:, 2:]) and visible.all()
    behind, seen = project(Camera(tz=-5.0, yaw=0, pitch=0), np.zeros((1, 3)))
    assert not seen[0]
    straight = Camera(yaw=0, pitch=0, tz=5)
    centre, _ = project(straight, np.zeros((1, 3)))
    assert np.allclose(centre, [[straight.cx, straight.cy]])
    corner = np.array([[1.0, 1.0, 0.0]])
    plain, _ = project(straight, corner)
    barrel, _ = project(Camera(yaw=0, pitch=0, tz=5, k1=0.5), corner)
    centre = np.array([straight.cx, straight.cy])
    assert np.linalg.norm(barrel - centre) > np.linalg.norm(plain - centre)


def test_saved_simulations(tmp_path) -> None:
    folder = records.save(tmp_path, "circuit", "My Bell pair!", {"circuit": preset("Bell state").to_dict()},
                          {"circuit.qasm": "OPENQASM 2.0;", "../escape.txt": "no"})
    assert (folder / "circuit.qasm").is_file() and (folder / "escape.txt").is_file()
    assert not (tmp_path / "simulations" / "escape.txt").exists()
    saved = records.list_saved(tmp_path)
    assert [s.name for s in saved] == ["My Bell pair!"] and saved[0].title == "Quantum circuit"
    assert records.load(saved[0].folder)["settings"]["circuit"]["qubits"] == 2
    with pytest.raises(ValueError):
        records.save(tmp_path, "unknown", "x", {})
    records.delete(saved[0], tmp_path)
    assert records.list_saved(tmp_path) == []


SCRIPT_SETTINGS = {
    "circuit": {"circuit": preset("Teleportation").to_dict(), "shots": 64, "seed": 2},
    "waves": {"setup": WaveSetup(points=256).to_dict(), "duration": 2.0},
    "image": {"source": {"sample": "Shapes"},
              "pipeline": [Step("gaussian", {"sigma": 2.0}).to_dict(),
                           Step("canny", enabled=False).to_dict()]},
    "scenes": {"spec": SceneSpec(width=48, height=48, max_size=20).to_dict(), "count": 2},
    "camera": {"camera": Camera(k1=0.1).to_dict()},
}


@pytest.mark.parametrize("kind", sorted(SCRIPT_SETTINGS))
def test_exported_scripts_run_on_their_own(kind: str, tmp_path) -> None:
    script = tmp_path / f"{kind}.py"
    script.write_text(script_for(kind, SCRIPT_SETTINGS[kind]), encoding="utf-8")
    env = {**os.environ, "PYTHONPATH": str(CODE_ROOT)}
    result = subprocess.run([sys.executable, str(script)], cwd=tmp_path, capture_output=True, text=True,
                            timeout=180, env=env, stdin=subprocess.DEVNULL)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip()
