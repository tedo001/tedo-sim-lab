"""The inference worker through the app's service: start on first request, stream partial
results, finish, fail clearly, cancel, and restart after it stops."""

from __future__ import annotations

import sys

import numpy as np
import pytest

from app.services.inference import InferenceService
from core.common import AppConfig, AppPaths
from tests.fixtures import tiny_models

pytest.importorskip("transformers")


@pytest.fixture
def service(tmp_path, qapp):
    paths = AppPaths.resolve(tmp_path / "ws").ensure()
    found = InferenceService(paths, AppConfig(python_executable=sys.executable))
    yield found
    found.stop()


def wait(qtbot, results, key, timeout=120_000):
    qtbot.waitUntil(lambda: key in results, timeout=timeout)
    return results[key]


def test_generation_streams_and_finishes(service, qtbot, tmp_path) -> None:
    model = tiny_models.causal_lm(tmp_path / "lm")
    results, parts = {}, []
    service.request("generate", {"model": str(model), "messages": [{"role": "user", "content": "hi"}],
                                 "max_new_tokens": 6, "temperature": 0},
                    on_partial=parts.append, on_done=lambda d: results.update(done=d),
                    on_error=lambda e: results.update(error=e))
    assert service.state == "starting"
    done = wait(qtbot, results, "done")
    assert service.state == "ready" and service.info["device"] in ("cpu", "cuda")
    assert done["tokens"] >= 1 and "".join(p["text"] for p in parts) == done["text"]


def test_errors_and_unknown_models_are_reported(service, qtbot, tmp_path) -> None:
    results = {}
    service.request("embed", {"model": str(tmp_path / "nowhere"), "texts": ["a"]},
                    on_error=lambda e: results.update(error=e))
    assert "is not downloaded" in wait(qtbot, results, "error")
    service.request("nonsense", {}, on_error=lambda e: results.update(unknown=e))
    assert "unknown task" in wait(qtbot, results, "unknown")


def test_detection_and_restart_after_stop(service, qtbot, tmp_path) -> None:
    from PIL import Image

    image = tmp_path / "x.png"
    Image.fromarray((np.random.default_rng(0).random((40, 50, 3)) * 255).astype("uint8")).save(image)
    detector = tiny_models.detector(tmp_path / "det")
    results = {}
    service.request("detect", {"model": str(detector), "image": str(image), "threshold": 0.0},
                    on_done=lambda d: results.update(first=d))
    first = wait(qtbot, results, "first")
    assert first["width"] == 50 and first["height"] == 40 and first["boxes"]
    assert {"box", "label", "score"} <= set(first["boxes"][0])
    service.stop()
    assert service.state == "stopped"
    service.request("detect", {"model": str(detector), "image": str(image)},
                    on_done=lambda d: results.update(again=d))
    assert wait(qtbot, results, "again")["width"] == 50
