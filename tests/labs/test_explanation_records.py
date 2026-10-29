"""Explanation records: an explanation can be traced to the precise input and model version that
generated it (the acceptance test), and a changed checkpoint or input is noticed."""

from __future__ import annotations

import json

import pytest

pytest.importorskip("torch")
pytest.importorskip("torchvision")

from labs.inference.explanations import list_records, verify  # noqa: E402
from labs.inference.visualize import visualize  # noqa: E402
from labs.simulation.vision.png import write_png  # noqa: E402
from tests.fixtures.fake_vision import make_images  # noqa: E402
from tests.labs.test_vision_training import make_spec, train  # noqa: E402
from tests.labs.test_vision_training import workspace as workspace  # noqa: E402, F401 (fixture)


def test_every_explanation_is_recorded_and_traceable(workspace, tmp_path):
    run_dir, _recorder, result = train(workspace, make_spec(epochs=1, max_steps_per_epoch=3), "explained")
    assert result.status == "completed"
    images, _labels = make_images(2, seed=9)
    first = write_png(tmp_path / "digit.png", images[0])
    answer = visualize({"run": str(run_dir), "input": {"path": str(first)}, "target": 3}, lambda _: None,
                       lambda: False, workspace.workspace)
    folder = answer["record"]["folder"]
    record = json.loads((run_dir / "explanations" / answer["record"]["id"] / "record.json").read_text())
    # The precise input …
    assert record["input"]["original_path"] == str(first)
    import hashlib

    assert record["input"]["sha256"] == hashlib.sha256(first.read_bytes()).hexdigest()
    # … the model version …
    checkpoint = run_dir / record["model"]["checkpoint"]["file"]
    assert record["model"]["checkpoint"]["file"] == "checkpoints/best.pt"
    assert record["model"]["checkpoint"]["sha256"] == hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    assert record["model"]["run_id"] == run_dir.name and len(record["model"]["spec_hash"]) == 64
    # … the target class, the method and its settings, and the outputs.
    classes = record["model"]["classes"]
    assert record["target"] == {"index": 3, "how": "chosen by the person", "name": classes[3]}
    assert len(record["prediction"]["probabilities"]) == 10
    names = [method["name"] for method in record["methods"]]
    assert names == ["Grad-CAM", "gradient saliency", "layer activations"]
    assert record["methods"][0]["layer"] and "Normalize" in record["settings"]["transform"]
    assert {"gradcam", "saliency", "input_as_seen"} <= set(record["outputs"])
    assert verify(folder) == []
    # A second explanation, of the predicted class; both are listed, newest first.
    second = visualize({"run": str(run_dir), "input": {"path": str(first)}}, lambda _: None, lambda: False,
                       workspace.workspace)
    listed = list_records(run_dir)
    assert [r["id"] for r in listed][:2] == sorted([answer["record"]["id"], second["record"]["id"]],
                                                   reverse=True)
    assert listed[0]["target"]["how"] == "the predicted class"
    # Retraining changes the checkpoint: the old explanation no longer matches the model, and says so.
    checkpoint.write_bytes(checkpoint.read_bytes() + b"\0")
    assert any("checkpoint has changed" in problem for problem in verify(folder))
    (run_dir / "explanations" / answer["record"]["id"] / "input.png").write_bytes(b"not the input")
    assert any("saved input differs" in problem for problem in verify(folder))


def test_recording_can_be_switched_off(workspace, tmp_path):
    run_dir, _recorder, _result = train(workspace, make_spec(epochs=1, max_steps_per_epoch=2), "quiet")
    picture = write_png(tmp_path / "x.png", make_images(1, seed=1)[0][0])
    answer = visualize({"run": str(run_dir), "input": {"path": str(picture)}, "record": False},
                       lambda _: None, lambda: False, workspace.workspace)
    assert "record" not in answer and list_records(run_dir) == []
