"""Explanation records: every layer view, Grad-CAM and saliency map the lab makes is saved with what
produced it, so it can be traced to the precise input and model version later.

``<run>/explanations/<id>/`` holds ``record.json`` and the files it names:

- the **input** as given (a copy, with its SHA-256 and original path) and as the network saw it
  (``input_as_seen.png``, after resizing and normalising);
- the **model version**: the run, its spec hash, the checkpoint file with its SHA-256, size and time;
- the **target class** (index and name, and whether it was chosen or the predicted one) with the
  predicted class and every probability;
- the **method and settings**: Grad-CAM on the named last convolution (Selvaraju et al. 2017), gradient
  saliency (max |∂score/∂x| over channels), the layers recorded, how many channels, the transform;
- the **outputs** (``gradcam.png``, ``saliency.png``) with their SHA-256, and the software versions.

:func:`verify` re-hashes the input copy, the outputs and the checkpoint: a record whose checkpoint has
changed since says so. Nothing here imports PyTorch, so the app can list and verify records."""

from __future__ import annotations

import base64
import hashlib
import json
import platform
import shutil
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np

__all__ = ["FOLDER", "RECORD", "list_records", "verify", "write_record"]

FOLDER = "explanations"
RECORD = "record.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for block in iter(lambda: file.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _png(folder: Path, name: str, item: dict[str, Any]) -> dict[str, Any]:
    from labs.simulation.vision.png import write_png

    array = np.frombuffer(base64.b64decode(item["data"]), dtype=np.uint8).reshape(item["shape"])
    path = write_png(folder / name, array)
    return {"file": name, "sha256": _sha256(path), "shape": list(item["shape"])}


def write_record(run: Any, args: dict[str, Any], result: dict[str, Any], *, chosen: bool,
                 layer: str | None) -> dict[str, Any]:
    """Save one explanation of ``run`` (a loaded :class:`RunModel`) for ``args["input"]``."""
    from core.experiment_engine.spec import spec_hash

    created = datetime.now(UTC)
    stamp = hashlib.sha1(str(created.timestamp()).encode()).hexdigest()[:6]
    record_id = created.strftime("%Y%m%d-%H%M%S-") + stamp
    folder = Path(run.run_dir) / FOLDER / record_id
    folder.mkdir(parents=True, exist_ok=True)
    item = args["input"] if isinstance(args["input"], dict) else {"path": args["input"]}
    source = Path(item["path"])
    copy = folder / f"input{source.suffix.lower() or '.bin'}"
    shutil.copy2(source, copy)
    checkpoint = Path(run.checkpoint)
    probabilities = result["probabilities"]
    predicted = int(np.argmax(probabilities))
    target = int(result["target"])
    outputs = {"input_as_seen": _png(folder, "input_as_seen.png", result["input"]),
               "saliency": _png(folder, "saliency.png", result["saliency"])}
    if "gradcam" in result:
        outputs["gradcam"] = _png(folder, "gradcam.png", result["gradcam"])
    record = {
        "id": record_id, "created_at": created.isoformat(timespec="seconds"),
        "input": {"file": copy.name, "original_path": str(source), "sha256": _sha256(copy),
                  "bytes": copy.stat().st_size, "inverted": bool(item.get("invert"))},
        "model": {"run_id": Path(run.run_dir).name, "experiment": run.spec.name,
                  "model": run.spec.model.model, "dataset": run.spec.data.dataset,
                  "spec_hash": spec_hash(run.spec),
                  "checkpoint": {"file": checkpoint.relative_to(run.run_dir).as_posix(),
                                 "sha256": _sha256(checkpoint), "bytes": checkpoint.stat().st_size,
                                 "modified": datetime.fromtimestamp(checkpoint.stat().st_mtime, UTC)
                                 .isoformat(timespec="seconds")},
                  "classes": list(run.classes)},
        "target": {"index": target, "how": "chosen by the person" if chosen else "the predicted class",
                   "name": run.classes[target] if target < len(run.classes) else str(target)},
        "prediction": {"index": predicted, "name": run.classes[predicted] if predicted < len(run.classes)
                       else str(predicted), "probabilities": [float(p) for p in probabilities]},
        "methods": [
            {"name": "Grad-CAM", "reference": "Selvaraju et al., ICCV 2017", "layer": layer,
             "settings": "gradient of the target score, averaged over each map (channel weights); ReLU of "
                         "the weighted sum of the maps; resized to the input"}
            if "gradcam" in result else None,
            {"name": "gradient saliency", "settings": "max over channels of |∂ target score / ∂ input|"},
            {"name": "layer activations", "settings": f"{len(result.get('layers', []))} layers; the "
                                                      "strongest channels by mean activation"}],
        "settings": {"device": str(getattr(run, "device", "cpu")), "transform": repr(run.transform),
                     "model_mode": "eval", "batch": 1},
        "outputs": outputs,
        "software": {"python": platform.python_version(), "torch": _version("torch"),
                     "torchvision": _version("torchvision"), "numpy": _version("numpy")},
    }
    record["methods"] = [m for m in record["methods"] if m]
    (folder / RECORD).write_text(json.dumps(record, indent=2), encoding="utf-8")
    return {"id": record_id, "folder": str(folder)}


def list_records(run_dir: Path) -> list[dict[str, Any]]:
    """Every explanation record of a run, newest first."""
    folder = Path(run_dir) / FOLDER
    found = []
    for path in sorted(folder.glob(f"*/{RECORD}"), reverse=True) if folder.is_dir() else []:
        try:
            found.append({**json.loads(path.read_text(encoding="utf-8")), "folder": str(path.parent)})
        except (OSError, ValueError):
            continue
    return found


def verify(record_folder: Path, run_dir: Path | None = None) -> list[str]:
    """What no longer matches the record (``[]`` = the explanation still traces to its input and
    model version): the input copy, every output and the checkpoint are hashed again."""
    folder = Path(record_folder)
    record = json.loads((folder / RECORD).read_text(encoding="utf-8"))
    run_dir = Path(run_dir) if run_dir else folder.parent.parent
    problems = []
    if not (folder / record["input"]["file"]).is_file() or _sha256(folder / record["input"]["file"]) != \
            record["input"]["sha256"]:
        problems.append("the saved input differs from the one explained")
    for name, output in record["outputs"].items():
        path = folder / output["file"]
        if not path.is_file() or _sha256(path) != output["sha256"]:
            problems.append(f"{name} differs from the one recorded")
    checkpoint = run_dir / record["model"]["checkpoint"]["file"]
    if not checkpoint.is_file():
        problems.append(f"the checkpoint {record['model']['checkpoint']['file']} is gone")
    elif _sha256(checkpoint) != record["model"]["checkpoint"]["sha256"]:
        problems.append("the checkpoint has changed since (the model was retrained or resumed)")
    return problems
