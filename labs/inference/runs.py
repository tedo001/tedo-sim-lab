"""Predictions with a finished run, for the Model Testing page: an image (or a WAV file, for runs
trained on spectrograms), a row of a table, a text, or a point for a quantum classifier. Each run
is loaded once from its folder, rebuilt exactly as its runner built it, and kept.

``predict`` answers one input or a batch; ``sweep`` varies one input over a range and returns
the probability of every class along it (a what-if simulation of the model)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from core.experiment_engine.spec import ExperimentSpec, load_spec

__all__ = ["RunModel", "kind_of", "predict", "sweep"]

_RUNS: dict[str, RunModel] = {}


def kind_of(spec: ExperimentSpec) -> str:
    """What a run takes as input: ``image`` | ``audio`` | ``table`` | ``text`` | ``quantum``."""
    if spec.training.kind == "quantum":
        return "quantum"
    if spec.task.value == "text_classification":
        return "text"
    if spec.training.kind == "sklearn":
        return "table"
    if spec.data.dataset in _audio_sets():
        return "audio"
    return "image"


def _audio_sets() -> set[str]:
    return {"spoken_digits"}


class RunModel:
    def __init__(self, run_dir: Path) -> None:
        self.run_dir = Path(run_dir)
        self.spec = load_spec(self.run_dir / "experiment.yaml")
        self.kind = kind_of(self.spec)
        self.info = json.loads((self.run_dir / "run_info.json").read_text(encoding="utf-8")) \
            if (self.run_dir / "run_info.json").is_file() else {}
        self.classes: list[str] = []
        getattr(self, f"_load_{'image' if self.kind == 'audio' else self.kind}")()

    # Loading -------------------------------------------------------------------------------
    def _checkpoint(self, *names: str) -> Path:
        for name in names:
            path = self.run_dir / "checkpoints" / name
            if path.exists():
                return path
        raise FileNotFoundError(f"{self.run_dir.name} has no {names[0]} (did the run finish?)")

    def _load_image(self) -> None:
        import torch

        from labs.computer_vision.classification import TorchClassificationRunner
        from labs.computer_vision.training.data import _transforms, input_size_for

        runner = TorchClassificationRunner()
        paths, adapter, _card, builder = runner._parts(self.spec)
        size = input_size_for(self.spec, adapter, builder)
        self.classes = list(adapter.classes)
        self.adapter, self.channels = adapter, adapter.in_channels
        self.transform = _transforms(self.spec, adapter, size, train=False)
        model = builder(num_classes=len(self.classes), in_channels=adapter.in_channels, input_size=size,
                        pretrained=None, **self.spec.model.params)
        state = torch.load(self._checkpoint("best.pt", "last.pt"), map_location="cpu", weights_only=True)
        model.load_state_dict(state["model"])
        from .loader import device

        self.device = device()
        self.model = model.to(self.device).eval()

    def _load_table(self) -> None:
        import joblib

        self.pipeline = joblib.load(self._checkpoint("model.joblib"))
        self.features = list(self.info.get("features", []))
        self.classes = [str(c) for c in self.info.get("classes") or []]

    def _load_text(self) -> None:
        folder = self.run_dir / "checkpoints" / "best"
        if folder.is_dir():  # a fine-tuned transformer
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            from .loader import device

            self.tokenizer = AutoTokenizer.from_pretrained(folder)
            self.model = AutoModelForSequenceClassification.from_pretrained(folder).to(device()).eval()
            self.pipeline = None
        else:
            import joblib

            self.pipeline = joblib.load(self._checkpoint("model.joblib"))
        self.classes = [str(c) for c in self.info.get("classes") or []]

    def _load_quantum(self) -> None:
        from labs.quantum_ml.runner import QuantumRunner

        runner = QuantumRunner()
        paths, adapter = runner._parts(self.spec)
        self.params = runner._params(self.spec)
        self.data = runner._data(self.spec, adapter, paths.datasets, int(self.params["qubits"]))
        self.classes = self.data.classes
        self.features = self.data.columns
        saved = np.load(self._checkpoint("best.npz"))
        if self.spec.model.model == "vqc":
            self.model = runner._model(self.params, len(self.classes), self.spec.seed)
            self.model.params = saved["params"]
        else:
            from sklearn.svm import SVC

            from labs.quantum_ml.kernel import fidelity_kernel

            self.x_fit = saved["x_fit"]
            self.svm = SVC(kernel="precomputed", C=float(self.params["C"]), probability=True,
                           random_state=self.spec.seed)
            self.svm.fit(fidelity_kernel(self.x_fit, encoding=self.params["encoding"]), saved["y_fit"])

    # Predicting ----------------------------------------------------------------------------
    def probabilities(self, inputs: list[Any]) -> np.ndarray:
        return getattr(self, f"_p_{'image' if self.kind == 'audio' else self.kind}")(inputs)

    def _p_image(self, inputs: list[Any]) -> np.ndarray:
        import torch
        from PIL import Image, ImageOps

        tensors = []
        for item in inputs:
            path, invert = (item["path"], bool(item.get("invert"))) if isinstance(item, dict) else (item,
                                                                                                    False)
            if self.kind == "audio" or str(path).lower().endswith(".wav"):
                from labs.audio.dsp import load_wav, log_mel_image

                image = Image.fromarray(log_mel_image(*load_wav(path)))
            else:
                with Image.open(path) as opened:
                    image = opened.convert("L" if self.channels == 1 else "RGB")
                if invert:
                    image = ImageOps.invert(image)
            tensors.append(self.transform(image.convert("L" if self.channels == 1 else "RGB")))
        with torch.inference_mode():
            logits = self.model(torch.stack(tensors).to(self.device))
        return logits.softmax(-1).float().cpu().numpy()

    def _p_table(self, inputs: list[Any]) -> np.ndarray:
        import pandas as pd

        frame = pd.DataFrame([{name: row.get(name) for name in self.features} for row in inputs])
        if hasattr(self.pipeline, "predict_proba"):
            return np.asarray(self.pipeline.predict_proba(frame))
        predicted = np.asarray(self.pipeline.predict(frame), dtype=float)
        return predicted[:, None]  # regression: the value itself

    def _p_text(self, inputs: list[Any]) -> np.ndarray:
        texts = [str(t) for t in inputs]
        if self.pipeline is not None:
            if hasattr(self.pipeline, "predict_proba"):
                return np.asarray(self.pipeline.predict_proba(texts))
            scores = np.asarray(self.pipeline.decision_function(texts), dtype=float)
            scores = np.stack([-scores, scores], 1) if scores.ndim == 1 else scores
            exp = np.exp(scores - scores.max(1, keepdims=True))
            return exp / exp.sum(1, keepdims=True)
        import torch

        batch = self.tokenizer(texts, padding=True, truncation=True, max_length=256, return_tensors="pt")
        with torch.inference_mode():
            logits = self.model(**batch.to(self.model.device)).logits
        return logits.softmax(-1).float().cpu().numpy()

    def _p_quantum(self, inputs: list[Any]) -> np.ndarray:
        raw = np.array([[float(row[name]) for name in self.features] for row in inputs])
        angles = self.data.transform(raw)
        if self.spec.model.model == "vqc":
            return self.model.predict_proba(angles)
        from labs.quantum_ml.kernel import fidelity_kernel

        return self.svm.predict_proba(fidelity_kernel(angles, self.x_fit, encoding=self.params["encoding"]))

    def describe(self) -> dict[str, Any]:
        out: dict[str, Any] = {"kind": self.kind, "classes": self.classes, "name": self.spec.name,
                               "task": self.spec.task.value, "model": self.spec.model.model}
        if self.kind in ("table", "quantum"):
            out["features"] = list(self.features)
        if self.kind == "image":
            out["channels"] = self.channels
        return out


def _model(run_dir: str) -> RunModel:
    key = str(Path(run_dir).resolve())
    if key not in _RUNS:
        _RUNS[key] = RunModel(Path(run_dir))
    return _RUNS[key]


def predict(args: dict[str, Any], emit: Any, cancelled: Any, workspace: Path) -> dict[str, Any]:
    """``run`` (its folder) and ``inputs`` (a list); per input: the classes ranked by probability."""
    run = _model(args["run"])
    inputs = list(args.get("inputs") or [])
    rows = []
    for start in range(0, len(inputs), 32):
        if cancelled():
            raise InterruptedError("stopped")
        probs = run.probabilities(inputs[start:start + 32])
        for p in probs:
            if p.shape[0] == 1 and len(run.classes) != 1:  # a regression value
                rows.append({"value": float(p[0])})
                continue
            order = np.argsort(-p)
            rows.append({"classes": [run.classes[i] if i < len(run.classes) else str(i) for i in order],
                         "probabilities": [float(p[i]) for i in order]})
        emit({"done": min(start + 32, len(inputs)), "total": len(inputs)})
    return {"model": run.describe(), "predictions": rows}


def sweep(args: dict[str, Any], emit: Any, cancelled: Any, workspace: Path) -> dict[str, Any]:
    """Vary ``feature`` of ``base`` (a table row or quantum point) from ``low`` to ``high``."""
    run = _model(args["run"])
    if run.kind not in ("table", "quantum"):
        raise ValueError("what-if sweeps are for table and quantum models")
    values = np.linspace(float(args["low"]), float(args["high"]), int(args.get("steps", 41)))
    rows = [{**args["base"], args["feature"]: float(v)} for v in values]
    probs = run.probabilities(rows)
    return {"values": values.tolist(), "classes": run.classes or ["value"], "probabilities": probs.T.tolist(),
            "model": run.describe()}
