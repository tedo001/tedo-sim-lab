"""Speech recognition in the inference worker: a WAV file is transcribed in 30-second windows
(Whisper's own input length), each returned as soon as it is done. Whisper models generate text
(and can translate to English); wav2vec 2.0 models read characters off a CTC head."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .loader import device, dtype_for, load, resolve_model

__all__ = ["transcribe"]

RATE = 16_000
WINDOW = 30 * RATE


def _whisper(folder: Path) -> tuple[Any, Any, Any]:
    from transformers import AutoTokenizer, WhisperFeatureExtractor, WhisperForConditionalGeneration

    model = WhisperForConditionalGeneration.from_pretrained(folder, dtype=dtype_for(True)).to(device()).eval()
    return model, WhisperFeatureExtractor.from_pretrained(folder), AutoTokenizer.from_pretrained(folder)


def _ctc(folder: Path) -> tuple[Any, Any, Any]:
    from transformers import AutoModelForCTC, AutoProcessor

    model = AutoModelForCTC.from_pretrained(folder).to(device()).eval()
    return model, AutoProcessor.from_pretrained(folder), None


def _kind(folder: Path) -> str:
    import json

    config = json.loads((folder / "config.json").read_text(encoding="utf-8"))
    return "whisper" if config.get("model_type") == "whisper" else "ctc"


def transcribe(args: dict[str, Any], emit: Any, cancelled: Any, workspace: Path) -> dict[str, Any]:
    """``audio`` (a WAV path); ``language`` ("" = detect) and ``task`` ("transcribe" | "translate")."""
    import torch

    from labs.audio.dsp import load_wav, resample

    folder = resolve_model(args["model"], workspace)
    kind = _kind(folder)
    model, features, tokenizer = load(kind, folder, _whisper if kind == "whisper" else _ctc)
    signal, rate = load_wav(args["audio"])
    signal = resample(signal, rate, RATE)
    limit = int(getattr(model.config, "max_target_positions", 448)) - 4  # room for the start tokens
    options: dict[str, Any] = {"max_new_tokens": max(1, min(int(args.get("max_new_tokens", 200)), limit))}
    if args.get("language"):
        options["language"] = str(args["language"])
    if args.get("task") in ("transcribe", "translate"):
        options["task"] = args["task"]
    segments = []
    for start in range(0, max(len(signal), 1), WINDOW):
        if cancelled():
            break
        piece = signal[start:start + WINDOW]
        inputs = features(piece, sampling_rate=RATE, return_tensors="pt")
        with torch.inference_mode():
            if kind == "whisper":
                ids = model.generate(inputs["input_features"].to(model.device, model.dtype), **options)
                text = tokenizer.decode(ids[0], skip_special_tokens=True).strip()
            else:
                logits = model(inputs["input_values"].to(model.device)).logits
                text = features.batch_decode(logits.argmax(-1))[0].strip().lower()
        segment = {"start": start / RATE, "end": min(start + WINDOW, len(signal)) / RATE, "text": text}
        segments.append(segment)
        emit(segment)
    return {"text": " ".join(s["text"] for s in segments).strip(), "segments": segments,
            "seconds_of_audio": len(signal) / RATE}
