"""Inside a trained image network, for one input: what every layer produced (feature maps of
convolutions, pooling and residual blocks; the activations of dense layers), where the network
looked (Grad-CAM on the last convolution, Selvaraju et al. 2017) and which pixels mattered most
(gradient saliency). Works for any PyTorch image classifier the lab trains — SimpleCNN, LeNet-5,
TinyVGG, ResNet-18, layer stacks — and for spoken-digit runs, whose inputs are spectrograms.

Maps are sent as raw bytes (uint8, base64) with their shape and value range, so the page can
colour them like the CNN Explainer does."""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Any

import numpy as np

from .runs import _model

__all__ = ["visualize"]

MAX_CHANNELS = 16
MAX_SIDE = 64
MAX_LAYERS = 24
_SPATIAL = ("Conv2d", "MaxPool2d", "AvgPool2d", "AdaptiveAvgPool2d", "BasicBlock", "Bottleneck", "ReLU")
_DENSE = ("Linear",)


def _bytes(array: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(array, dtype=np.uint8).tobytes()).decode("ascii")


def _scaled(array: np.ndarray) -> tuple[np.ndarray, float, float]:
    low, high = float(array.min()), float(array.max())
    span = high - low or 1.0
    return ((array - low) / span * 255).round().astype(np.uint8), low, high


def _shrink(maps: np.ndarray) -> np.ndarray:
    """channels × h × w, each side at most MAX_SIDE (block average)."""
    _c, h, w = maps.shape
    step = max(1, int(np.ceil(max(h, w) / MAX_SIDE)))
    if step == 1:
        return maps
    h2, w2 = h // step * step, w // step * step
    return maps[:, :h2, :w2].reshape(maps.shape[0], h2 // step, step, w2 // step, step).mean(axis=(2, 4))


def _layers(model: Any) -> list[tuple[str, Any]]:
    """The layers worth showing, in order: convolutions, activations, pooling and dense layers;
    for residual networks each block's output instead of the layers inside it."""
    named = [(name, module) for name, module in model.named_modules() if name]
    blocks = [name for name, module in named if type(module).__name__ in ("BasicBlock", "Bottleneck")]
    chosen = [(name, module) for name, module in named
              if (type(module).__name__ in _SPATIAL or type(module).__name__ in _DENSE)
              and not any(name.startswith(block + ".") for block in blocks)]
    return chosen[:MAX_LAYERS]


def visualize(args: dict[str, Any], emit: Any, cancelled: Any, workspace: Path) -> dict[str, Any]:
    """``run`` and ``input`` (``{"path", "invert"}``); ``target`` = class index to explain (default:
    the predicted one)."""

    run = _model(args["run"])
    if run.kind not in ("image", "audio"):
        raise ValueError("layer views are for image and spectrogram networks; table, text and quantum models "
                         "are explained on their own pages")
    item = args["input"]
    seen: list[tuple[str, str, Any]] = []
    hooks = [module.register_forward_hook(lambda _m, _i, out, name=name, kind=type(module).__name__:
                                          seen.append((name, kind, out)))
             for name, module in _layers(run.model)]
    convs = [m for m in run.model.modules() if type(m).__name__ == "Conv2d"]
    last_conv: dict[str, Any] = {}
    if convs:
        hooks.append(convs[-1].register_forward_hook(lambda _m, _i, out: last_conv.update(out=out)))
    try:
        tensor = _input_tensor(run, item).to(run.device).requires_grad_(True)
        logits = run.model(tensor)
        probs = logits.softmax(-1)[0].detach().float().cpu().numpy()
        target = int(args.get("target", int(probs.argmax())))
        if "out" in last_conv:
            last_conv["out"].retain_grad()
        logits[0, target].backward()
    finally:
        for hook in hooks:
            hook.remove()
    shown = _show_input(tensor[0].detach().float().cpu().numpy())
    layers = []
    for name, kind, out in seen:
        value = out.detach().float().cpu().numpy()[0]
        entry: dict[str, Any] = {"name": name, "kind": kind, "shape": list(value.shape)}
        if value.ndim == 3:  # the strongest channels (largest mean activation), in that order
            order = np.argsort(-value.reshape(value.shape[0], -1).mean(axis=1))[:MAX_CHANNELS]
            scaled, low, high = _scaled(_shrink(value[order]))
            entry.update(maps=_bytes(scaled), map_shape=list(scaled.shape), low=low, high=high,
                         channels=int(value.shape[0]), strongest=[int(i) for i in order])
        else:
            flat = value.reshape(-1)
            entry.update(values=[float(v) for v in flat[:64]], size=int(flat.size))
        layers.append(entry)
    result: dict[str, Any] = {"model": run.describe(), "input": shown, "layers": layers,
                              "classes": run.classes, "probabilities": [float(p)
                                                                        for p in probs], "target": target,
                              "parameters": int(sum(p.numel() for p in run.model.parameters()))}
    grad = tensor.grad[0].detach().float().cpu().numpy()
    saliency, _, _ = _scaled(np.abs(grad).max(axis=0))
    result["saliency"] = {"data": _bytes(saliency), "shape": list(saliency.shape)}
    if "out" in last_conv and last_conv["out"].grad is not None:
        act = last_conv["out"].detach().float().cpu().numpy()[0]
        weights = last_conv["out"].grad.detach().float().cpu().numpy()[0].mean(axis=(1, 2))
        cam = np.maximum((weights[:, None, None] * act).sum(axis=0), 0)
        cam = _resize(cam, saliency.shape)
        heat, _, _ = _scaled(cam)
        result["gradcam"] = {"data": _bytes(heat), "shape": list(heat.shape), "layer": "last convolution"}
    return result


def _input_tensor(run: Any, item: Any) -> Any:
    from PIL import Image, ImageOps

    path, invert = (item["path"], bool(item.get("invert"))) if isinstance(item, dict) else (item, False)
    if run.kind == "audio" or str(path).lower().endswith(".wav"):
        from labs.audio.dsp import load_wav, log_mel_image

        image = Image.fromarray(log_mel_image(*load_wav(path)))
    else:
        with Image.open(path) as opened:
            image = opened.convert("L" if run.channels == 1 else "RGB")
        if invert:
            image = ImageOps.invert(image)
    return run.transform(image.convert("L" if run.channels == 1 else "RGB")).unsqueeze(0)


def _show_input(array: np.ndarray) -> dict[str, Any]:
    """The input as the network sees it (after resizing and normalising), rescaled to 0..255."""
    picture = array[0] if array.shape[0] == 1 else np.transpose(array, (1, 2, 0))
    scaled, _, _ = _scaled(picture)
    return {"data": _bytes(scaled), "shape": list(scaled.shape)}


def _resize(array: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    from PIL import Image

    image = Image.fromarray(array.astype(np.float32), mode="F")
    return np.asarray(image.resize((shape[1], shape[0]), Image.Resampling.BILINEAR))
