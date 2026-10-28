"""Image models in the inference worker: zero-shot classification and text-to-image search with a
CLIP model, captions with BLIP, and object detection with any ``transformers`` detector (YOLOS,
DETR-style, RT-DETR) — boxes come back in the image's own pixels."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .loader import device, dtype_for, load, resolve_model

__all__ = ["caption", "detect", "image_search", "open_image", "zero_shot"]

IMAGE_TYPES = (".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp", ".tif", ".tiff")
_SEARCH_CACHE: dict[tuple[str, str, float], Any] = {}


def open_image(path: str | Path) -> Any:
    from PIL import Image

    with Image.open(path) as image:
        return image.convert("RGB")


def _clip(folder: Path) -> tuple[Any, Any, Any]:
    from transformers import AutoTokenizer, CLIPImageProcessor, CLIPModel

    return (CLIPModel.from_pretrained(folder).to(device()).eval(), CLIPImageProcessor.from_pretrained(folder),
            AutoTokenizer.from_pretrained(folder))


def _clip_text(model: Any, tokenizer: Any, texts: list[str]) -> Any:
    import torch

    batch = tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        features = model.get_text_features(**batch)
    features = getattr(features, "pooler_output", features)
    return torch.nn.functional.normalize(features.float(), dim=-1)


def _clip_images(model: Any, processor: Any, images: list[Any]) -> Any:
    import torch

    pixels = processor(images=images, return_tensors="pt")["pixel_values"].to(model.device)
    with torch.inference_mode():
        features = model.get_image_features(pixel_values=pixels)
    features = getattr(features, "pooler_output", features)
    return torch.nn.functional.normalize(features.float(), dim=-1)


def zero_shot(args: dict[str, Any], emit: Any, cancelled: Any, workspace: Path) -> dict[str, Any]:
    """``image`` against ``labels``, each put into ``template`` ("a photo of a {}")."""
    model, processor, tokenizer = load("clip", resolve_model(args["model"], workspace), _clip)
    labels = [str(label) for label in args["labels"] if str(label).strip()]
    if not labels:
        raise ValueError("give at least one label")
    template = str(args.get("template") or "a photo of a {}")
    text = _clip_text(model, tokenizer, [template.replace("{}", label) for label in labels])
    image = _clip_images(model, processor, [open_image(args["image"])])
    scale = float(model.logit_scale.exp().item())
    logits = (scale * image @ text.T)[0]
    probs = logits.softmax(-1).tolist()
    order = sorted(range(len(labels)), key=lambda i: -probs[i])
    return {"labels": [labels[i] for i in order], "probabilities": [probs[i] for i in order],
            "similarity": [float((image @ text.T)[0, i]) for i in order]}


def image_search(args: dict[str, Any], emit: Any, cancelled: Any, workspace: Path) -> dict[str, Any]:
    """Rank the images in ``folder`` by similarity to ``query``; image vectors are kept between calls."""
    import torch

    folder_arg = resolve_model(args["model"], workspace)
    model, processor, tokenizer = load("clip", folder_arg, _clip)
    folder = Path(args["folder"])
    files = sorted(p for p in folder.rglob("*") if p.suffix.lower() in IMAGE_TYPES)[: int(args.get("limit",
                                                                                                   500))]
    if not files:
        raise FileNotFoundError(f"no images in {folder}")
    vectors = []
    for start in range(0, len(files), 16):
        if cancelled():
            raise InterruptedError("stopped")
        chunk = files[start:start + 16]
        missing = [p for p in chunk if (str(folder_arg), str(p), p.stat().st_mtime) not in _SEARCH_CACHE]
        if missing:
            found = _clip_images(model, processor, [open_image(p) for p in missing]).cpu()
            for path, vector in zip(missing, found, strict=True):
                _SEARCH_CACHE[(str(folder_arg), str(path), path.stat().st_mtime)] = vector
        vectors += [_SEARCH_CACHE[(str(folder_arg), str(p), p.stat().st_mtime)] for p in chunk]
        emit({"done": min(start + 16, len(files)), "total": len(files)})
    matrix = torch.stack(vectors)
    query = _clip_text(model, tokenizer, [str(args["query"])]).cpu()[0]
    scores = (matrix @ query).tolist()
    order = sorted(range(len(files)), key=lambda i: -scores[i])
    return {"files": [str(files[i]) for i in order], "scores": [scores[i] for i in order],
            "vectors": matrix.tolist(), "query_vector": query.tolist(), "all_files": [str(p) for p in files]}


def _blip(folder: Path) -> tuple[Any, Any, Any]:
    from transformers import AutoTokenizer, BlipForConditionalGeneration, BlipImageProcessor

    model = BlipForConditionalGeneration.from_pretrained(folder, dtype=dtype_for(True)).to(device()).eval()
    return model, BlipImageProcessor.from_pretrained(folder), AutoTokenizer.from_pretrained(folder)


def caption(args: dict[str, Any], emit: Any, cancelled: Any, workspace: Path) -> dict[str, Any]:
    """A caption for ``image``; ``prompt`` starts it ("a photograph of")."""
    import torch

    model, processor, tokenizer = load("blip", resolve_model(args["model"], workspace), _blip)
    pixels = processor(images=[open_image(args["image"])], return_tensors="pt")["pixel_values"]
    inputs: dict[str, Any] = {"pixel_values": pixels.to(model.device, model.dtype)}
    prompt = str(args.get("prompt") or "").strip()
    if prompt:
        inputs.update(tokenizer(prompt, return_tensors="pt").to(model.device))
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=int(args.get("max_new_tokens", 30)),
                             num_beams=int(args.get("beams", 3)))
    return {"caption": tokenizer.decode(out[0], skip_special_tokens=True).strip()}


def _detector(folder: Path) -> tuple[Any, Any]:
    from transformers import AutoImageProcessor, AutoModelForObjectDetection

    return (AutoModelForObjectDetection.from_pretrained(folder).to(device()).eval(),
            AutoImageProcessor.from_pretrained(folder))


def detect(args: dict[str, Any], emit: Any, cancelled: Any, workspace: Path) -> dict[str, Any]:
    """Boxes above ``threshold`` in ``image``: ``[x0, y0, x1, y1]`` pixels, label, score."""
    import torch

    model, processor = load("detector", resolve_model(args["model"], workspace), _detector)
    image = open_image(args["image"])
    inputs = processor(images=image, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        outputs = model(**inputs)
    found = processor.post_process_object_detection(outputs, threshold=float(args.get("threshold", 0.5)),
                                                    target_sizes=[(image.height, image.width)])[0]
    names = model.config.id2label
    boxes = [{"box": [round(float(v), 1) for v in box], "label": names.get(int(label), str(int(label))),
              "score": float(score)}
             for box, label, score in zip(found["boxes"].tolist(), found["labels"].tolist(),
                                          found["scores"].tolist(), strict=True)]
    boxes.sort(key=lambda b: -b["score"])
    return {"boxes": boxes, "width": image.width, "height": image.height}
