"""Language models in the inference worker: chat with a small instruction-tuned model (tokens
stream back as they are generated), sentence embeddings (mean of the token vectors, normalised),
and a tokenizer view (each token, its id and where it came from in the text)."""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any

from .loader import device, dtype_for, load, resolve_model

__all__ = ["embed", "generate", "tokenize"]


def _causal(folder: Path) -> tuple[Any, Any]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(folder)
    model = AutoModelForCausalLM.from_pretrained(folder, dtype=dtype_for(True)).to(device()).eval()
    return model, tokenizer


def _prompt(tokenizer: Any, messages: list[dict[str, str]]) -> dict[str, Any]:
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt",
                                             return_dict=True)
    text = "".join(f"{m['role'].capitalize()}: {m['content']}\n" for m in messages) + "Assistant:"
    return tokenizer(text, return_tensors="pt")


def generate(args: dict[str, Any], emit: Any, cancelled: Any, workspace: Path) -> dict[str, Any]:
    """``messages`` (role/content), ``max_new_tokens``, ``temperature`` (0 = greedy), ``top_p``."""
    import torch
    from transformers import StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer

    folder = resolve_model(args["model"], workspace)
    model, tokenizer = load("causal", folder, _causal)
    inputs = {key: value.to(model.device) for key, value in _prompt(tokenizer, args["messages"]).items()}

    class Stop(StoppingCriteria):
        def __call__(self, input_ids: Any, scores: Any, **kwargs: Any) -> Any:
            return torch.full((input_ids.shape[0],), bool(cancelled()), dtype=torch.bool,
                              device=input_ids.device)

    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    temperature = float(args.get("temperature", 0.7))
    options: dict[str, Any] = {"max_new_tokens": int(args.get("max_new_tokens", 256)), "streamer": streamer,
                               "stopping_criteria": StoppingCriteriaList([Stop()]),
                               "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id}
    if temperature > 0:
        options.update(do_sample=True, temperature=temperature, top_p=float(args.get("top_p", 0.9)))
    else:
        options["do_sample"] = False
    failure: list[BaseException] = []

    def work() -> None:
        try:
            with torch.inference_mode():
                model.generate(**inputs, **options)
        except BaseException as exc:  # reported after the stream ends
            failure.append(exc)
            streamer.end()

    started = time.monotonic()
    thread = threading.Thread(target=work, daemon=True)
    thread.start()
    pieces = []
    for piece in streamer:
        if piece:
            pieces.append(piece)
            emit({"text": piece})
    thread.join()
    if failure:
        raise failure[0]
    seconds = time.monotonic() - started
    text = "".join(pieces)
    count = len(tokenizer(text, add_special_tokens=False)["input_ids"]) if text else 0
    return {"text": text, "tokens": count, "seconds": seconds,
            "tokens_per_s": count / seconds if seconds else 0.0,
            "stopped": bool(cancelled()), "device": str(model.device)}


def _encoder(folder: Path) -> tuple[Any, Any]:
    from transformers import AutoModel, AutoTokenizer

    return AutoModel.from_pretrained(folder).to(device()).eval(), AutoTokenizer.from_pretrained(folder)


def embed(args: dict[str, Any], emit: Any, cancelled: Any, workspace: Path) -> dict[str, Any]:
    """``texts`` → unit vectors (mean pooling over real tokens), and ``query`` similarities if given."""
    import torch

    model, tokenizer = load("encoder", resolve_model(args["model"], workspace), _encoder)
    texts = [str(t) for t in args["texts"]]
    query = args.get("query")
    everything = texts + ([str(query)] if query else [])
    vectors = []
    for start in range(0, len(everything), 32):
        if cancelled():
            raise InterruptedError("stopped")
        batch = tokenizer(everything[start:start + 32], padding=True, truncation=True, max_length=256,
                          return_tensors="pt").to(model.device)
        with torch.inference_mode():
            hidden = model(**batch).last_hidden_state
        mask = batch["attention_mask"].unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        vectors.append(torch.nn.functional.normalize(pooled, dim=-1).float().cpu())
        emit({"done": min(start + 32, len(everything)), "total": len(everything)})
    matrix = torch.cat(vectors)
    result: dict[str, Any] = {"vectors": matrix[:len(texts)].tolist(), "dimensions": int(matrix.shape[1])}
    if query:
        result["similarity"] = (matrix[:len(texts)] @ matrix[-1]).tolist()
    return result


def _tokenizer(folder: Path) -> Any:
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(folder)


def tokenize(args: dict[str, Any], emit: Any, cancelled: Any, workspace: Path) -> dict[str, Any]:
    """Each token of ``text`` with its id and (fast tokenizers) the span of text it covers."""
    tokenizer = load("tokenizer", resolve_model(args["model"], workspace), _tokenizer)
    text = str(args["text"])
    fast = getattr(tokenizer, "is_fast", False)
    encoded = tokenizer(text, add_special_tokens=bool(args.get("special", False)),
                        return_offsets_mapping=fast)
    ids = list(encoded["input_ids"])
    tokens = tokenizer.convert_ids_to_tokens(ids)
    spans = [list(span) for span in encoded["offset_mapping"]] if fast else []
    return {"ids": ids, "tokens": tokens, "spans": spans, "vocabulary": len(tokenizer),
            "characters": len(text), "kind": type(tokenizer).__name__}
