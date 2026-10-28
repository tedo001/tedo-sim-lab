"""What can be tested: every finished run with a saved model (image, spectrogram, text, table and
quantum classifiers, regressors) and every downloaded pretrained model that answers on its own
(object detectors, CLIP, BLIP, speech recognisers)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from core.common.vocab import Task
from labs.inference.runs import kind_of

from ....services.context import AppContext
from ..lab_parts import hub_repo_of

__all__ = ["MODEL_KINDS", "Subject", "subjects"]

#: Pretrained-model task → the kind of test it gets.
MODEL_KINDS = {Task.OBJECT_DETECTION: "detect", Task.ZERO_SHOT_CLASSIFICATION: "zero_shot",
               Task.IMAGE_CAPTIONING: "caption", Task.SPEECH_RECOGNITION: "speech"}
_UNTESTABLE = {Task.CLUSTERING.value, Task.DIMENSIONALITY_REDUCTION.value}
KIND_TITLES = {"image": "image classifier", "audio": "spoken-word classifier (spectrograms)",
               "text": "text classifier", "table": "table model", "quantum": "quantum classifier",
               "detect": "object detector", "zero_shot": "zero-shot image classifier",
               "caption": "image captioner", "speech": "speech recogniser"}


@dataclass(frozen=True)
class Subject:
    key: str             # "run:<id>" | "model:<card id>"
    title: str
    kind: str
    run_id: str | None = None
    run_dir: Path | None = None
    repo: str | None = None

    @property
    def is_run(self) -> bool:
        return self.run_id is not None

    @property
    def describe(self) -> str:
        return KIND_TITLES.get(self.kind, self.kind)


def subjects(ctx: AppContext) -> list[Subject]:
    found: list[Subject] = []
    for view in ctx.experiments.views():
        if view.status not in ("completed", "early_stopped") or view.spec is None or view.task in _UNTESTABLE:
            continue
        if not (view.run_dir / "checkpoints").is_dir():
            continue
        kind = kind_of(view.spec)
        found.append(Subject(f"run:{view.id}", f"{view.name} · {view.model} · {view.created_at[:16]}", kind,
                             view.id, view.run_dir))
    models = ctx.catalog.models
    for card in models.all():
        kinds = [MODEL_KINDS[task] for task in card.tasks if task in MODEL_KINDS]
        if kinds and card.hub_weights and models.downloaded(card.id):
            found.append(Subject(f"model:{card.id}", f"{card.name} (pretrained)", kinds[0],
                                 repo=hub_repo_of(card)))
    return found
