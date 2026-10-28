"""Where a dataset goes next: the lab page for its kind of data, and a starter experiment (the task,
the dataset and the first model ready for it) to open in the Experiment Builder."""

from __future__ import annotations

from core.common.vocab import TABULAR_TASKS, Modality, Task
from core.dataset_registry.cards import DatasetCard
from core.experiment_engine.spec import ExperimentSpec
from core.model_registry.cards import ModelRegistry

__all__ = ["LAB_FOR", "lab_for", "starter_spec", "trainable_tasks"]

#: Modality → the lab page that works with it.
LAB_FOR = {Modality.IMAGE: "computer_vision", Modality.TABULAR: "classical_ml", Modality.TEXT: "nlp_llm",
           Modality.AUDIO: "audio", Modality.MULTIMODAL: "multimodal"}
#: Preferred first model per task (if ready), else the first ready one.
_FIRST = {Task.IMAGE_CLASSIFICATION: "simple_cnn", Task.AUDIO_CLASSIFICATION: "simple_cnn",
          Task.TEXT_CLASSIFICATION: "tfidf_logistic", Task.QUANTUM_CLASSIFICATION: "vqc",
          Task.TABULAR_CLASSIFICATION: "random_forest", Task.TABULAR_REGRESSION: "random_forest",
          Task.CLUSTERING: "kmeans", Task.DIMENSIONALITY_REDUCTION: "pca"}


def lab_for(card: DatasetCard) -> str:
    if Task.QUANTUM_CLASSIFICATION in card.tasks:
        return "quantum_ml"
    return LAB_FOR.get(card.modality, "dataset_hub")


def trainable_tasks(card: DatasetCard, stable: set[Task]) -> list[Task]:
    """The card's tasks some stable runner trains, in the card's order."""
    return [task for task in card.tasks if task in stable]


def _model(models: ModelRegistry, task: Task) -> str | None:
    ready = [card.id for card in models.all() if task in card.tasks and models.status(card.id) == "ready"]
    preferred = _FIRST.get(task)
    return preferred if preferred in ready else (ready[0] if ready else None)


def starter_spec(card: DatasetCard, task: Task, models: ModelRegistry) -> ExperimentSpec:
    """A sensible first experiment for ``card`` and ``task``; ``LookupError`` if no model is ready."""
    model = _model(models, task)
    if model is None:
        raise LookupError(f"no model for {task.value.replace('_', ' ')} is ready; see the Model Zoo")
    name = f"{card.id}-{model}".replace("_", "-")
    if task in (Task.IMAGE_CLASSIFICATION, Task.AUDIO_CLASSIFICATION):
        from labs.computer_vision.presets import image_spec

        return image_spec(name, card.id, model, 5, task=task)
    if task == Task.TEXT_CLASSIFICATION:
        from labs.nlp.presets import text_spec, transformer_spec

        if models.get(model).framework.value == "huggingface":
            return transformer_spec(name, card.id, model)
        return text_spec(name, card.id, model)
    if task == Task.QUANTUM_CLASSIFICATION:
        from labs.quantum_ml.presets import quantum_spec

        return quantum_spec(name, card.id, model)
    if task in TABULAR_TASKS:
        from core.experiment_engine.spec import DataSpec, ModelSpec, RuntimeSpec, SklearnTrainingSpec

        return ExperimentSpec(name=name, task=task, data=DataSpec(dataset=card.id),
                              model=ModelSpec(model=model),
                              training=SklearnTrainingSpec(), runtime=RuntimeSpec(device="cpu"))
    raise LookupError(f"{task.value.replace('_', ' ')} cannot be trained in this release")
