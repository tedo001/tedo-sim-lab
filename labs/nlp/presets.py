"""Preset text experiments for the NLP / LLM lab: TF-IDF classifiers (minutes on a CPU) and a
transformer fine-tune (needs the model downloaded in the Model Zoo)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from core.common.vocab import Task
from core.experiment_engine.spec import (
    CheckpointSpec,
    DataSpec,
    ExperimentSpec,
    ModelSpec,
    OptimizerSpec,
    RuntimeSpec,
    SchedulerSpec,
    SklearnTrainingSpec,
    TorchTrainingSpec,
)

__all__ = ["TEXT_PRESETS", "TextPreset", "text_spec", "transformer_spec"]


def text_spec(name: str, dataset: str, model: str, *, test_fraction: float = 0.2, subset: int | None = None,
              **params: Any) -> ExperimentSpec:
    return ExperimentSpec(name=name, task=Task.TEXT_CLASSIFICATION, data=DataSpec(dataset=dataset,
                                                                                  subset=subset),
                          model=ModelSpec(model=model, params=params),
                          training=SklearnTrainingSpec(test_fraction=test_fraction),
                          runtime=RuntimeSpec(device="cpu"))


def transformer_spec(name: str, dataset: str, model: str, *, epochs: int = 3, lr: float = 3e-5,
                     batch_size: int = 16, subset: int | None = None,
                     max_length: int = 256) -> ExperimentSpec:
    return ExperimentSpec(
        name=name, task=Task.TEXT_CLASSIFICATION, data=DataSpec(dataset=dataset, subset=subset),
        model=ModelSpec(model=model, pretrained="pretrained", params={"max_length": max_length}),
        training=TorchTrainingSpec(epochs=epochs, batch_size=batch_size,
                                   optimizer=OptimizerSpec(name="adamw", lr=lr, weight_decay=0.01),
                                   scheduler=SchedulerSpec(name="onecycle"),
                                   checkpoint=CheckpointSpec(monitor="val_acc",
                                                             mode="max"), grad_clip_norm=1.0))


@dataclass(frozen=True)
class TextPreset:
    id: str
    title: str
    summary: str
    dataset: str
    model: str
    cost: str
    make: Callable[[], ExperimentSpec]


TEXT_PRESETS: tuple[TextPreset, ...] = (
    TextPreset("reviews_tfidf", "Movie reviews · TF-IDF + logistic",
               "Positive or negative film reviews; about "
               "87% accuracy, and the words that decide it.", "movie_reviews", "tfidf_logistic", "seconds",
               lambda: text_spec("reviews-tfidf-logistic", "movie_reviews", "tfidf_logistic")),
    TextPreset("spam_nb", "SMS spam · naive Bayes", "Ham or spam text messages with complement naive Bayes; "
               "about 98% accuracy.", "sms_spam", "tfidf_naive_bayes", "seconds",
               lambda: text_spec("sms-spam-naive-bayes", "sms_spam", "tfidf_naive_bayes")),
    TextPreset("news_svm", "20 Newsgroups · linear SVM", "Twenty discussion topics with the authors' split "
               "by date; about 70% accuracy without headers or quotes.", "twenty_newsgroups",
               "tfidf_linear_svm", "under a minute",
               lambda: text_spec("newsgroups-linear-svm", "twenty_newsgroups", "tfidf_linear_svm")),
    TextPreset("reviews_distilbert", "Movie reviews · fine-tune DistilBERT", "A pretrained transformer "
               "with a new classification head; download DistilBERT in the Model Zoo first. Best on a GPU.",
               "movie_reviews", "distilbert_base_uncased", "about 10 minutes on a 4 GB GPU",
               lambda: transformer_spec("reviews-distilbert", "movie_reviews", "distilbert_base_uncased")),
)
