"""The words cards, specs and runners share: tasks, data modalities, frameworks."""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

__all__ = ["Framework", "InstallStatus", "Maturity", "Modality", "Task", "TABULAR_TASKS"]


class Task(StrEnum):
    # Vision
    IMAGE_CLASSIFICATION = "image_classification"
    OBJECT_DETECTION = "object_detection"
    SEMANTIC_SEGMENTATION = "semantic_segmentation"
    INSTANCE_SEGMENTATION = "instance_segmentation"
    OCR = "ocr"
    POSE_ESTIMATION = "pose_estimation"
    TRACKING = "tracking"
    IMAGE_CAPTIONING = "image_captioning"
    IMAGE_GENERATION = "image_generation"
    ANOMALY_DETECTION = "anomaly_detection"
    SUPER_RESOLUTION = "super_resolution"
    # Tabular / classical
    TABULAR_CLASSIFICATION = "tabular_classification"
    TABULAR_REGRESSION = "tabular_regression"
    CLUSTERING = "clustering"
    DIMENSIONALITY_REDUCTION = "dimensionality_reduction"
    # Language and audio (later releases)
    TEXT_CLASSIFICATION = "text_classification"
    TEXT_GENERATION = "text_generation"
    SPEECH_RECOGNITION = "speech_recognition"
    AUDIO_CLASSIFICATION = "audio_classification"
    ZERO_SHOT_CLASSIFICATION = "zero_shot_classification"


#: Tasks trained with scikit-learn-style estimators rather than a PyTorch loop.
TABULAR_TASKS = frozenset({Task.TABULAR_CLASSIFICATION, Task.TABULAR_REGRESSION,
                           Task.CLUSTERING, Task.DIMENSIONALITY_REDUCTION})


class Modality(StrEnum):
    IMAGE = "image"
    TABULAR = "tabular"
    TEXT = "text"
    AUDIO = "audio"
    MULTIMODAL = "multimodal"


class Framework(StrEnum):
    PYTORCH = "pytorch"
    SKLEARN = "sklearn"
    XGBOOST = "xgboost"
    ULTRALYTICS = "ultralytics"
    PADDLE = "paddle"
    HUGGINGFACE = "huggingface"


#: How finished a card's integration is. ``planned`` cards are catalogue entries only.
Maturity = Literal["stable", "experimental", "planned"]

#: What the lab can do with a card right now, shown as a pill.
InstallStatus = Literal["ready", "not_downloaded", "not_installed", "not_connected",
                        "experimental", "planned", "catalog_only"]
