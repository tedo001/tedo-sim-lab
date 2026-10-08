"""Vision tasks the lab lists but cannot run yet (Advanced CV is planned for v0.5).

Registering them lets the Computer Vision lab show each task with an honest
``Experimental`` badge; choosing one explains when it arrives instead of failing
somewhere deep inside a run.
"""

from __future__ import annotations

from core.common.vocab import Task
from core.experiment_engine.runner import ExperimentalRunner

__all__ = ["ObjectDetectionRunner", "OcrRunner", "PoseRunner", "SegmentationRunner",
           "TrackingRunner"]


class ObjectDetectionRunner(ExperimentalRunner):
    id = "object_detection"
    title = "Object detection"
    tasks = frozenset({Task.OBJECT_DETECTION})
    planned_for = "v0.5 (RT-DETR, DETR, Faster R-CNN)"


class SegmentationRunner(ExperimentalRunner):
    id = "segmentation"
    title = "Segmentation"
    tasks = frozenset({Task.SEMANTIC_SEGMENTATION, Task.INSTANCE_SEGMENTATION})
    planned_for = "v0.5 (U-Net, DeepLab, Mask R-CNN, SAM)"


class OcrRunner(ExperimentalRunner):
    id = "ocr"
    title = "OCR / text detection"
    tasks = frozenset({Task.OCR})
    planned_for = "v0.5 (PaddleOCR, Tesseract)"


class PoseRunner(ExperimentalRunner):
    id = "pose_estimation"
    title = "Pose estimation"
    tasks = frozenset({Task.POSE_ESTIMATION})
    planned_for = "v0.5"


class TrackingRunner(ExperimentalRunner):
    id = "tracking"
    title = "Tracking"
    tasks = frozenset({Task.TRACKING})
    planned_for = "v0.5"
