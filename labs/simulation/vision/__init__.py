"""Computer vision: an image-processing pipeline, test images, synthetic scenes with COCO
annotations, a pinhole camera, and PNG files without an imaging library."""

from .camera import Camera, project, projection_matrix, wireframe
from .ops import OPERATIONS, Operation, Param, Step, run_pipeline
from .samples import SAMPLES, sample
from .scenes import CATEGORIES, SceneSpec, generate_dataset, render_scene

__all__ = ["CATEGORIES", "OPERATIONS", "SAMPLES", "Camera", "Operation", "Param", "SceneSpec", "Step",
           "generate_dataset", "project", "projection_matrix", "render_scene", "run_pipeline", "sample",
           "wireframe"]
