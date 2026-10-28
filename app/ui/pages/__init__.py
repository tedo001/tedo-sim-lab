"""Built pages by id. Every page in :data:`app.navigation.NAV` with
``planned_for=None`` must appear here; everything else gets a placeholder.
"""

from __future__ import annotations

from collections.abc import Callable

from PySide6.QtWidgets import QWidget

from ...navigation import page
from ...services.context import AppContext
from .audio import AudioPage
from .benchmarking import BenchmarkingPage
from .builder import ExperimentBuilderPage
from .classical_ml import ClassicalMlPage
from .colab_page import ColabPage
from .compare import ComparePage
from .computer_vision import ComputerVisionPage
from .dataset_hub import DatasetHubPage
from .deep_learning import DeepLearningPage
from .documentation import DocumentationPage
from .evaluation import EvaluationPage
from .explainer import ExplainerPage
from .hardware import HardwarePage
from .home import HomePage
from .ide import IdePage
from .jupyter_page import JupyterPage
from .logs import LogsPage
from .mlflow_page import MlflowPage
from .model_registry import ModelRegistryPage
from .model_zoo import ModelZooPage
from .multimodal import MultimodalPage
from .nlp import NlpLlmPage
from .placeholder import PlaceholderPage
from .plugin_store import PluginStorePage
from .quantum_ml import QuantumMlPage
from .settings import SettingsPage
from .simulation import SimulationLabPage
from .source_control import SourceControlPage
from .terminal import TerminalPage
from .testing import TestingPage
from .training import TrainingPage

__all__ = ["PAGE_FACTORIES", "build_page"]

PAGE_FACTORIES: dict[str, Callable[[AppContext], QWidget]] = {
    "home": HomePage,
    "benchmarking": BenchmarkingPage,
    "cnn_explainer": ExplainerPage,
    "classical_ml": ClassicalMlPage,
    "compare": ComparePage,
    "computer_vision": ComputerVisionPage,
    "dataset_hub": DatasetHubPage,
    "code": IdePage,
    "colab": ColabPage,
    "jupyter": JupyterPage,
    "terminal": TerminalPage,
    "logs": LogsPage,
    "source_control": SourceControlPage,
    "simulation_lab": SimulationLabPage,
    "plugin_store": PluginStorePage,
    "deep_learning": DeepLearningPage,
    "experiment_builder": ExperimentBuilderPage,
    "evaluation": EvaluationPage,
    "mlflow": MlflowPage,
    "model_registry": ModelRegistryPage,
    "model_zoo": ModelZooPage,
    "documentation": DocumentationPage,
    "hardware": HardwarePage,
    "settings": SettingsPage,
    "training": TrainingPage,
    "quantum_ml": QuantumMlPage,
    "nlp_llm": NlpLlmPage,
    "audio": AudioPage,
    "multimodal": MultimodalPage,
    "testing": TestingPage,
}


def build_page(page_id: str, ctx: AppContext) -> QWidget:
    """The page for ``page_id``: its factory if built, otherwise a placeholder."""
    spec = page(page_id)
    factory = PAGE_FACTORIES.get(page_id)
    return factory(ctx) if factory else PlaceholderPage(ctx, spec)
