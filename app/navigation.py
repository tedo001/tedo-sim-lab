"""Every page of the lab, in sidebar order: one table drives the sidebar, the
Ctrl+K search, the placeholder pages and the smoke test.

A page with ``planned_for=None`` is built and must have a factory in
:mod:`app.ui.pages`; anything else renders as a placeholder that says when it
arrives and what it will do. Pages are added here, never wired by hand.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["NAV", "SECTIONS", "PageSpec", "page", "pages_in"]

#: (id, sidebar heading). ``home`` has no heading; ``system`` is not in the sidebar.
SECTIONS: tuple[tuple[str, str], ...] = (
    ("home", ""),
    ("labs", "Research Labs"),
    ("data", "Data & Models"),
    ("experiments", "Experimentation"),
    ("tools", "Tools"),
    ("system", ""),
)


@dataclass(frozen=True)
class PageSpec:
    id: str
    title: str
    section: str
    icon: str
    summary: str
    #: ``None`` = built. Otherwise the release it arrives in: "v0.1", "v0.2", ... or "TBD".
    planned_for: str | None = None
    #: For V0.1 pages still being built: the build phase that delivers them.
    build_phase: int | None = None
    features: tuple[str, ...] = ()
    in_sidebar: bool = True

    @property
    def built(self) -> bool:
        return self.planned_for is None


def _p(id: str, title: str, section: str, icon: str, summary: str, planned_for: str | None = None,
       build_phase: int | None = None, *features: str, in_sidebar: bool = True) -> PageSpec:
    return PageSpec(id, title, section, icon, summary, planned_for, build_phase,
                    tuple(features), in_sidebar)


NAV: tuple[PageSpec, ...] = (
    _p("home", "Home", "home", "house",
       "Live resources, the active job, recent work and the state of this workspace."),

    # ── Research labs ────────────────────────────────────────────────
    _p("computer_vision", "Computer Vision", "labs", "scan-eye",
       "Pick a vision task and start from a preset experiment."),
    _p("cnn_explainer", "CNN Explainer", "labs", "brain-circuit",
       "See how a convolutional network turns an image into a prediction, layer by layer."),
    _p("classical_ml", "Classical ML", "labs", "chart-scatter",
       "Classical models on tabular data, tracked like neural networks.", "v0.1", 6,
       "Linear and logistic regression, decision tree, random forest, gradient boosting, "
       "XGBoost, SVM, KNN, naive Bayes, PCA, k-means",
       "scikit-learn built-in datasets and CSV import; feature selection, preprocessing, "
       "cross-validation, grid and random search",
       "Metrics, feature importance, confusion matrix, ROC and PR curves; SHAP when installed"),
    _p("deep_learning", "Deep Learning", "labs", "network",
       "Design a network layer by layer and train it like any other model.", "v0.1", 9,
       "Layer stack: Conv, BatchNorm, ReLU, Pool, Dropout, Linear, Output",
       "Generated PyTorch module, architecture diagram and parameter count",
       "RNN, LSTM, GRU, autoencoders and Transformers listed as Experimental"),
    _p("nlp_llm", "NLP / LLM", "labs", "message-square-text",
       "Text and language-model experiments.", "v0.3", None,
       "Text classification and tokenization experiments",
       "Hugging Face models and datasets; gated models need your token and accepted terms"),
    _p("audio", "Audio", "labs", "audio-lines",
       "Speech and audio experiments.", "v0.2", None,
       "Audio datasets, spectrogram pipelines and speech-recognition models"),
    _p("multimodal", "Multimodal", "labs", "layers",
       "Vision-language experiments.", "v0.4", None,
       "Zero-shot classification and captioning with CLIP-style models"),
    _p("quantum_ml", "Quantum ML", "labs", "atom",
       "Quantum machine-learning experiments.", "TBD", None,
       "Scope not decided yet"),
    _p("simulation_lab", "Simulation Lab", "labs", "flask-conical",
       "Simulation experiments.", "TBD", None,
       "Scope not decided yet"),

    # ── Data & models ────────────────────────────────────────────────
    _p("dataset_hub", "Dataset Hub", "data", "database",
       "Find datasets, check their licenses, download the ones that allow it.", "v0.1", 7,
       "Cards for ImageNet, COCO, Pascal VOC, CIFAR, MNIST, Fashion-MNIST, Open Images, "
       "Cityscapes, KITTI, ADE20K, Places365, LVIS, CelebA, Oxford-IIIT Pet, Caltech-256",
       "License, source, size, classes and access requirements on every card; "
       "download only where the license permits",
       "Ontology Explorer (WordNet, ImageNet class ↔ synset) and a COCO annotation inspector"),
    _p("model_zoo", "Model Zoo", "data", "boxes",
       "Third-party models with their license, size and hardware needs.", "v0.1", 7,
       "Vision, OCR, NLP, audio and classical models",
       "Only permissively licensed models (MIT, Apache-2.0, BSD); copyleft ones such as "
       "Ultralytics YOLO (AGPL-3.0) are excluded"),
    _p("model_registry", "Model Registry", "data", "package-check",
       "Models trained in this lab and the runs they came from.", "v0.1", 7,
       "Versions, metrics, checkpoint and source run for every model"),

    # ── Experimentation ──────────────────────────────────────────────
    _p("experiment_builder", "Experiment Builder", "experiments", "sliders-horizontal",
       "Describe an experiment as a reproducible experiment.yaml and queue it."),
    _p("training", "Training", "experiments", "activity",
       "Queued, running and finished runs, live."),
    _p("evaluation", "Evaluation", "experiments", "target",
       "Evaluate a checkpoint on a held-out split.", "v0.1", 5,
       "Metrics, confusion matrix and per-class report"),
    _p("benchmarking", "Benchmarking", "experiments", "timer",
       "Inference latency and throughput.", "v0.1", 8,
       "FPS and latency by batch size and device"),
    _p("compare", "Compare Experiments", "experiments", "git-compare",
       "Runs side by side.", "v0.1", 8,
       "Model, dataset, accuracy, latency, parameters and VRAM in one table",
       "Overlaid training curves",
       "Export to CSV, JSON, Markdown and PDF"),
    _p("mlflow", "MLflow", "experiments", "chart-line",
       "Runs tracked in MLflow.", "v0.1", 5,
       "Run list and run details",
       "Launch or connect to the MLflow UI",
       "Default store: SQLite at database/mlflow.db, artifacts in mlruns/"),

    # ── Tools ────────────────────────────────────────────────────────
    _p("terminal", "Terminal", "tools", "terminal",
       "A shell in the workspace.", "v0.1", 9,
       "PowerShell on Windows, bash elsewhere",
       "Not a terminal emulator: full-screen programs such as vim or top are not supported"),
    _p("jupyter", "Jupyter Notebook", "tools", "notebook-pen",
       "Research notebook and Jupyter Lab.", "v0.1", 9,
       "Python, Markdown and shell cells on a local Jupyter kernel",
       "Attach a notebook to an experiment",
       "Launch Jupyter Lab when it is installed"),
    _p("colab", "Google Colab", "tools", "cloud",
       "Run an experiment on Google Colab.", "v0.1", 9,
       "Generate a training notebook from an experiment spec",
       "Import the results back into the lab"),
    _p("plugin_store", "Plugin Store", "tools", "puzzle",
       "Integrations and their status.", "v0.1", 9,
       "Connect Kaggle, Roboflow, Hugging Face and GitHub; credentials stay in the OS "
       "keyring or environment variables",
       "Every plugin shows its license, capabilities, and whether it is installed and connected"),
    _p("hardware", "Hardware Monitor", "tools", "cpu",
       "CPU, memory and GPU: what this machine has and how busy it is."),
    _p("documentation", "Documentation", "tools", "book-open-text",
       "The lab's own documentation."),

    # ── Not in the sidebar ───────────────────────────────────────────
    _p("settings", "Settings", "system", "settings",
       "Workspace, configuration and credential status.", in_sidebar=False),
)

_BY_ID = {spec.id: spec for spec in NAV}


def page(page_id: str) -> PageSpec:
    """The page called ``page_id``; ``KeyError`` if there is none."""
    return _BY_ID[page_id]


def pages_in(section: str) -> tuple[PageSpec, ...]:
    return tuple(spec for spec in NAV if spec.section == section)
