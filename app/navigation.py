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
       "Classical models on tabular data, tracked like neural networks."),
    _p("deep_learning", "Deep Learning", "labs", "network",
       "Design a network layer by layer and train it like any other model."),
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
       "Find datasets, check their licences, download the ones that allow it; explore ImageNet "
       "classes in WordNet; inspect COCO annotations."),
    _p("model_zoo", "Model Zoo", "data", "boxes",
       "Models with their licence, weights' terms, size and hardware needs; copyleft ones are kept out."),
    _p("model_registry", "Model Registry", "data", "package-check",
       "Models trained in this lab and the runs they came from."),

    # ── Experimentation ──────────────────────────────────────────────
    _p("experiment_builder", "Experiment Builder", "experiments", "sliders-horizontal",
       "Describe an experiment as a reproducible experiment.yaml and queue it."),
    _p("training", "Training", "experiments", "activity",
       "Queued, running and finished runs, live."),
    _p("evaluation", "Evaluation", "experiments", "target",
       "Scores, confusion matrix and per-class report; re-score a checkpoint on a split."),
    _p("benchmarking", "Benchmarking", "experiments", "timer",
       "Inference latency and throughput by batch size and device."),
    _p("compare", "Compare Experiments", "experiments", "git-compare",
       "Runs side by side, curves overlaid, exported as CSV, JSON, Markdown or PDF."),
    _p("mlflow", "MLflow", "experiments", "chart-line",
       "Runs tracked in MLflow, and the MLflow UI."),

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
