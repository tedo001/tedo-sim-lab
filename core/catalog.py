"""Everything the lab knows about: datasets, models, plugins and runners, in one place.

Cards are read from the cards shipped with the code (``<code>/configs``) and,
when the workspace is somewhere else, from the workspace's own ``configs/`` as
well, so a project can add its own datasets and models without touching the
installation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from core.common.paths import AppPaths
from core.common.secrets import CredentialStore
from core.dataset_registry import DatasetRegistry
from core.experiment_engine.runner import RunnerRegistry
from core.model_registry import ModelRegistry
from core.plugin_api import PluginRegistry

__all__ = ["IMPORTED", "Catalog", "config_roots", "load_cards", "load_catalog"]

#: Folder under the workspace's datasets/ holding imported files and their cards.
IMPORTED = "imported"


@dataclass
class Catalog:
    datasets: DatasetRegistry
    models: ModelRegistry
    plugins: PluginRegistry
    runners: RunnerRegistry

    @property
    def errors(self) -> list[str]:
        """Every card, manifest or runner that failed to load, in words."""
        return ([f"datasets/{error}" for error in self.datasets.errors]
                + [f"models/{error}" for error in self.models.errors]
                + [f"plugins/{error}" for error in self.plugins.errors]
                + [f"runners: {error}" for error in self.runners.errors])

    def counts(self) -> dict[str, int]:
        return {"datasets": len(self.datasets), "models": len(self.models),
                "plugins": len(self.plugins), "runners": len(self.runners)}


def config_roots(paths: AppPaths) -> list[Path]:
    shipped = paths.code_root / "configs"
    roots = [shipped]
    if paths.configs.resolve() != shipped.resolve():
        roots.append(paths.configs)
    return roots


def load_cards(paths: AppPaths) -> tuple[DatasetRegistry, ModelRegistry]:
    """Just the dataset and model cards (what a worker process needs)."""
    datasets = DatasetRegistry(paths.datasets)
    models = ModelRegistry()
    for root in config_roots(paths):
        datasets.load_dir(root / "datasets")
        models.load_dir(root / "models")
    imported = paths.datasets / IMPORTED
    if imported.is_dir():  # CSV files a person imported into this workspace
        datasets.load_dir(imported)
    return datasets, models


def load_catalog(paths: AppPaths, credentials: CredentialStore, *,
                 python: str | None = None) -> Catalog:
    datasets, models = load_cards(paths)
    plugins = PluginRegistry(credentials, python=python)
    plugins.discover(paths.code_root / "plugins")
    runners = RunnerRegistry()
    runners.load_config(paths.code_root / "configs" / "runners.yaml")
    return Catalog(datasets, models, plugins, runners)
