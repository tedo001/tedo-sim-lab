"""Model cards (``configs/models/*.yaml``) and the model registry.

Code and pretrained weights are licensed separately: ResNet-18's code is
BSD-3-Clause, but weights trained on ImageNet inherit ImageNet's terms. Each
:class:`WeightsInfo` carries its own licence for that reason.
"""

from __future__ import annotations

from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, field_validator

from core.common.cards import Card, CardRegistry
from core.common.licensing import LicenseInfo
from core.common.optional import missing_requirements
from core.common.references import UnresolvedReference, is_reference, resolve
from core.common.vocab import Framework, InstallStatus, Maturity

__all__ = ["ModelBuilder", "ModelCard", "ModelRegistry", "WeightsInfo"]


class ModelBuilder(Protocol):
    """Builds a ready-to-train model. Neural builders return a ``torch.nn.Module``."""

    def __call__(self, *, num_classes: int, in_channels: int, input_size: tuple[int, int],
                 pretrained: str | None, **params: Any) -> Any: ...


class WeightsInfo(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    license: LicenseInfo
    description: str = ""
    size_mb: float | None = None
    auto_download_allowed: bool = False
    access_requirements: str = ""


class ModelCard(Card):
    architecture: str
    framework: Framework
    #: Parameter count, when known exactly.
    params: int | None = None
    input_spec: str = ""
    weights: tuple[WeightsInfo, ...] = ()
    #: ``None`` = runs comfortably on a CPU.
    min_vram_gb: float | None = None
    #: pip requirements beyond the core install.
    requires: tuple[str, ...] = ()
    #: ``"module:function"`` returning the model; ``None`` = catalogue entry only.
    builder: str | None = None
    maturity: Maturity = "planned"
    #: For planned and experimental models: the release that makes them usable.
    planned_for: str | None = None

    @field_validator("builder")
    @classmethod
    def _reference(cls, value: str | None) -> str | None:
        if value is not None and not is_reference(value):
            raise ValueError("builder must look like 'package.module:function'")
        return value

    def weights_by_id(self, weights_id: str) -> WeightsInfo:
        for weights in self.weights:
            if weights.id == weights_id:
                return weights
        raise KeyError(f"{self.name} has no pretrained weights {weights_id!r}")


class ModelRegistry(CardRegistry[ModelCard]):
    card_type = ModelCard

    def builder(self, card_id: str) -> ModelBuilder:
        card = self.get(card_id)
        if card.maturity == "planned":
            when = f" (planned for {card.planned_for})" if card.planned_for else ""
            raise NotImplementedError(f"{card.name} cannot be built yet{when}.")
        if card.builder is None:
            raise NotImplementedError(f"{card.name} is a catalogue entry: the lab cannot build it.")
        missing = missing_requirements(card.requires)
        if missing:
            raise NotImplementedError(f"{card.name} needs {', '.join(missing)} installed.")
        return resolve(card.builder)

    def status_detail(self, card_id: str) -> tuple[InstallStatus, str]:
        card = self.get(card_id)
        if card.maturity == "planned":
            when = f" for {card.planned_for}" if card.planned_for else ""
            return "planned", f"Planned{when}."
        missing = missing_requirements(card.requires)
        if missing:
            return "not_installed", f"Needs {', '.join(missing)}."
        if card.builder is None:
            return "catalog_only", "Listed for reference; the lab cannot build it."
        try:
            resolve(card.builder)
        except UnresolvedReference as exc:
            return "not_installed", f"Builder unavailable: {exc}"
        if card.maturity == "experimental":
            return "experimental", "Works, but not yet validated."
        return "ready", "Ready to train."

    def status(self, card_id: str) -> InstallStatus:
        return self.status_detail(card_id)[0]
