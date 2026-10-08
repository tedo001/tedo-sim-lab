"""Dataset cards (``configs/datasets/*.yaml``) and the dataset registry."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field, field_validator

from core.common.cards import Card, CardRegistry
from core.common.references import UnresolvedReference, is_reference, resolve
from core.common.vocab import InstallStatus, Maturity, Modality

from .adapter import DatasetAdapter

__all__ = ["DatasetCard", "DatasetRegistry"]


class DatasetCard(Card):
    modality: Modality
    #: Human-readable download / disk size: "11 MB", "~150 GB".
    size: str
    num_classes: int | None = None
    #: Split → number of samples. Empty when unknown.
    splits: dict[str, int] = Field(default_factory=dict)
    #: "" when anyone may fetch it; otherwise what a person must do first.
    access_requirements: str = ""
    auto_download_allowed: bool = False
    #: ``"module:Class"`` of the :class:`DatasetAdapter`; ``None`` = catalogue entry only.
    adapter: str | None = None
    maturity: Maturity = "planned"
    citation: str = ""

    @field_validator("adapter")
    @classmethod
    def _reference(cls, value: str | None) -> str | None:
        if value is not None and not is_reference(value):
            raise ValueError("adapter must look like 'package.module:Class'")
        return value


class DatasetRegistry(CardRegistry[DatasetCard]):
    """Dataset cards plus a live view of what is on disk under ``root``."""

    card_type = DatasetCard

    def __init__(self, root: Path) -> None:
        super().__init__()
        self.root = root

    def adapter(self, card_id: str) -> DatasetAdapter:
        """The card's adapter, ready to use with :attr:`root`."""
        card = self.get(card_id)
        if card.adapter is None:
            raise NotImplementedError(f"{card.name} is a catalogue entry: the lab has no loader for "
                                      "it. Use the source link.")
        if card.maturity == "planned":
            raise NotImplementedError(f"The {card.name} loader is planned but not built yet.")
        adapter_class = resolve(card.adapter)
        if not (isinstance(adapter_class, type) and issubclass(adapter_class, DatasetAdapter)):
            raise TypeError(f"{card.adapter} is not a DatasetAdapter")
        return adapter_class(card)

    def status_detail(self, card_id: str) -> tuple[InstallStatus, str]:
        card = self.get(card_id)
        if card.adapter is None:
            return "catalog_only", "Listed for reference; the lab has no loader for it."
        if card.maturity == "planned":
            return "planned", "Loader planned, not built yet."
        try:
            adapter = self.adapter(card_id)
        except (UnresolvedReference, TypeError) as exc:
            return "not_installed", f"Loader unavailable: {exc}"
        state = adapter.local_status(self.root)
        if state == "present":
            return ("experimental" if card.maturity == "experimental" else "ready"), "On disk."
        return "not_downloaded", "Not downloaded yet." if state == "absent" else "Partly downloaded."

    def status(self, card_id: str) -> InstallStatus:
        return self.status_detail(card_id)[0]
