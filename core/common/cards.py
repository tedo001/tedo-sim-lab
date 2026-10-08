"""Catalogue cards loaded from YAML, and the registry that searches them.

A card file holds one card (a mapping with an ``id``), a list of cards, or a
mapping with a ``cards:`` list. A broken card is recorded as a
:class:`CardError` (shown in the UI) and skipped; it never stops the others
loading or the app starting.
"""

from __future__ import annotations

import difflib
from collections.abc import Collection, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generic, TypeVar

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .licensing import LicenseCategory, LicenseInfo
from .vocab import InstallStatus, Task

__all__ = ["Card", "CardError", "CardRegistry", "format_validation_error"]


class Card(BaseModel):
    """Fields every catalogue card has."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(pattern=r"^[a-z0-9][a-z0-9_.-]*$")
    name: str
    tasks: tuple[Task, ...]
    license: LicenseInfo
    source_url: str
    description: str = ""
    tags: tuple[str, ...] = ()

    def search_text(self) -> str:
        """Everything a free-text search should match, lower-cased."""
        parts = [self.id, self.name, self.description, self.license.name, self.license.category,
                 *self.tags, *self.tasks]
        return " ".join(str(part) for part in parts).lower()


CardT = TypeVar("CardT", bound=Card)


@dataclass(frozen=True)
class CardError:
    path: Path
    message: str
    card_id: str | None = None

    def __str__(self) -> str:
        where = f"{self.path.name}" + (f" [{self.card_id}]" if self.card_id else "")
        return f"{where}: {self.message}"


def format_validation_error(error: ValidationError) -> str:
    return "; ".join(f"{'.'.join(map(str, e['loc'])) or 'card'}: {e['msg']}" for e in error.errors())


def _card_entries(data: Any) -> list[Any]:
    if isinstance(data, dict) and "cards" in data:
        data = data["cards"]
    if isinstance(data, dict):
        return [data]
    if isinstance(data, list):
        return data
    raise ValueError("expected a card, a list of cards, or a mapping with a 'cards' list")


class CardRegistry(Generic[CardT]):
    """Cards of one type, by id. Subclasses set :attr:`card_type` and :meth:`status`."""

    card_type: type[CardT]

    def __init__(self) -> None:
        self._cards: dict[str, CardT] = {}
        self._sources: dict[str, Path] = {}
        self.errors: list[CardError] = []

    # Loading -------------------------------------------------------------
    def load_dir(self, directory: Path) -> list[CardError]:
        """Load every ``*.yaml`` in ``directory`` (sorted); return the new errors."""
        errors: list[CardError] = []
        if directory.is_dir():
            for file in sorted(directory.glob("*.yaml")):
                errors += self.load_file(file)
        return errors

    def load_file(self, path: Path) -> list[CardError]:
        errors: list[CardError] = []
        try:
            entries = _card_entries(yaml.safe_load(path.read_text(encoding="utf-8")) or [])
        except (OSError, yaml.YAMLError, ValueError) as exc:
            errors.append(CardError(path, f"unreadable: {exc}"))
        else:
            for entry in entries:
                error = self._add_entry(entry, path)
                if error:
                    errors.append(error)
        self.errors += errors
        return errors

    def _add_entry(self, entry: Any, path: Path) -> CardError | None:
        card_id = entry.get("id") if isinstance(entry, dict) else None
        try:
            card = self.card_type.model_validate(entry)
        except ValidationError as exc:
            return CardError(path, format_validation_error(exc), card_id)
        return self.add(card, path)

    def add(self, card: CardT, source: Path) -> CardError | None:
        if card.id in self._cards:
            return CardError(source, f"duplicate id (first defined in {self._sources[card.id].name})",
                             card.id)
        self._cards[card.id] = card
        self._sources[card.id] = source
        return None

    # Lookup --------------------------------------------------------------
    def get(self, card_id: str) -> CardT:
        try:
            return self._cards[card_id]
        except KeyError:
            close = difflib.get_close_matches(card_id, self._cards, n=3)
            hint = f"; did you mean {', '.join(close)}?" if close else ""
            raise KeyError(f"no {self.card_type.__name__} {card_id!r}{hint}") from None

    def __contains__(self, card_id: object) -> bool:
        return card_id in self._cards

    def __len__(self) -> int:
        return len(self._cards)

    def all(self) -> list[CardT]:
        return sorted(self._cards.values(), key=lambda card: card.name.lower())

    def source(self, card_id: str) -> Path:
        return self._sources[card_id]

    def search(self, text: str = "", *, task: Task | None = None,
               license: Collection[LicenseCategory] | None = None,
               status: InstallStatus | None = None, **fields: Any) -> list[CardT]:
        """Cards matching every word of ``text`` and every filter given.

        ``fields`` compares card attributes by equality, e.g. ``framework="pytorch"``
        or ``modality="image"``.
        """
        words = text.lower().split()
        found = []
        for card in self.all():
            haystack = card.search_text()
            if any(word not in haystack for word in words):
                continue
            if task is not None and task not in card.tasks:
                continue
            if license is not None and card.license.category not in license:
                continue
            if any(getattr(card, name, None) != value for name, value in fields.items()):
                continue
            if status is not None and self.status(card.id) != status:
                continue
            found.append(card)
        return found

    def ids(self) -> Iterable[str]:
        return iter(self._cards)

    # Status --------------------------------------------------------------
    def status(self, card_id: str) -> InstallStatus:
        """What the lab can do with the card now. Subclasses probe for real."""
        self.get(card_id)
        return "catalog_only"
