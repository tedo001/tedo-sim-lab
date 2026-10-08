"""``"package.module:attribute"`` strings: how cards and manifests point at code.

Cards are data. They name the adapter, builder, runner or plugin class to use
as a string that is imported only when needed, so a card whose code (or whose
optional dependency) is missing never stops the app starting.
"""

from __future__ import annotations

import importlib
import re
from typing import Any

__all__ = ["REFERENCE_PATTERN", "UnresolvedReference", "resolve", "is_reference"]

REFERENCE_PATTERN = re.compile(r"^[A-Za-z_][\w.]*:[A-Za-z_][\w.]*$")


class UnresolvedReference(ImportError):
    """The module or attribute a reference names could not be loaded."""


def is_reference(value: str) -> bool:
    return bool(REFERENCE_PATTERN.match(value))


def resolve(reference: str) -> Any:
    """Import ``module`` and return ``attribute`` (dotted paths allowed after the colon)."""
    if not is_reference(reference):
        raise UnresolvedReference(f"{reference!r} is not of the form 'package.module:attribute'")
    module_name, _, attribute = reference.partition(":")
    try:
        target: Any = importlib.import_module(module_name)
    except ImportError as exc:
        raise UnresolvedReference(f"cannot import {module_name}: {exc}") from exc
    for part in attribute.split("."):
        try:
            target = getattr(target, part)
        except AttributeError:
            raise UnresolvedReference(f"{module_name} has no attribute {attribute}") from None
    return target
