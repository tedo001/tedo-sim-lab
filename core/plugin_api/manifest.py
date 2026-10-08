"""``plugins/<name>/plugin.yaml``: what a plugin is, before any of its code is imported."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

from core.common.licensing import LicenseInfo
from core.common.references import is_reference

__all__ = ["CAPABILITIES", "CredentialSpec", "PluginManifest", "load_manifest"]

#: What plugins can contribute. Pages ask the registry for plugins by capability.
CAPABILITIES = frozenset({"dataset_source", "model_source", "notebook", "tracker", "ocr",
                          "detector", "exporter", "custom"})


class CredentialSpec(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    #: Environment variable name and OS-keyring entry name.
    key: str = Field(pattern=r"^[A-Z][A-Z0-9_]*$")
    label: str
    secret: bool = True
    #: ``False`` = the plugin works without it (e.g. public Hugging Face repos).
    required: bool = True


class PluginManifest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    title: str
    version: str
    author: str
    description: str
    #: The licence of what the plugin integrates (the third-party package or service).
    license: LicenseInfo
    capabilities: frozenset[str]
    #: ``"plugins.kaggle.plugin:KagglePlugin"``, imported only when the plugin is used.
    entry: str
    requires: tuple[str, ...] = ()
    credentials: tuple[CredentialSpec, ...] = ()
    maturity: Literal["stable", "experimental", "planned"] = "planned"
    #: ``builtin`` = this lab's own code; ``third_party`` integrations must pass the
    #: permissive-licence policy.
    origin: Literal["builtin", "third_party"] = "third_party"
    #: For planned plugins: when they arrive, e.g. "v0.1 · build phase 9" or "v0.5".
    planned_for: str | None = None
    homepage: str | None = None

    @field_validator("capabilities")
    @classmethod
    def _known(cls, value: frozenset[str]) -> frozenset[str]:
        unknown = value - CAPABILITIES
        if unknown:
            raise ValueError(f"unknown capabilities {sorted(unknown)}; known: {sorted(CAPABILITIES)}")
        return value

    @field_validator("entry")
    @classmethod
    def _reference(cls, value: str) -> str:
        if not is_reference(value):
            raise ValueError("entry must look like 'plugins.name.module:Class'")
        return value


def load_manifest(path: Path) -> PluginManifest:
    """Parse and validate one ``plugin.yaml``; raises ``ValueError`` or ``ValidationError``."""
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("plugin.yaml must be a mapping")
    return PluginManifest.model_validate(data)
