"""Find plugins (``*/plugin.yaml``) and load their code only when one is used."""

from __future__ import annotations

import sys
from pathlib import Path

import yaml
from pydantic import ValidationError

from core.common.cards import CardError, format_validation_error
from core.common.references import UnresolvedReference, resolve
from core.common.secrets import CredentialStore

from .manifest import PluginManifest, load_manifest
from .plugin import ManifestPlugin, Plugin, PluginStatus

__all__ = ["PluginLoadError", "PluginRegistry"]


class PluginLoadError(ImportError):
    """A plugin's entry module could not be imported or is not a Plugin."""


class PluginRegistry:
    def __init__(self, credentials: CredentialStore, *, python: str | None = None) -> None:
        self.credentials = credentials
        self.python = python or sys.executable
        self._manifests: dict[str, PluginManifest] = {}
        self._folders: dict[str, Path] = {}
        self._plugins: dict[str, Plugin] = {}
        self.errors: list[CardError] = []

    def discover(self, root: Path) -> list[CardError]:
        """Read every ``root/*/plugin.yaml``; import nothing. Returns the new errors."""
        errors: list[CardError] = []
        for file in sorted(root.glob("*/plugin.yaml")) if root.is_dir() else []:
            try:
                manifest = load_manifest(file)
            except ValidationError as exc:
                errors.append(CardError(file, format_validation_error(exc), file.parent.name))
                continue
            except (OSError, ValueError, yaml.YAMLError) as exc:
                errors.append(CardError(file, f"unreadable: {exc}", file.parent.name))
                continue
            if manifest.name in self._manifests:
                errors.append(CardError(file, "duplicate plugin name", manifest.name))
                continue
            if manifest.name != file.parent.name:
                errors.append(CardError(file, f"name {manifest.name!r} must match its folder "
                                              f"{file.parent.name!r}", manifest.name))
                continue
            self._manifests[manifest.name] = manifest
            self._folders[manifest.name] = file.parent
        self.errors += errors
        return errors

    def manifests(self) -> list[PluginManifest]:
        return sorted(self._manifests.values(), key=lambda manifest: manifest.title.lower())

    def manifest(self, name: str) -> PluginManifest:
        return self._manifests[name]

    def __len__(self) -> int:
        return len(self._manifests)

    def __contains__(self, name: object) -> bool:
        return name in self._manifests

    def with_capability(self, capability: str) -> list[PluginManifest]:
        return [manifest for manifest in self.manifests() if capability in manifest.capabilities]

    def get(self, name: str) -> Plugin:
        """The plugin instance, importing its entry module on first use."""
        if name not in self._plugins:
            manifest = self._manifests[name]
            try:
                plugin_class = resolve(manifest.entry)
            except UnresolvedReference as exc:
                raise PluginLoadError(f"{manifest.title}: {exc}") from exc
            if not (isinstance(plugin_class, type) and issubclass(plugin_class, Plugin)):
                raise PluginLoadError(f"{manifest.entry} is not a Plugin subclass")
            if issubclass(plugin_class, ManifestPlugin):
                plugin = plugin_class(manifest, self.credentials, python=self.python)
            else:
                plugin = plugin_class(manifest, self.credentials)
            self._plugins[name] = plugin
        return self._plugins[name]

    def status(self, name: str) -> PluginStatus:
        try:
            return self.get(name).status
        except PluginLoadError:
            return "not_installed"
