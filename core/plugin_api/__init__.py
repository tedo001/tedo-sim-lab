"""The plugin contract, manifests and the plugin registry."""

from .manifest import CAPABILITIES, CredentialSpec, PluginManifest, load_manifest
from .plugin import (
    ActionSpec,
    ConnectionResult,
    InstallPlan,
    ManifestPlugin,
    Plugin,
    PluginAction,
    PluginError,
    PluginStatus,
)
from .registry import PluginLoadError, PluginRegistry

__all__ = ["CAPABILITIES", "ActionSpec", "ConnectionResult", "CredentialSpec", "InstallPlan",
           "ManifestPlugin", "Plugin", "PluginAction", "PluginError", "PluginLoadError",
           "PluginManifest", "PluginRegistry", "PluginStatus", "load_manifest"]
