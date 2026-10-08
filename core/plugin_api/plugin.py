"""The plugin contract, and :class:`ManifestPlugin`, which derives most of it from the manifest.

A plugin never pretends: its status is computed live from what is installed,
which credentials are set and how finished it is, and every action it lists
is either runnable or carries the reason it is not.
"""

from __future__ import annotations

import re
import subprocess
import sys
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, ClassVar, Literal

from core.common.licensing import LicenseCategory, LicenseInfo
from core.common.masking import mask_text
from core.common.optional import missing_requirements
from core.common.secrets import CredentialStore

from .manifest import PluginManifest

__all__ = ["ActionSpec", "ConnectionResult", "InstallPlan", "ManifestPlugin", "Plugin", "PluginAction",
           "PluginError", "PluginStatus"]

PluginStatus = Literal["available", "not_installed", "not_connected", "experimental", "planned"]

_CREDENTIAL_KEY = re.compile(r"(?i)(key|token|secret|password|passwd|credential)")


class PluginError(RuntimeError):
    """A plugin operation failed; the message is safe to show (secrets masked)."""


@dataclass(frozen=True)
class PluginAction:
    id: str
    label: str
    description: str = ""
    enabled: bool = True
    #: Tooltip when disabled.
    disabled_reason: str = ""


@dataclass(frozen=True)
class InstallPlan:
    requirements: tuple[str, ...]
    missing: tuple[str, ...]
    command: tuple[str, ...]
    #: Text to accept first (copyleft or restrictive terms); ``None`` = nothing to accept.
    acknowledgement: str | None = None

    @property
    def needed(self) -> bool:
        return bool(self.missing)


@dataclass(frozen=True)
class ConnectionResult:
    ok: bool
    message: str


class Plugin(ABC):
    def __init__(self, manifest: PluginManifest, credentials: CredentialStore) -> None:
        self.manifest = manifest
        self.credentials = credentials

    @property
    def name(self) -> str:
        return self.manifest.name

    @property
    def version(self) -> str:
        return self.manifest.version

    @property
    def author(self) -> str:
        return self.manifest.author

    @property
    def license(self) -> LicenseInfo:
        return self.manifest.license

    @property
    def capabilities(self) -> frozenset[str]:
        return self.manifest.capabilities

    @property
    @abstractmethod
    def status(self) -> PluginStatus: ...

    @abstractmethod
    def install_plan(self) -> InstallPlan: ...

    @abstractmethod
    def install(self) -> None:
        """Install what :meth:`install_plan` lists. Blocking: call it from a background job."""

    @abstractmethod
    def configure(self, settings: Mapping[str, Any]) -> None: ...

    @abstractmethod
    def actions(self) -> list[PluginAction]: ...

    @abstractmethod
    def run(self, action: str, **kwargs: Any) -> Any: ...

    @abstractmethod
    def test_connection(self) -> ConnectionResult: ...

    @abstractmethod
    def uninstall(self) -> None: ...


@dataclass(frozen=True)
class ActionSpec:
    id: str
    label: str
    description: str = ""
    needs_connection: bool = False
    #: Set while the action is not built: "v0.1 · build phase 9", "v0.5", ...
    planned_for: str | None = None


class ManifestPlugin(Plugin):
    """Status, install, configure and action gating straight from the manifest.

    Subclasses declare :attr:`ACTIONS` and implement :meth:`_run` for the
    actions that are built.
    """

    ACTIONS: ClassVar[tuple[ActionSpec, ...]] = ()

    def __init__(self, manifest: PluginManifest, credentials: CredentialStore, *,
                 python: str | None = None) -> None:
        super().__init__(manifest, credentials)
        self.python = python or sys.executable
        self.settings: dict[str, Any] = {}

    # Status ---------------------------------------------------------------
    def missing_credentials(self) -> list[str]:
        return [spec.key for spec in self.manifest.credentials
                if spec.required and self.credentials.source(spec.key) == "missing"]

    @property
    def status(self) -> PluginStatus:
        if self.manifest.maturity == "planned":
            return "planned"
        if missing_requirements(self.manifest.requires):
            return "not_installed"
        if self.manifest.maturity == "experimental":
            return "experimental"
        if self.missing_credentials():
            return "not_connected"
        return "available"

    # Install --------------------------------------------------------------
    def install_plan(self) -> InstallPlan:
        missing = tuple(missing_requirements(self.manifest.requires))
        licence = self.manifest.license
        acknowledgement = None
        if licence.copyleft:
            acknowledgement = (f"{licence.name} is copyleft: work that uses it may have to be "
                               "released under the same licence.")
        elif licence.category not in (LicenseCategory.OPEN_SOURCE,):
            acknowledgement = f"I accept the terms of {licence.name}."
        return InstallPlan(self.manifest.requires, missing,
                           (self.python, "-m", "pip", "install", *missing), acknowledgement)

    def install(self) -> None:
        plan = self.install_plan()
        if not plan.needed:
            return
        if getattr(sys, "frozen", False) and self.python == sys.executable:
            raise PluginError("The installed app cannot add packages to itself. Choose a Python "
                              "environment for experiments in Settings, then install there.")
        result = subprocess.run(plan.command, capture_output=True, text=True)
        if result.returncode != 0:
            tail = "\n".join((result.stdout + result.stderr).strip().splitlines()[-15:])
            raise PluginError(mask_text(f"pip failed ({result.returncode}):\n{tail}"))

    def uninstall(self) -> None:
        from packaging.requirements import Requirement
        names = [Requirement(requirement).name for requirement in self.manifest.requires]
        if names:
            result = subprocess.run((self.python, "-m", "pip", "uninstall", "-y", *names),
                                    capture_output=True, text=True)
            if result.returncode != 0:
                raise PluginError(mask_text(f"pip uninstall failed: {result.stderr.strip()[-500:]}"))

    # Settings -------------------------------------------------------------
    def configure(self, settings: Mapping[str, Any]) -> None:
        for key in settings:
            if _CREDENTIAL_KEY.search(str(key)):
                raise PluginError(f"{key!r} looks like a credential: store credentials in the OS "
                                  "keyring or an environment variable, not in plugin settings.")
        self.settings.update(settings)

    # Actions --------------------------------------------------------------
    def actions(self) -> list[PluginAction]:
        status = self.status
        result = []
        for spec in self.ACTIONS:
            reason = ""
            planned = spec.planned_for or (self.manifest.planned_for
                                           if self.manifest.maturity == "planned" else None)
            if planned:
                reason = f"Planned for {planned}"
            elif status == "not_installed":
                reason = f"Install {', '.join(missing_requirements(self.manifest.requires))} first"
            elif spec.needs_connection and self.missing_credentials():
                reason = f"Not connected: set {', '.join(self.missing_credentials())}"
            result.append(PluginAction(spec.id, spec.label, spec.description, not reason, reason))
        return result

    def run(self, action: str, **kwargs: Any) -> Any:
        found = {item.id: item for item in self.actions()}
        if action not in found:
            raise PluginError(f"{self.manifest.title} has no action {action!r}")
        if not found[action].enabled:
            raise NotImplementedError(f"{self.manifest.title}: {found[action].label} is not "
                                      f"available. {found[action].disabled_reason}.")
        return self._run(action, **kwargs)

    def _run(self, action: str, **kwargs: Any) -> Any:
        raise NotImplementedError(f"{self.manifest.title}: {action} is not implemented")

    def test_connection(self) -> ConnectionResult:
        if not self.manifest.credentials:
            return ConnectionResult(True, "No account needed.")
        missing = self.missing_credentials()
        if missing:
            return ConnectionResult(False, f"Not connected: set {', '.join(missing)}.")
        planned = self.manifest.planned_for if self.manifest.maturity == "planned" else None
        if planned:
            return ConnectionResult(False, f"Credentials are set; testing them arrives in {planned}.")
        return self._test_connection()

    def _test_connection(self) -> ConnectionResult:
        return ConnectionResult(False, "This plugin cannot test its connection yet.")
