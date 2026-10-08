"""Credentials: read from environment variables or the OS keyring, never from files.

Lookup order is the environment first, then the keyring under the service name
``tedo-ai-lab``. Writing goes to the keyring only; nothing in the lab writes a
credential to YAML, JSON, the database or a log. Every value handed out is
registered with :mod:`core.common.masking` so it is masked wherever it shows
up later.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping
from typing import Literal, Protocol

from .masking import MASK, register_secret

__all__ = ["KNOWN_CREDENTIALS", "KEYRING_SERVICE", "CredentialError", "CredentialStore",
           "KeyringBackend", "Secret"]

log = logging.getLogger("tedo.credentials")

KEYRING_SERVICE = "tedo-ai-lab"

#: The credentials the built-in integrations use, with a label for the UI.
KNOWN_CREDENTIALS: dict[str, str] = {
    "KAGGLE_USERNAME": "Kaggle username",
    "KAGGLE_KEY": "Kaggle API key",
    "ROBOFLOW_API_KEY": "Roboflow API key",
    "HF_TOKEN": "Hugging Face token",
    "GITHUB_TOKEN": "GitHub token",
}

CredentialSource = Literal["env", "keyring", "missing"]


class CredentialError(RuntimeError):
    """A credential could not be stored or removed."""


class KeyringBackend(Protocol):
    """The part of a :mod:`keyring` backend the store uses."""

    def get_password(self, service: str, username: str) -> str | None: ...
    def set_password(self, service: str, username: str, password: str) -> None: ...
    def delete_password(self, service: str, username: str) -> None: ...


class Secret:
    """A credential value that does not print itself.

    ``str()``/``repr()`` give a mask, so an accidental ``log.info(secret)`` or
    an f-string in an exception is safe. Call :meth:`reveal` at the one place
    the raw value is needed (an HTTP header, a client constructor).
    """

    __slots__ = ("_value",)

    def __init__(self, value: str) -> None:
        self._value = value
        register_secret(value)

    def reveal(self) -> str:
        return self._value

    def __repr__(self) -> str:
        return f"Secret('{MASK}')"

    __str__ = __repr__

    def __bool__(self) -> bool:
        return bool(self._value)

    def __reduce__(self):  # pragma: no cover - guard against pickling into a file
        raise TypeError("Secret values cannot be pickled")


def _system_keyring() -> KeyringBackend | None:
    """The OS keyring, or ``None`` when no usable backend exists (headless Linux)."""
    try:
        import keyring
        from keyring.backends import fail, null
    except Exception:  # keyring missing or broken
        return None
    backend = keyring.get_keyring()
    if isinstance(backend, (fail.Keyring, null.Keyring)):
        return None
    return backend


class CredentialStore:
    """Read, store and remove credentials.

    ``env`` defaults to ``os.environ``. ``backend="system"`` uses the OS
    keyring when one is available; pass ``None`` for no keyring, or any object
    with the keyring backend methods (tests use an in-memory one).
    """

    def __init__(self, env: Mapping[str, str] | None = None,
                 backend: KeyringBackend | Literal["system"] | None = "system") -> None:
        self._env = os.environ if env is None else env
        self._backend = _system_keyring() if backend == "system" else backend

    @property
    def keyring_available(self) -> bool:
        return self._backend is not None

    def _from_keyring(self, key: str) -> str | None:
        if self._backend is None:
            return None
        try:
            return self._backend.get_password(KEYRING_SERVICE, key)
        except Exception as exc:  # a locked or broken keyring is "missing", not a crash
            log.warning("Keyring lookup for %s failed: %s", key, type(exc).__name__)
            return None

    def get(self, key: str) -> Secret | None:
        value = self._env.get(key) or self._from_keyring(key)
        return Secret(value) if value else None

    def source(self, key: str) -> CredentialSource:
        if self._env.get(key):
            return "env"
        return "keyring" if self._from_keyring(key) else "missing"

    def set(self, key: str, value: str) -> None:
        if self._backend is None:
            raise CredentialError(
                f"No OS keyring is available on this machine; set the {key} "
                "environment variable instead.")
        if not value:
            raise CredentialError(f"Refusing to store an empty value for {key}.")
        register_secret(value)
        try:
            self._backend.set_password(KEYRING_SERVICE, key, value)
        except Exception as exc:
            raise CredentialError(f"The keyring refused to store {key}: "
                                  f"{type(exc).__name__}") from None
        log.info("Stored %s in the OS keyring", key)

    def delete(self, key: str) -> None:
        if self._backend is None:
            raise CredentialError("No OS keyring is available on this machine.")
        try:
            self._backend.delete_password(KEYRING_SERVICE, key)
        except Exception as exc:
            raise CredentialError(f"Could not remove {key} from the keyring: "
                                  f"{type(exc).__name__}") from None
        log.info("Removed %s from the OS keyring", key)
