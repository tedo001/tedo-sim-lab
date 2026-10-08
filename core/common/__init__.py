"""Shared foundations: workspace paths, settings, logging, credentials, optional imports.

Nothing here imports Qt, ``app``, ``labs`` or ``plugins``.
"""

from .config import AppConfig, ConfigError, experiment_python, load_config, mlflow_tracking_uri, save_config
from .logging_setup import get_logger, setup_logging
from .masking import SecretMaskingFilter, mask_text, register_secret
from .optional import distribution_version, is_installed, optional_import
from .paths import AppPaths
from .secrets import KNOWN_CREDENTIALS, CredentialError, CredentialStore, Secret

__all__ = [
    "AppConfig", "AppPaths", "ConfigError", "CredentialError", "CredentialStore", "KNOWN_CREDENTIALS",
    "Secret", "SecretMaskingFilter", "distribution_version", "experiment_python", "get_logger",
    "is_installed", "load_config", "mask_text", "mlflow_tracking_uri", "optional_import",
    "register_secret", "save_config", "setup_logging",
]
