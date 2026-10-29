"""What a child process of the lab inherits: only what it needs, never the person's secrets.

Every process the lab starts gets its environment from :func:`child_environment`, which builds it
from an allow-list instead of copying the app's:

- **lab processes** (experiment and inference workers, the remote dispatcher, the hardware probe,
  data checks, environment queries, the MLflow UI) see the operating system's essentials (paths,
  locale, temporary folder, user folders), proxies and certificate bundles, GPU and thread
  settings, and the lab's own variables, plus the names the person passes on in
  ``settings.yaml`` (``pass_environment``). Anything that looks like a credential (``TOKEN``,
  ``KEY``, ``SECRET``, ``PASSWORD``, ``CREDENTIAL``) is dropped unless passed on by name;
- **installs** (pip into the experiment Python, environment packs, plugins) see the same plus
  pip's own settings, so an index or proxy configured for pip keeps working;
- both drop the host's Python settings (``PYTHONPATH``, ``PYTHONHOME``, ``VIRTUAL_ENV``, conda)
  and compile bytecode into a per-user cache instead of the lab's code folder
  (``PYTHONPYCACHEPREFIX``). The experiment Python the app makes is a virtual environment, which
  never sees the user's ``~/.local`` packages; a plain interpreter the person chose keeps them
  (``pip install --user`` is how many set one up), unless they set ``PYTHONNOUSERSITE`` themselves;
- lab processes keep their caches in the workspace (Hugging Face, PyTorch hub, matplotlib, other
  XDG caches, nltk data), so a project folder holds everything its runs downloaded.

Programs the person runs themselves (the Code page, terminals, Jupyter) keep their own environment:
:func:`user_environment` only removes the host's Python settings that would mix interpreters."""

from __future__ import annotations

import os
import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Literal

__all__ = ["HOST_PYTHON", "MLFLOW_AUTH", "SECRET_NAME", "SSH_AGENT", "child_environment", "is_secret_name",
           "passed_names", "pycache_dir", "user_environment", "workspace_caches"]

Kind = Literal["lab", "install"]

#: Variables a lab process may inherit (case-insensitive on Windows, as the OS treats them).
_SYSTEM = frozenset(name.upper() for name in (
    # Paths and the user's folders
    "PATH", "PATHEXT", "HOME", "USER", "USERNAME", "LOGNAME", "SHELL", "TMPDIR", "TEMP", "TMP",
    "USERPROFILE", "HOMEDRIVE", "HOMEPATH", "APPDATA", "LOCALAPPDATA", "PROGRAMDATA", "PROGRAMFILES",
    "PROGRAMFILES(X86)", "COMMONPROGRAMFILES", "SYSTEMROOT", "SYSTEMDRIVE", "WINDIR", "COMSPEC",
    "COMPUTERNAME", "OS", "NUMBER_OF_PROCESSORS", "PROCESSOR_ARCHITECTURE", "PROCESSOR_IDENTIFIER",
    "XDG_RUNTIME_DIR", "XDG_DATA_HOME", "XDG_CONFIG_HOME", "LD_LIBRARY_PATH", "DYLD_LIBRARY_PATH",
    # Locale and time
    "LANG", "LANGUAGE", "TZ",
    # Network: proxies and certificate bundles
    "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY", "SSL_CERT_FILE", "SSL_CERT_DIR",
    "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE", "NODE_EXTRA_CA_CERTS",
    # GPUs and threads
    "CUDA_VISIBLE_DEVICES", "CUDA_PATH", "CUDA_HOME", "CUDA_DEVICE_ORDER", "PYTORCH_CUDA_ALLOC_CONF",
    "CUBLAS_WORKSPACE_CONFIG", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
    # The person's own choice to hide user site-packages
    "PYTHONNOUSERSITE",
))
_PREFIXES = ("LC_", "TEDO_LAB_", "CUDA_PATH_", "NVIDIA_", "KMP_", "MLFLOW_", "HF_", "NLTK_")
_INSTALL_PREFIXES = ("PIP_",)
#: The host's Python settings: never handed to another interpreter.
HOST_PYTHON = frozenset(("PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP", "PYTHONUSERBASE", "PYTHONSAFEPATH",
                         "PYTHONEXECUTABLE", "__PYVENV_LAUNCHER__", "VIRTUAL_ENV", "VIRTUAL_ENV_PROMPT",
                         "CONDA_PREFIX", "CONDA_DEFAULT_ENV", "CONDA_PYTHON_EXE", "CONDA_EXE",
                         "CONDA_SHLVL", "PYTHONPYCACHEPREFIX"))
#: What OpenSSH needs to reach the person's agent (passed to the remote dispatcher only).
SSH_AGENT = ("SSH_AUTH_SOCK", "SSH_AGENT_PID")
#: MLflow's own login for a tracking server (passed on only when the tracking URI is a server).
MLFLOW_AUTH = ("MLFLOW_TRACKING_USERNAME", "MLFLOW_TRACKING_PASSWORD", "MLFLOW_TRACKING_TOKEN",
               "MLFLOW_TRACKING_CLIENT_CERT_PATH", "MLFLOW_TRACKING_SERVER_CERT_PATH", "DATABRICKS_HOST",
               "DATABRICKS_TOKEN")
SECRET_NAME = re.compile(r"(?i)(token|secret|passw|credential|api_?key|(^|_)key($|_)|_auth$|^auth_)")


def is_secret_name(name: str) -> bool:
    """Whether a variable's name says it holds a credential (``HF_TOKEN``, ``KAGGLE_KEY``,
    ``AWS_SECRET_ACCESS_KEY``, ``MLFLOW_TRACKING_PASSWORD`` …)."""
    return bool(SECRET_NAME.search(name))


def passed_names(config) -> tuple[str, ...]:  # noqa: ANN001 (AppConfig; core.common.config imports this)
    """The names a person's settings let through: ``pass_environment``, and MLflow's login when
    runs are tracked on a server rather than in the workspace."""
    names = list(config.pass_environment)
    server = config.mlflow_tracking_uri.startswith(("http://", "https://", "databricks"))
    if config.mlflow_tracking and server:
        names += MLFLOW_AUTH
    return tuple(names)


def _wanted(name: str, kind: Kind) -> bool:
    upper = name.upper()
    if upper in HOST_PYTHON:
        return False
    if upper in _SYSTEM or upper.startswith(_PREFIXES):
        return True
    return kind == "install" and upper.startswith(_INSTALL_PREFIXES)


def pycache_dir() -> Path:
    """Per-user folder for compiled bytecode, so nothing is written into the lab's code folder."""
    from .experiment_env import default_env_dir

    return default_env_dir().parent / "pycache"


def workspace_caches(workspace: Path) -> dict[str, str]:
    """Library caches inside the workspace: a project keeps what its runs downloaded."""
    workspace = Path(workspace)
    return {"HF_HOME": str(workspace / "models" / "huggingface-cache"),
            "TORCH_HOME": str(workspace / "models" / "torch-hub"),
            "MPLCONFIGDIR": str(workspace / "cache" / "matplotlib"),
            "XDG_CACHE_HOME": str(workspace / "cache"),
            "NLTK_DATA": str(workspace / "datasets" / "nltk_data")}


def child_environment(*, kind: Kind = "lab", base: Mapping[str, str] | None = None,
                      code_root: Path | None = None, workspace: Path | None = None,
                      allow: Iterable[str] = (), extra: Mapping[str, str] | None = None) -> dict[str, str]:
    """The whole environment of a child process (pass it as is, never on top of the app's own).

    ``base``: where values come from (the app's environment); ``code_root`` goes on
    ``PYTHONPATH`` (nothing else does); ``workspace`` sets ``TEDO_LAB_WORKSPACE`` and the caches;
    ``allow``: names passed on even when they look like credentials (``pass_environment``, the
    SSH agent for the remote dispatcher); ``extra`` is set last."""
    source = os.environ if base is None else base
    allowed = {name.upper() for name in allow}
    env = {name: value for name, value in source.items()
           if name.upper() in allowed or (_wanted(name, kind) and not is_secret_name(name))}
    env["PYTHONPYCACHEPREFIX"] = str(pycache_dir())
    env["PYTHONIOENCODING"] = "utf-8"
    if code_root is not None:
        env["PYTHONPATH"] = str(code_root)
    if workspace is not None:
        from .paths import WORKSPACE_ENV

        env[WORKSPACE_ENV] = str(workspace)
        for name, value in workspace_caches(workspace).items():
            if name == "NLTK_DATA" and name in env:  # the person's own corpora; the lab adds its folder
                continue
            env[name] = value
    env.update(extra or {})
    return env


def user_environment(base: Mapping[str, str] | None = None) -> dict[str, str]:
    """The person's own environment for programs they run (Code page, terminals, notebooks),
    without the host's Python settings that would mix one interpreter's packages into another."""
    source = os.environ if base is None else base
    return {name: value for name, value in source.items() if name.upper() not in HOST_PYTHON}
