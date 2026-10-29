"""What child processes inherit: no credentials, no host Python settings, caches in the workspace,
names passed on only when the person says so; the person's own programs keep their environment."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from core.common import AppConfig, AppPaths, load_config, save_config
from core.common.isolation import (
    MLFLOW_AUTH,
    child_environment,
    is_secret_name,
    passed_names,
    user_environment,
)
from core.common.paths import CODE_ROOT, WORKSPACE_ENV

HOST = {
    "PATH": "/usr/bin", "HOME": "/home/lab", "LC_ALL": "C.UTF-8", "HTTPS_PROXY": "http://proxy:3128",
    "no_proxy": "localhost", "CUDA_VISIBLE_DEVICES": "0", "OMP_NUM_THREADS": "4", "TEDO_LAB_MLFLOW": "0",
    "HF_HUB_OFFLINE": "1",
    # Credentials of every shape
    "HF_TOKEN": "hf_x", "KAGGLE_KEY": "k", "GITHUB_TOKEN": "g", "AWS_SECRET_ACCESS_KEY": "a",
    "OPENAI_API_KEY": "o", "MLFLOW_TRACKING_PASSWORD": "m", "DB_PASSWORD": "d", "WANDB_API_KEY": "w",
    # Not the lab's business
    "RANDOM_APP_SETTING": "r", "SSH_AUTH_SOCK": "/tmp/agent.sock", "PIP_INDEX_URL": "https://index",
    # The host's Python
    "PYTHONPATH": "/host/site", "PYTHONHOME": "/host", "VIRTUAL_ENV": "/host/venv", "CONDA_PREFIX": "/conda",
    "PYTHONSTARTUP": "/host/startup.py",
}
SECRETS = ("HF_TOKEN", "KAGGLE_KEY", "GITHUB_TOKEN", "AWS_SECRET_ACCESS_KEY", "OPENAI_API_KEY",
           "MLFLOW_TRACKING_PASSWORD", "DB_PASSWORD", "WANDB_API_KEY")


def test_a_lab_process_gets_what_it_needs_and_nothing_else(tmp_path) -> None:
    env = child_environment(base=HOST, code_root=CODE_ROOT, workspace=tmp_path)
    for name in ("PATH", "HOME", "LC_ALL", "HTTPS_PROXY", "no_proxy", "CUDA_VISIBLE_DEVICES",
                 "OMP_NUM_THREADS", "TEDO_LAB_MLFLOW", "HF_HUB_OFFLINE"):
        assert env[name] == HOST[name], name
    for name in (*SECRETS, "RANDOM_APP_SETTING", "SSH_AUTH_SOCK", "PIP_INDEX_URL", "PYTHONHOME",
                 "VIRTUAL_ENV", "CONDA_PREFIX", "PYTHONSTARTUP"):
        assert name not in env, name
    assert env["PYTHONPATH"] == str(CODE_ROOT)  # the lab's code, never the host's path
    assert env[WORKSPACE_ENV] == str(tmp_path) and "PYTHONNOUSERSITE" not in env  # the interpreter's own
    assert child_environment(base={**HOST, "PYTHONNOUSERSITE": "1"})["PYTHONNOUSERSITE"] == "1"  # theirs
    assert not Path(env["PYTHONPYCACHEPREFIX"]).is_relative_to(CODE_ROOT)  # nothing written into the code
    for name in ("HF_HOME", "TORCH_HOME", "MPLCONFIGDIR", "XDG_CACHE_HOME", "NLTK_DATA"):
        assert Path(env[name]).is_relative_to(tmp_path), name  # every cache inside the workspace


def test_names_pass_only_when_named_and_installs_keep_pip(tmp_path) -> None:
    env = child_environment(base=HOST, allow=("wandb_api_key", "SSH_AUTH_SOCK"))
    assert env["WANDB_API_KEY"] == "w" and env["SSH_AUTH_SOCK"] == "/tmp/agent.sock"
    assert "HF_TOKEN" not in env and "PYTHONPATH" not in env  # no code root: nothing on the path
    install = child_environment(kind="install", base=HOST)
    assert install["PIP_INDEX_URL"] == "https://index" and "PIP_INDEX_URL" not in child_environment(base=HOST)
    assert not set(SECRETS) & set(install)


@pytest.mark.parametrize("name", [*SECRETS, "MY_SERVICE_AUTH", "AZURE_CLIENT_SECRET", "GOOGLE_CREDENTIALS",
                                  "GH_TOKEN", "KEY"])
def test_credential_names(name) -> None:
    assert is_secret_name(name)


@pytest.mark.parametrize("name", ["PATH", "KEYBOARD_LAYOUT", "MONKEY_MODE", "HF_HUB_OFFLINE",
                                  "CUDA_PATH_V12_4", "NO_PROXY", "MKL_NUM_THREADS"])
def test_ordinary_names(name) -> None:
    assert not is_secret_name(name)


def test_passed_names_follow_the_settings(tmp_path) -> None:
    assert passed_names(AppConfig()) == ()
    assert passed_names(AppConfig(pass_environment=["WANDB_API_KEY"])) == ("WANDB_API_KEY",)
    server = AppConfig(mlflow_tracking_uri="https://mlflow.example.org")
    assert set(MLFLOW_AUTH) <= set(passed_names(server))  # MLflow's login reaches workers tracking there
    assert not passed_names(server.model_copy(update={"mlflow_tracking": False}))
    with pytest.raises(ValueError):
        AppConfig(pass_environment=["WANDB_API_KEY=secret"])  # names only, never a value
    paths = AppPaths.resolve(tmp_path).ensure()
    save_config(AppConfig(pass_environment=["WANDB_API_KEY"]), paths)  # a name in a list: allowed
    assert load_config(paths).pass_environment == ["WANDB_API_KEY"]


def test_the_persons_own_programs_keep_their_environment() -> None:
    env = user_environment(HOST)
    assert env["HF_TOKEN"] == "hf_x" and env["RANDOM_APP_SETTING"] == "r"
    assert not {"PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV", "CONDA_PREFIX"} & set(env)


def test_a_real_child_process_sees_the_isolated_environment(tmp_path) -> None:
    import os

    leaked = tmp_path / "leaked-host-pythonpath"  # a unique folder: CI's own paths contain "host"
    leaked.mkdir()
    base = {**os.environ, "TEDO_TEST_API_TOKEN": "must-not-leak", "PYTHONPATH": str(leaked)}
    code = ("import json, os, sys, core; print(json.dumps({'token': os.environ.get('TEDO_TEST_API_TOKEN'), "
            "'prefix': sys.pycache_prefix, "
            "'host': any('leaked-host-pythonpath' in p for p in sys.path), 'core': core.__file__}))")
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60,
                          stdin=subprocess.DEVNULL, cwd=tmp_path,
                          env=child_environment(base=base, code_root=CODE_ROOT, workspace=tmp_path))
    assert done.returncode == 0, done.stderr
    seen = json.loads(done.stdout)
    assert seen["token"] is None and not seen["host"]
    assert seen["prefix"] and Path(seen["core"]).is_relative_to(CODE_ROOT)
