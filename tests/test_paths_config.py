"""Workspace resolution and the settings file."""

from __future__ import annotations

from pathlib import Path

import pytest

from core.common import AppConfig, AppPaths, ConfigError, load_config, mlflow_tracking_uri, save_config
from core.common.paths import CODE_ROOT, WORKSPACE_DIRS, WORKSPACE_ENV


def test_explicit_workspace_wins(tmp_path: Path) -> None:
    paths = AppPaths.resolve(tmp_path / "a", env={WORKSPACE_ENV: str(tmp_path / "b")})
    assert paths.workspace == (tmp_path / "a").resolve()


def test_environment_variable_next(tmp_path: Path) -> None:
    paths = AppPaths.resolve(env={WORKSPACE_ENV: str(tmp_path / "b")})
    assert paths.workspace == (tmp_path / "b").resolve()


def test_source_checkout_is_the_default() -> None:
    assert AppPaths.resolve(env={}).workspace == CODE_ROOT


def test_installed_copy_falls_back_to_home(tmp_path: Path) -> None:
    paths = AppPaths.resolve(env={}, code_root=tmp_path / "site-packages")
    assert paths.workspace == (Path.home() / "TEDO-AI-Lab").resolve()


def test_ensure_creates_every_folder(tmp_path: Path) -> None:
    paths = AppPaths.resolve(tmp_path / "ws").ensure()
    for name in WORKSPACE_DIRS:
        assert (paths.workspace / name).is_dir()
    assert paths.lab_db.parent == paths.database


def test_defaults_without_a_settings_file(workspace: AppPaths) -> None:
    config = load_config(workspace)
    assert config == AppConfig()
    assert config.max_concurrent_runs == 1


def test_settings_round_trip(workspace: AppPaths) -> None:
    save_config(AppConfig(log_level="DEBUG", max_concurrent_runs=2), workspace)
    assert "terminal_shell" not in workspace.settings_file.read_text()  # defaults are not written
    loaded = load_config(workspace)
    assert loaded.log_level == "DEBUG"
    assert loaded.max_concurrent_runs == 2


@pytest.mark.parametrize("key", ["hf_token", "KAGGLE_KEY", "github_password", "api_secret"])
def test_credentials_are_rejected(workspace: AppPaths, key: str) -> None:
    workspace.settings_file.write_text(f"{key}: abc123\n")
    with pytest.raises(ConfigError, match="must not contain credentials"):
        load_config(workspace)


def test_unknown_settings_are_rejected(workspace: AppPaths) -> None:
    workspace.settings_file.write_text("colour_scheme: neon\n")
    with pytest.raises(ConfigError, match="colour_scheme"):
        load_config(workspace)


def test_invalid_values_are_rejected(workspace: AppPaths) -> None:
    workspace.settings_file.write_text("max_concurrent_runs: 0\n")
    with pytest.raises(ConfigError, match="max_concurrent_runs"):
        load_config(workspace)


def test_broken_yaml_is_reported(workspace: AppPaths) -> None:
    workspace.settings_file.write_text("log_level: [unclosed\n")
    with pytest.raises(ConfigError, match="not valid YAML"):
        load_config(workspace)


def test_mlflow_defaults_to_workspace_sqlite(workspace: AppPaths) -> None:
    uri = mlflow_tracking_uri(AppConfig(), workspace)
    assert uri.startswith("sqlite:///") and uri.endswith("database/mlflow.db")
    assert mlflow_tracking_uri(AppConfig(mlflow_tracking_uri="http://localhost:5000"),
                               workspace) == "http://localhost:5000"
