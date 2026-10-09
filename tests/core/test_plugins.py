"""Plugin discovery, lazy loading, honest status and gated actions."""

from __future__ import annotations

import sys
from pathlib import Path
from textwrap import dedent

import pytest

from core.common import CredentialStore
from core.common.paths import CODE_ROOT
from core.plugin_api import ManifestPlugin, PluginError, PluginRegistry
from core.plugin_api import plugin as plugin_module

EXPECTED = {"kaggle", "roboflow", "huggingface", "colab", "mlflow", "paddleocr", "rtdetr", "custom"}


@pytest.fixture
def registry() -> PluginRegistry:
    registry = PluginRegistry(CredentialStore(env={}, backend=None))
    assert registry.discover(CODE_ROOT / "plugins") == []
    return registry


def test_all_plugins_are_discovered(registry: PluginRegistry) -> None:
    assert {manifest.name for manifest in registry.manifests()} == EXPECTED


def test_discovery_imports_no_plugin_code() -> None:
    for name in [m for m in sys.modules if m.startswith("plugins.")]:
        del sys.modules[name]
    registry = PluginRegistry(CredentialStore(env={}, backend=None))
    registry.discover(CODE_ROOT / "plugins")
    assert not [m for m in sys.modules if m.startswith("plugins.")]
    registry.get("kaggle")
    assert "plugins.kaggle.plugin" in sys.modules


def test_manifests_carry_licence_author_and_capabilities(registry: PluginRegistry) -> None:
    for manifest in registry.manifests():
        assert manifest.license.name and manifest.capabilities, manifest.name
        assert manifest.author == "tedo001 <durgamani.d.e.c.e.50@gmail.com>"
    assert "yolo" not in registry  # AGPL: excluded by licence policy
    assert not any(manifest.license.copyleft for manifest in registry.manifests())
    assert {m.name for m in registry.with_capability("dataset_source")} >= {"kaggle", "huggingface"}


def test_unbuilt_plugins_say_planned_and_refuse_to_run(registry: PluginRegistry) -> None:
    ocr = registry.get("paddleocr")
    assert ocr.status == "planned"
    actions = ocr.actions()
    assert actions and not any(action.enabled for action in actions)
    assert all("v0.5" in action.disabled_reason for action in actions)
    with pytest.raises(NotImplementedError, match="v0.5"):
        ocr.run("recognise")
    with pytest.raises(PluginError, match="no action"):
        ocr.run("delete_everything")


def test_sources_need_their_credentials_before_acting(registry: PluginRegistry) -> None:
    kaggle = registry.get("kaggle")
    assert kaggle.status == "not_connected"
    assert all(not action.enabled and "KAGGLE_KEY" in action.disabled_reason for action in kaggle.actions())
    with pytest.raises(NotImplementedError, match="Not connected"):
        kaggle.run("search", query="mnist")


def test_connection_reports_missing_credentials(registry: PluginRegistry) -> None:
    result = registry.get("kaggle").test_connection()
    assert not result.ok and "KAGGLE_USERNAME" in result.message
    assert registry.get("colab").test_connection().ok  # no account needed


def write_plugin(root: Path, name: str, *, maturity: str = "stable", requires: str = "[]",
                 credentials: str = "", licence: str = "{category: open-source, name: MIT, spdx: MIT}"
                 ) -> None:
    folder = root / name
    folder.mkdir(parents=True)
    (folder / "plugin.yaml").write_text(dedent(f"""\
        name: {name}
        title: {name.title()}
        version: 1.0.0
        author: test
        description: test plugin
        license: {licence}
        capabilities: [custom]
        entry: core.plugin_api.plugin:ManifestPlugin
        requires: {requires}
        maturity: {maturity}
        """) + credentials)


def test_status_follows_requirements_maturity_and_credentials(tmp_path: Path) -> None:
    write_plugin(tmp_path, "needs_pkg", requires='["definitely-not-installed-pkg>=1"]')
    write_plugin(tmp_path, "experimental_one", maturity="experimental")
    write_plugin(tmp_path, "needs_key",
                 credentials="credentials:\n  - {key: DEMO_KEY, label: Demo key}\n")
    write_plugin(tmp_path, "ready_one", requires='["pyyaml>=5"]')
    store = CredentialStore(env={}, backend=None)
    registry = PluginRegistry(store)
    assert registry.discover(tmp_path) == []
    assert registry.status("needs_pkg") == "not_installed"
    assert registry.status("experimental_one") == "experimental"
    assert registry.status("needs_key") == "not_connected"
    assert registry.status("ready_one") == "available"
    connected = PluginRegistry(CredentialStore(env={"DEMO_KEY": "value-1234"}, backend=None))
    connected.discover(tmp_path)
    assert connected.status("needs_key") == "available"


def test_install_plan(tmp_path: Path) -> None:
    write_plugin(tmp_path, "pkg_one", requires='["definitely-not-installed-pkg>=1"]')
    registry = PluginRegistry(CredentialStore(env={}, backend=None), python="/py/bin/python")
    registry.discover(tmp_path)
    plan = registry.get("pkg_one").install_plan()
    assert plan.needed and plan.missing == ("definitely-not-installed-pkg>=1",)
    assert plan.command[:4] == ("/py/bin/python", "-m", "pip", "install")
    assert plan.acknowledgement is None  # permissive: nothing to accept


@pytest.mark.parametrize("licence", [
    "{category: open-source, name: AGPL-3.0, spdx: AGPL-3.0-only, copyleft: true}",
    "{category: open-source, name: GPL-3.0, spdx: GPL-3.0-only}",
    "{category: proprietary, name: Some service terms}",
])
def test_non_permissive_plugins_are_refused(tmp_path: Path, licence: str) -> None:
    write_plugin(tmp_path, "bad_one", licence=licence)
    registry = PluginRegistry(CredentialStore(env={}, backend=None))
    errors = registry.discover(tmp_path)
    assert "bad_one" not in registry and "not allowed" in errors[0].message


def test_install_runs_pip_and_masks_failures(tmp_path: Path, monkeypatch) -> None:
    write_plugin(tmp_path, "pkg", requires='["definitely-not-installed-pkg>=1"]')
    registry = PluginRegistry(CredentialStore(env={}, backend=None))
    registry.discover(tmp_path)
    calls = []

    class Failed:
        returncode, stdout, stderr = 1, "", "error: token hf_abcdefghijklmnopqrstuvwxyz12 rejected"

    monkeypatch.setattr(plugin_module.subprocess, "run", lambda cmd, **kw: calls.append(cmd) or Failed)
    with pytest.raises(PluginError) as error:
        registry.get("pkg").install()
    assert calls and calls[0][-1] == "definitely-not-installed-pkg>=1"
    assert "hf_abc" not in str(error.value)


def test_plugin_settings_reject_credentials(registry: PluginRegistry) -> None:
    plugin = registry.get("huggingface")
    assert isinstance(plugin, ManifestPlugin)
    plugin.configure({"cache_dir": "/tmp/hf"})
    with pytest.raises(PluginError, match="credential"):
        plugin.configure({"hf_token": "abc"})


@pytest.mark.parametrize("text, message", [
    ("name: Bad Name\n", "String should match"),
    ("name: [unclosed\n", "unreadable"),
    ("- a list\n", "unreadable"),
])
def test_broken_manifests_are_reported(tmp_path: Path, text: str, message: str) -> None:
    (tmp_path / "broken").mkdir()
    (tmp_path / "broken" / "plugin.yaml").write_text(text)
    registry = PluginRegistry(CredentialStore(env={}, backend=None))
    errors = registry.discover(tmp_path)
    assert errors and message in errors[0].message
    assert len(registry) == 0


def test_name_must_match_folder(tmp_path: Path) -> None:
    write_plugin(tmp_path, "real_name")
    (tmp_path / "real_name").rename(tmp_path / "other_folder")
    errors = PluginRegistry(CredentialStore(env={}, backend=None)).discover(tmp_path)
    assert "must match its folder" in errors[0].message
