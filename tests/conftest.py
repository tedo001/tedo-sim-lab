"""Shared fixtures. Qt runs offscreen so the suite needs no display."""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest  # noqa: E402

from core.common import AppConfig, AppPaths, CredentialStore  # noqa: E402
from core.common.masking import forget_secrets  # noqa: E402


class MemoryKeyring:
    """An in-memory stand-in for an OS keyring backend."""

    def __init__(self) -> None:
        self.store: dict[tuple[str, str], str] = {}

    def get_password(self, service: str, username: str) -> str | None:
        return self.store.get((service, username))

    def set_password(self, service: str, username: str, password: str) -> None:
        self.store[(service, username)] = password

    def delete_password(self, service: str, username: str) -> None:
        self.store.pop((service, username))


@pytest.fixture(autouse=True)
def _clean_secrets():
    forget_secrets()
    yield
    forget_secrets()


@pytest.fixture
def workspace(tmp_path) -> AppPaths:
    return AppPaths.resolve(tmp_path / "workspace").ensure()


@pytest.fixture
def keyring_backend() -> MemoryKeyring:
    return MemoryKeyring()


@pytest.fixture
def ctx(workspace, keyring_backend):
    from app.services.context import AppContext
    return AppContext(workspace, AppConfig(), CredentialStore(env={}, backend=keyring_backend))


@pytest.fixture
def window(qapp, qtbot, ctx):
    from app.main_window import MainWindow
    from app.ui.theme import apply_theme
    apply_theme(qapp)
    main_window = MainWindow(ctx)
    qtbot.addWidget(main_window)
    main_window.show()
    return main_window
