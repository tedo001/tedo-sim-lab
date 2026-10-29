"""The Plugin Store's Environments card lists the packs and their state; nothing is installed until
the person asks."""

from __future__ import annotations

from app.ui.pages.plugin_envs import EnvironmentsCard
from core.common import packs


def test_environments_card(ctx, qtbot, tmp_path, monkeypatch) -> None:
    monkeypatch.setenv(packs.PACKS_ENV, str(tmp_path / "packs"))
    card = EnvironmentsCard(ctx)
    qtbot.addWidget(card)
    assert card.names == ["language", "explain", "onnx"] and card.table.rowCount() == 3
    assert card.selected == "language" and card.install_button.text() == "Install"
    assert not card.remove_button.isEnabled() and "transformers" in card.table.item(0, 1).text()
    assert "PyTorch is not installed twice" in card.note.text()
    assert card.remove() is False
