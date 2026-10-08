"""The download policy: what the lab may fetch, and on what terms."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from core.common.licensing import LicenseCategory, LicenseInfo, download_policy


@dataclass
class Item:
    license: LicenseInfo
    auto_download_allowed: bool = True
    access_requirements: str = ""


def licence(category: str, **extra) -> LicenseInfo:
    return LicenseInfo(category=LicenseCategory(category), name=extra.pop("name", "X"), **extra)


@pytest.mark.parametrize("category", ["gated", "proprietary"])
def test_gated_and_proprietary_are_never_downloaded(category: str) -> None:
    decision = download_policy(Item(licence(category)))
    assert not decision.allowed
    assert "will not download" in decision.reason


def test_access_requirements_block_download() -> None:
    decision = download_policy(Item(licence("open-source"), access_requirements="Sign the form"))
    assert not decision.allowed
    assert "Sign the form" in decision.reason


def test_cards_that_do_not_allow_download() -> None:
    assert not download_policy(Item(licence("open-source"), auto_download_allowed=False)).allowed


def test_open_source_needs_no_acknowledgement() -> None:
    decision = download_policy(Item(licence("open-source", name="MIT")))
    assert decision.allowed and decision.acknowledgement is None


@pytest.mark.parametrize("category", ["research-only", "non-commercial", "commercial", "unspecified"])
def test_restricted_terms_must_be_acknowledged(category: str) -> None:
    decision = download_policy(Item(licence(category, notes="Cite the paper.")))
    assert decision.allowed
    assert decision.acknowledgement and "Cite the paper." in decision.acknowledgement


def test_copyleft_is_spelled_out() -> None:
    decision = download_policy(Item(licence("open-source", name="AGPL-3.0", copyleft=True)))
    assert decision.allowed
    assert "copyleft" in decision.acknowledgement and "AGPL-3.0" in decision.acknowledgement


def test_unspecified_is_not_called_open() -> None:
    info = licence("unspecified", name="No licence published")
    assert info.label == "No licence published"
    assert "open" not in download_policy(Item(info)).reason.lower()
