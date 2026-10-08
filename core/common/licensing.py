"""Licences on datasets, weights, models and plugins, and what they allow the lab to do.

Every card carries a :class:`LicenseInfo`. Nothing is ever called "free"
because it can be downloaded: the category says what its terms allow, and
:func:`download_policy` decides whether the lab may fetch it at all. Downloads
only ever start when a person asks; the policy decides whether the button is
there, and whether they must acknowledge terms first.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

from pydantic import BaseModel, ConfigDict

__all__ = ["CATEGORY_LABELS", "DownloadDecision", "Downloadable", "LicenseCategory", "LicenseInfo",
           "PERMISSIVE_SPDX", "download_policy", "tool_licence_problem"]

#: Owner policy for *tools* (libraries, plugins, model code): permissive licences only.
#: MIT and Apache-2.0, plus BSD, ISC and the PSF licence, which grant the same freedoms.
#: Copyleft (AGPL, GPL) is never allowed. Data and pretrained weights are judged by
#: :func:`download_policy` instead.
PERMISSIVE_SPDX = frozenset({"MIT", "Apache-2.0", "BSD-2-Clause", "BSD-3-Clause", "ISC", "PSF-2.0"})


class LicenseCategory(StrEnum):
    OPEN_SOURCE = "open-source"
    RESEARCH_ONLY = "research-only"
    NON_COMMERCIAL = "non-commercial"
    COMMERCIAL = "commercial"
    GATED = "gated"
    PROPRIETARY = "proprietary"
    #: No licence published by the authors. Not the same as permissive.
    UNSPECIFIED = "unspecified"


CATEGORY_LABELS: dict[LicenseCategory, str] = {
    LicenseCategory.OPEN_SOURCE: "Open source",
    LicenseCategory.RESEARCH_ONLY: "Research only",
    LicenseCategory.NON_COMMERCIAL: "Non-commercial",
    LicenseCategory.COMMERCIAL: "Commercial licence",
    LicenseCategory.GATED: "Gated",
    LicenseCategory.PROPRIETARY: "Proprietary",
    LicenseCategory.UNSPECIFIED: "No licence published",
}


class LicenseInfo(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    category: LicenseCategory
    #: The licence's own name: "MIT", "AGPL-3.0", "ImageNet Terms of Access".
    name: str
    url: str | None = None
    spdx: str | None = None
    #: AGPL/GPL-style terms: using it can oblige you to publish your own source.
    copyleft: bool = False
    #: Shown verbatim on the card.
    notes: str = ""

    @property
    def label(self) -> str:
        return CATEGORY_LABELS[self.category]


class Downloadable(Protocol):
    """Anything :func:`download_policy` can judge: dataset cards, weights."""

    license: LicenseInfo
    auto_download_allowed: bool
    access_requirements: str


@dataclass(frozen=True)
class DownloadDecision:
    allowed: bool
    #: Why not, or what applies. Shown next to (or instead of) the download button.
    reason: str
    #: Text the person must accept before the download starts; ``None`` = nothing to accept.
    acknowledgement: str | None = None


_NEVER = {LicenseCategory.GATED: "Access is gated by the provider",
          LicenseCategory.PROPRIETARY: "Proprietary terms"}
_ACKNOWLEDGE = {LicenseCategory.RESEARCH_ONLY, LicenseCategory.NON_COMMERCIAL,
                LicenseCategory.COMMERCIAL, LicenseCategory.UNSPECIFIED}


def download_policy(item: Downloadable) -> DownloadDecision:
    """May the lab fetch ``item`` when a person clicks Download, and on what terms?"""
    licence = item.license
    if licence.category in _NEVER:
        return DownloadDecision(False, f"{_NEVER[licence.category]}: get it from the source "
                                       "yourself; the lab will not download it.")
    if item.access_requirements:
        return DownloadDecision(False, f"Needs: {item.access_requirements}")
    if not item.auto_download_allowed:
        return DownloadDecision(False, "The lab does not download this; use the source link.")
    same = licence.name.lower() == licence.label.lower()
    terms = licence.name if same else f"{licence.name} ({licence.label.lower()})"
    if licence.copyleft:
        return DownloadDecision(True, f"Copyleft: {terms}",
                                f"I understand {licence.name} is copyleft and may oblige me to "
                                "publish source code of work that uses it.")
    if licence.category in _ACKNOWLEDGE:
        note = f" {licence.notes}" if licence.notes else ""
        return DownloadDecision(True, terms,
                                f"I will use this under its terms: {terms}.{note}")
    return DownloadDecision(True, terms)


def tool_licence_problem(licence: LicenseInfo) -> str | None:
    """Why a third-party tool with ``licence`` is not allowed in the lab, or ``None`` if it is."""
    if licence.copyleft:
        return (f"{licence.name} is copyleft; the lab only uses permissively licensed tools "
                "(MIT, Apache-2.0, BSD)")
    if licence.spdx is None:
        return f"{licence.name} has no SPDX identifier, so its terms cannot be checked"
    options = [part.strip() for part in licence.spdx.split(" OR ")]
    if not any(option in PERMISSIVE_SPDX for option in options):
        return (f"{licence.spdx} is not a permissive licence; the lab only uses MIT, Apache-2.0 "
                "or BSD-licensed tools")
    return None
