"""PaddleOCR: text detection and recognition with PaddleOCR.

No optional dependency is imported here; actions import theirs when they run.
"""

from __future__ import annotations

from core.plugin_api import ActionSpec, ManifestPlugin

__all__ = ["PaddleOcrPlugin"]


class PaddleOcrPlugin(ManifestPlugin):
    ACTIONS = (
        ActionSpec("recognise", "Recognise text", "Run OCR on images",
                   needs_connection=False),
    )
