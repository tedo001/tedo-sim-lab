"""Headless smoke test: open every page in :data:`app.navigation.NAV`, then exit.

    QT_QPA_PLATFORM=offscreen python -m app.main --smoke-test [--screenshots DIR]

Fails (exit code 1) if any page raises while being built or shown, including
errors raised later inside slots. ``--screenshots`` saves one PNG per page.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from PyQt6.QtWidgets import QApplication

from .main_window import MainWindow
from .navigation import NAV

__all__ = ["SmokeReport", "run_smoke"]


@dataclass
class SmokeReport:
    visited: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    screenshots: list[Path] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors and len(self.visited) == len(NAV)

    def summary(self) -> str:
        if self.ok:
            return f"SMOKE OK - {len(self.visited)} pages: {', '.join(self.visited)}"
        return (f"SMOKE FAILED - {len(self.visited)}/{len(NAV)} pages, "
                f"{len(self.errors)} error(s):\n" + "\n".join(self.errors))


def run_smoke(window: MainWindow, app: QApplication, *, uncaught: list[str],
              screenshot_dir: Path | None = None) -> SmokeReport:
    """Visit every page; ``uncaught`` is the list the excepthook appends to."""
    report = SmokeReport()
    window.show()
    app.processEvents()
    if screenshot_dir is not None:
        screenshot_dir.mkdir(parents=True, exist_ok=True)
    for spec in NAV:
        try:
            window.navigate(spec.id)
            app.processEvents()
            if window.current_page_id() != spec.id:
                raise RuntimeError(f"navigate({spec.id!r}) left the window on "
                                   f"{window.current_page_id()!r}")
        except Exception as exc:  # report every page, not just the first failure
            report.errors.append(f"{spec.id}: {type(exc).__name__}: {exc}")
            continue
        report.visited.append(spec.id)
        if screenshot_dir is not None:
            file = screenshot_dir / f"{len(report.visited):02d}-{spec.id}.png"
            window.grab().save(str(file))
            report.screenshots.append(file)
    report.errors.extend(uncaught)
    window.close()
    return report
