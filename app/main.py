"""Start TEDO AI Research Lab.

    python -m app.main                       # the desktop app
    python -m app.main --workspace D:/lab    # keep runtime data somewhere else
    QT_QPA_PLATFORM=offscreen python -m app.main --smoke-test [--screenshots DIR]
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path

from core.common import AppPaths, ConfigError, CredentialStore, load_config, mask_text, setup_logging

__all__ = ["main", "parse_args"]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="tedo-lab", description="TEDO AI Research Lab")
    parser.add_argument("--workspace", type=Path, default=None,
                        help="folder for datasets, runs, database and logs "
                             "(default: TEDO_LAB_WORKSPACE, else the source checkout)")
    parser.add_argument("--smoke-test", action="store_true",
                        help="open every page in a temporary workspace, then exit 0/1")
    parser.add_argument("--screenshots", type=Path, default=None, metavar="DIR",
                        help="with --smoke-test: save a PNG of every page to DIR")
    parser.add_argument("--log-level", default=None,
                        choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    workspace = args.workspace
    if args.smoke_test and workspace is None:
        workspace = Path(tempfile.mkdtemp(prefix="tedo-smoke-"))
    paths = AppPaths.resolve(workspace).ensure()
    try:
        config = load_config(paths)
    except ConfigError as exc:
        print(f"tedo-lab: {mask_text(str(exc))}", file=sys.stderr)
        return 2
    log = setup_logging(paths.logs, args.log_level or config.log_level)
    log.info("Starting with workspace %s", paths.workspace)

    from PyQt6.QtWidgets import QApplication

    from .errors import install_excepthook, install_qt_message_handler
    from .main_window import MainWindow
    from .services.context import AppContext
    from .ui.theme import apply_theme

    uncaught: list[str] = []
    install_excepthook(show_dialog=not args.smoke_test, collect=uncaught)
    install_qt_message_handler()
    app = QApplication.instance() or QApplication(sys.argv[:1])
    app.setApplicationName("TEDO AI Research Lab")
    app.setOrganizationName("TEDO")
    apply_theme(app)
    window = MainWindow(AppContext(paths, config, CredentialStore()))

    if args.smoke_test:
        from .smoke import run_smoke
        report = run_smoke(window, app, uncaught=uncaught, screenshot_dir=args.screenshots)
        print(report.summary())
        if report.ok and args.workspace is None:  # keep the workspace (and its log) on failure
            setup_logging(None, console=True)       # release the log file (Windows locks it)
            shutil.rmtree(paths.workspace, ignore_errors=True)
        return 0 if report.ok else 1

    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
