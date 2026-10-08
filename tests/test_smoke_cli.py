"""The headless smoke test, run exactly as CI and the brief run it."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from app.navigation import NAV

ROOT = Path(__file__).resolve().parents[1]


def run_app(*args: str, tmp_path: Path) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ, QT_QPA_PLATFORM="offscreen", TMPDIR=str(tmp_path), TEMP=str(tmp_path),
               TMP=str(tmp_path))
    return subprocess.run([sys.executable, "-m", "app.main", *args], cwd=ROOT, env=env,
                          capture_output=True, text=True, timeout=180)


def test_smoke_test_visits_every_page(tmp_path: Path) -> None:
    shots = tmp_path / "shots"
    result = run_app("--smoke-test", "--screenshots", str(shots), tmp_path=tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    assert f"SMOKE OK - {len(NAV)} pages" in result.stdout
    assert len(list(shots.glob("*.png"))) == len(NAV) + 1  # + the split-view layout
    assert not list(tmp_path.glob("tedo-smoke-*")), "temporary workspace was not removed"


def test_bad_settings_stop_startup_with_a_message(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    (workspace / "configs").mkdir(parents=True)
    (workspace / "configs" / "settings.yaml").write_text("hf_token: hf_should_not_be_here\n")
    result = run_app("--smoke-test", "--workspace", str(workspace), tmp_path=tmp_path)
    assert result.returncode == 2
    assert "must not contain credentials" in result.stderr
    assert "hf_should_not_be_here" not in result.stderr
