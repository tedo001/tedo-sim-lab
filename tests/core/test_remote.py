"""Remote dispatch: targets (no secrets), SSH command lines, and a whole run sent to a target, run
there and brought back (a local-folder transport stands in for the network)."""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

from core.common.paths import CODE_ROOT, AppPaths
from core.experiment_engine.spec import dump_spec, spec_hash
from core.remote.dispatch import _localise, code_digest
from core.remote.targets import Target, TargetError, load_targets, save_targets
from core.remote.transport import SshTransport

pytest.importorskip("sklearn")


def test_targets_hold_no_secrets(tmp_path):
    targets = {"gpu": Target("gpu", host="gpu.example.org", user="me", port=2222, python="/opt/py/bin/python",
                             identity_file="~/.ssh/id_ed25519")}
    save_targets(tmp_path, targets)
    assert load_targets(tmp_path) == targets
    (tmp_path / "remotes.yaml").write_text("targets:\n  bad:\n    host: h\n    password: hunter2\n")
    with pytest.raises(TargetError, match="never holds credentials"):
        load_targets(tmp_path)
    with pytest.raises(TargetError, match="needs a host"):
        Target("nohost")
    with pytest.raises(TargetError):
        Target("bad name!", host="h")


def test_ssh_command_lines():
    ssh = SshTransport(Target("gpu", host="gpu.example.org", user="me", port=2222, identity_file="/k/id"))
    command = ssh.command(["python3", "-c", "print('hi there')"])
    assert command[:3] == ["ssh", "-p", "2222"] and "BatchMode=yes" in command and "-i" in command
    assert command[-2] == "me@gpu.example.org" and command[-1] == "python3 -c 'print('\"'\"'hi there'\"'\"')'"
    assert ssh.path("runs", "r1") == "tedo-lab-remote/runs/r1"  # "~/" is the home folder ssh starts in


def test_paths_in_events_point_back_here(tmp_path):
    run_dir = tmp_path / "local" / "r9"
    payload = {"path": "/srv/lab/runs/r9/results.json", "kind": "results"}
    line = json.dumps({"run_id": "r9", "type": "artifact", "payload": payload})
    assert json.loads(_localise(line, run_dir))["payload"]["path"] == str(run_dir / "results.json")
    assert _localise("not json\n", run_dir) == "not json\n"
    assert len(code_digest()) == 16


def test_a_run_goes_to_the_target_and_comes_back(tmp_path):
    from labs.classical_ml.presets import CLASSICAL_PRESETS

    workspace = AppPaths.resolve(tmp_path / "here").ensure()
    remote = tmp_path / "there"
    save_targets(workspace.configs, {"box": Target("box", python=sys.executable, folder=str(remote),
                                                   transport="local")})
    spec = CLASSICAL_PRESETS[0].make()
    spec = spec.model_copy(update={"runtime": spec.runtime.model_copy(update={"target": "box"})})
    assert spec_hash(spec) == spec_hash(CLASSICAL_PRESETS[0].make())  # where it runs is not what it computes
    run_dir = workspace.experiments / "iris" / "r1"
    dump_spec(spec, run_dir / "experiment.yaml")
    env = {**os.environ, "TEDO_LAB_WORKSPACE": str(workspace.workspace), "TEDO_LAB_MLFLOW": "0",
           "PYTHONPATH": str(CODE_ROOT)}
    done = subprocess.run([sys.executable, "-m", "core.remote.dispatch", str(run_dir), "--target", "box",
                           "--workspace", str(workspace.workspace), "--prepare-data"],
                          capture_output=True, text=True, cwd=CODE_ROOT, env=env, stdin=subprocess.DEVNULL,
                          timeout=600)
    assert done.returncode == 0, done.stderr[-2000:]
    events = [json.loads(line) for line in done.stdout.splitlines()]
    end = next(e for e in events if e["type"] == "run_end")
    assert end["payload"]["status"] == "completed" and end["payload"]["metrics"]["test_acc"] > 0.8
    artifacts = [e["payload"]["path"] for e in events if e["type"] == "artifact"]
    assert artifacts and all(path.startswith(str(run_dir)) for path in artifacts)
    for name in ("results.json", "snapshot.json", "checkpoints", "reproduction"):
        assert (run_dir / name).exists(), name  # brought back
    assert (remote / "runs" / "r1" / "results.json").is_file()  # and it really ran there
    assert (remote / "code" / code_digest() / "core").is_dir()
    assert any("Running on box" in e["payload"].get("line", "") for e in events if e["type"] == "log")
