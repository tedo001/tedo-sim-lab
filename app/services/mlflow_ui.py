"""The MLflow web UI, started by a person from the MLflow page and stopped with the app.

Runs ``python -m mlflow ui`` against the workspace's tracking store, bound to
127.0.0.1 on the first free port from 5000.
"""

from __future__ import annotations

from core.common import mlflow_tracking_uri

from .web_ui import LocalWebUi, free_port

__all__ = ["MlflowUi", "free_port"]


class MlflowUi(LocalWebUi):
    START_PORT = 5000

    def command(self, port: int) -> list[str]:
        return ["-m", "mlflow", "ui", "--backend-store-uri", mlflow_tracking_uri(self.config, self.paths),
                "--default-artifact-root", self.paths.mlruns.as_uri(), "--host", "127.0.0.1",
                "--port", str(port)]

    def prepare(self) -> None:
        self.paths.mlruns.mkdir(parents=True, exist_ok=True)
