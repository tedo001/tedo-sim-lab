# TEDO AI Research Lab

A desktop workbench for machine-learning and computer-vision research, built with
PyQt6. Pick a dataset, a model, hyperparameters and a device; run the experiment
in the background; track it in MLflow and SQLite; compare runs; reproduce any run
from its saved folder.

> **Status: V0.1 in development — build phase 1 of 10.** The window, navigation and
> every page exist; pages that are not built yet say which phase or release
> delivers them. See [CLAUDE.md](CLAUDE.md) for the architecture and build status.

## Install

Python 3.11 or newer.

```bash
git clone https://github.com/tedo001/tedo-sim-lab.git
cd tedo-sim-lab
python -m venv .venv
# Windows: .venv\Scripts\activate    Linux/macOS: source .venv/bin/activate

# PyTorch first. With an NVIDIA GPU, use the CUDA command from https://pytorch.org;
# without one, the CPU build is much smaller:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

pip install -e ".[dev]"
```

Optional extras: `[cv]` (Ultralytics — AGPL-3.0, ONNX), `[ocr]` (PaddleOCR, EasyOCR,
Tesseract), `[nlp]` (Transformers, Hugging Face Hub), `[integrations]` (Kaggle,
Roboflow, Jupyter, SHAP). The app starts without any of them.

On Linux, Qt also needs: `libegl1 libgl1 libxkbcommon0 libfontconfig1 libdbus-1-3`.

## Run

```bash
python -m app.main                       # or: tedo-lab
python -m app.main --workspace D:/lab    # keep datasets, runs and logs elsewhere
```

Runtime data (datasets, checkpoints, runs, the database, MLflow's store, logs) lives in
the *workspace*: the repository folder by default, or `--workspace` /
`TEDO_LAB_WORKSPACE`. None of it is committed to git.

## Credentials

Integrations read credentials from environment variables or the OS keyring — never
from files. Recognised variables: `KAGGLE_USERNAME`, `KAGGLE_KEY`, `ROBOFLOW_API_KEY`,
`HF_TOKEN`, `GITHUB_TOKEN`. Settings shows which are set and where from, never the
values; logs mask them.

## Test

```bash
pytest
QT_QPA_PLATFORM=offscreen python -m app.main --smoke-test
```

## Licences of bundled assets

- Inter and JetBrains Mono fonts — SIL Open Font License 1.1
  (`app/resources/fonts/OFL-*.txt`)
- Lucide icons — ISC (`app/resources/icons/LICENSE-lucide.txt`)
