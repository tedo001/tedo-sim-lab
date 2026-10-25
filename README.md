# TEDO AI Research Lab

A desktop workbench for machine-learning and computer-vision research, built with
Qt for Python (PySide6). Pick a dataset, a model, hyperparameters and a device; run the experiment
in the background; track it in MLflow and SQLite; compare runs; reproduce any run
from its saved folder.

**V0.1, and since then:** a Windows installer, a Simulation Lab for quantum physics and computer
vision (including electrons in a crystal lattice, solved self-consistently and drawn in 3-D), 3-D
views of datasets and clusters, real terminals, Git and GitHub, a debugger in the Code page, a
Logs page and settings you edit in the app. Every page below works end to end; pages for later releases (NLP, audio,
multimodal, quantum ML) say which release brings them. See [ROADMAP.md](ROADMAP.md) for what comes
next, [docs/guide.md](docs/guide.md) for a tour of every page, and [CLAUDE.md](CLAUDE.md) for the
architecture and the project's rules.

**Next: a cleaner interface.** [design/ui-redesign.md](design/ui-redesign.md) proposes one system flow
(project → data → design → run → analyze → ship), a sidebar of 15 rows instead of 28, a run page
that gathers Training, Evaluation, Benchmarking and MLflow as tabs, a five-step New experiment with
preflight fixes, and contrast fixes to four colour tokens. Nothing in the app has changed yet.

![Home](docs/images/home.png)

## What it does

| Area | Pages | What you get |
| --- | --- | --- |
| Research labs | Computer Vision, CNN Explainer, Classical ML, Deep Learning, Simulation Lab | Presets for MNIST, Fashion-MNIST, CIFAR-10/100 with SimpleCNN, LeNet-5, ResNet-18 and TinyVGG; a layer-by-layer explainer of a trained TinyVGG; eleven scikit-learn/XGBoost models on built-in tables or your CSV (CV, search, ROC/PR, permutation importance, SHAP); a layer-stack network designer with live shapes and generated PyTorch code; quantum circuits you build by clicking (exact states, shots, Bloch spheres, OpenQASM), wave packets that tunnel and oscillate, electrons in a crystal lattice (a Kohn–Sham density-functional calculation: the lattice potential, the electrons moving on it and the density they average to, in 3-D, with V_eff = V_H + V_ion + V_xc taken apart), an image-processing pipeline, synthetic COCO datasets and a camera model, each saved and exported as a script |
| Data & models | Dataset Hub, Model Zoo, Model Registry | 23 datasets and 28 models with licences checked at their sources and downloads only where the terms allow; any table or MNIST-style image set explored in 3-D (three columns or the first three principal components, coloured by class); the ImageNet-1k classes in WordNet (names only); a COCO annotation inspector; versions of your trained models, mirrored into MLflow's registry |
| Experimentation | Experiment Builder, Training, Evaluation, Benchmarking, Compare, MLflow | An experiment is an `experiment.yaml`; it runs in a background worker (cancel, resume, early stopping, AMP, checkpoints), is tracked in SQLite and MLflow, snapshotted for reproduction, re-scored on any split, timed by batch size, compared eight at a time and exported as CSV, JSON, Markdown or PDF |
| Tools | Code (an IDE for a project folder), Terminal, Git & GitHub, Jupyter Notebook, Google Colab, Plugin Store, Hardware Monitor, Logs, Documentation | Edit, run and debug a whole project (pdb with breakpoints in the gutter; the editor follows); real terminals in tabs: bash, PowerShell, Command Prompt, Python, vim, Neovim, Docker; commits, history, branches, sync and your GitHub repositories; notebooks started from runs and Jupyter Lab on request; Colab notebooks whose results import back as runs; Kaggle, Hugging Face and Roboflow with keyring credentials and licence-first downloads; live CPU/RAM/GPU use; the lab's logs, live and filtered |

| | |
| --- | --- |
| ![Classical ML](docs/images/classical-ml.png) | ![Training](docs/images/training.png) |
| ![Deep Learning](docs/images/deep-learning.png) | ![CNN Explainer](docs/images/cnn-explainer.png) |
| ![Simulation Lab: quantum circuits](docs/images/simulation-lab.png) | ![Simulation Lab: wave mechanics](docs/images/wave-mechanics.png) |
| ![Code: debugging with pdb](docs/images/code.png) | ![Terminal](docs/images/terminal.png) |
| ![Git & GitHub](docs/images/git.png) | ![Logs](docs/images/logs.png) |
| ![Plugin Store](docs/images/plugin-store.png) | ![Split view](docs/images/split-view.png) |
| ![Settings](docs/images/settings.png) | ![Training: clusters in 3-D](docs/images/clusters-3d.png) |
| ![Simulation Lab: electrons in a lattice, in 3-D](docs/images/lattice-3d.png) | ![Dataset Hub: a dataset in 3-D](docs/images/dataset-3d.png) |

The 3-D views are drawn by the lab itself (Qt's painter; no OpenGL and no GPL 3-D module): drag to
turn, the wheel zooms, a double-click resets, and two views side by side turn together. Every page,
as it looks today, is in the [gallery below](#every-page).

Everything long runs outside the window: training, evaluation and benchmarks in a worker process,
downloads and searches on background threads. The UI never waits.

## Install on Windows

Download `TEDO-AI-Lab-<version>-win64.msi` from the repository's Releases, or from the latest
*Windows installer* run in Actions, and double-click it. No administrator rights are needed.
The installer is built only with MIT, Apache-2.0 and PSF licensed tools (cx_Freeze).

After installing, open Settings → Experiment Python → **Make the environment** once. This installs PyTorch for training (the CUDA build with an NVIDIA GPU); everything else works straight away.

To build the installer yourself, see [packaging/INSTALLER.md](packaging/INSTALLER.md).

## Quick start: one file

Install Python 3.11 or newer (https://www.python.org/downloads/), get the code, then run one file:

```bash
git clone https://github.com/tedo001/tedo-sim-lab.git
cd tedo-sim-lab
python run.py            # Windows: double-click run.bat
```

The first start makes a private environment in `.venv/`, installs PyTorch (the CUDA build when it
finds an NVIDIA GPU, otherwise the CPU build; a few GB) and the lab, then opens the app. Later starts
open it straight away and reinstall only when `pyproject.toml` changes. Started with an older Python,
it looks for Python 3.11+ on the machine and restarts with it; an existing `.venv` made with an older
Python is moved aside (to `.venv-python3.9-old`, say) and replaced. Arguments go to the app
(`python run.py --workspace D:/lab`); `--setup-only`, `--reinstall` and `--cpu` are the launcher's own.

## Install by hand

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

Optional extras: `[cv]` (ONNX, ONNX Runtime, Transformers for RT-DETR), `[ocr]` (PaddleOCR, EasyOCR,
Tesseract), `[nlp]` (Transformers, Hugging Face Hub), `[integrations]` (Jupyter Lab, nbformat,
SHAP). The app starts without any of them; the Jupyter page installs Jupyter Lab when you ask.

On Linux, Qt also needs: `libegl1 libgl1 libxkbcommon0 libfontconfig1 libdbus-1-3`.

## Run

```bash
python run.py                            # the launcher (installs on first use)
python -m app.main                       # or: tedo-lab, inside your own environment
python -m app.main --workspace D:/lab    # keep datasets, runs and logs elsewhere
```

Runtime data (datasets, checkpoints, runs, the database, MLflow's store, logs) lives in
the *workspace*: the repository folder by default, or `--workspace` /
`TEDO_LAB_WORKSPACE`. None of it is committed to git. Settings live in the workspace's
`configs/settings.yaml`; change them on the Settings page (the gear in the title row).

## Credentials

Integrations read credentials from environment variables or the OS keyring — never
from files. Recognised variables: `KAGGLE_USERNAME`, `KAGGLE_KEY`, `ROBOFLOW_API_KEY`,
`HF_TOKEN`, `GITHUB_TOKEN`. Store or remove them in the OS keyring from the Plugin Store
(Account), or connect GitHub on the Git & GitHub page; Settings shows which are set and where
from, never the values; logs and terminals mask them. Git receives a GitHub token only as an HTTP
header in its environment, never on a command line or in a file.

The Kaggle, Hugging Face and Roboflow plugins talk to each service's public REST API with the
lab's own code (no client libraries). A download lands in `datasets/<source>/` or
`models/<source>/` with a `source.json` recording where it came from and its licence.

## Test

```bash
pytest
QT_QPA_PLATFORM=offscreen python -m app.main --smoke-test
```

## Paper

A three-page paper in IEEE conference format describes the lab's design, the Kohn–Sham lattice
model and how both are verified: [docs/paper/paper.pdf](docs/paper/paper.pdf). Its LaTeX source
uses the IEEEtran class and IEEE reference style (`docs/paper/paper.tex`, `references.bib`);
build it with `pdflatex paper && bibtex paper && pdflatex paper && pdflatex paper`, or upload the
folder to Overleaf.

## Every page

Every page and view as it looks today, at the window's default size, with a few real runs in it (a
contact sheet of them all: [docs/images/pages/overview.png](docs/images/pages/overview.png)). The
pages for later releases (NLP / LLM, Audio, Multimodal, Quantum ML) say which release brings them.

| | |
| --- | --- |
| **Home**<br>![Home](docs/images/pages/01-home.png) | **Computer Vision**<br>![Computer Vision](docs/images/pages/02-computer-vision.png) |
| **CNN Explainer**<br>![CNN Explainer](docs/images/pages/03-cnn-explainer.png) | **Classical ML**<br>![Classical ML](docs/images/pages/04-classical-ml.png) |
| **Deep Learning**<br>![Deep Learning](docs/images/pages/05-deep-learning.png) | **NLP / LLM**<br>![NLP / LLM](docs/images/pages/06-nlp-llm.png) |
| **Audio**<br>![Audio](docs/images/pages/07-audio.png) | **Multimodal**<br>![Multimodal](docs/images/pages/08-multimodal.png) |
| **Quantum ML**<br>![Quantum ML](docs/images/pages/09-quantum-ml.png) | **Simulation Lab**<br>![Simulation Lab](docs/images/pages/10-simulation-lab.png) |
| **Dataset Hub**<br>![Dataset Hub](docs/images/pages/11-dataset-hub.png) | **Model Zoo**<br>![Model Zoo](docs/images/pages/12-model-zoo.png) |
| **Model Registry**<br>![Model Registry](docs/images/pages/13-model-registry.png) | **Experiment Builder**<br>![Experiment Builder](docs/images/pages/14-experiment-builder.png) |
| **Training**<br>![Training](docs/images/pages/15-training.png) | **Evaluation**<br>![Evaluation](docs/images/pages/16-evaluation.png) |
| **Benchmarking**<br>![Benchmarking](docs/images/pages/17-benchmarking.png) | **Compare Experiments**<br>![Compare Experiments](docs/images/pages/18-compare.png) |
| **MLflow**<br>![MLflow](docs/images/pages/19-mlflow.png) | **Code**<br>![Code](docs/images/pages/20-code.png) |
| **Terminal**<br>![Terminal](docs/images/pages/21-terminal.png) | **Git & GitHub**<br>![Git & GitHub](docs/images/pages/22-source-control.png) |
| **Jupyter Notebook**<br>![Jupyter Notebook](docs/images/pages/23-jupyter.png) | **Google Colab**<br>![Google Colab](docs/images/pages/24-colab.png) |
| **Plugin Store**<br>![Plugin Store](docs/images/pages/25-plugin-store.png) | **Hardware Monitor**<br>![Hardware Monitor](docs/images/pages/26-hardware.png) |
| **Logs**<br>![Logs](docs/images/pages/27-logs.png) | **Documentation**<br>![Documentation](docs/images/pages/28-documentation.png) |
| **Settings**<br>![Settings](docs/images/pages/29-settings.png) | **Simulation Lab · Circuits**<br>![Simulation Lab · Circuits](docs/images/pages/simulation-circuit.png) |
| **Simulation Lab · Waves**<br>![Simulation Lab · Waves](docs/images/pages/simulation-waves.png) | **Simulation Lab · Lattice**<br>![Simulation Lab · Lattice](docs/images/pages/simulation-lattice.png) |
| **Simulation Lab · Images**<br>![Simulation Lab · Images](docs/images/pages/simulation-image.png) | **Simulation Lab · Scenes**<br>![Simulation Lab · Scenes](docs/images/pages/simulation-scenes.png) |
| **Simulation Lab · Camera**<br>![Simulation Lab · Camera](docs/images/pages/simulation-camera.png) | **Dataset Hub · Catalogue**<br>![Dataset Hub · Catalogue](docs/images/pages/dataset-hub-catalogue.png) |
| **Dataset Hub · Explore in 3-D**<br>![Dataset Hub · Explore in 3-D](docs/images/pages/dataset-hub-explore.png) | **Dataset Hub · Ontology Explorer**<br>![Dataset Hub · Ontology Explorer](docs/images/pages/dataset-hub-ontology.png) |
| **Dataset Hub · COCO inspector**<br>![Dataset Hub · COCO inspector](docs/images/pages/dataset-hub-coco.png) | **Training · clusters in 3-D**<br>![Training · clusters in 3-D](docs/images/pages/training-clusters-3d.png) |
| **Evaluation · digits PCA in 3-D**<br>![Evaluation · digits PCA in 3-D](docs/images/pages/evaluation-pca-3d.png) |  |

## Licences

The lab only uses permissively licensed tools (MIT, Apache-2.0, BSD); copyleft tools such as
Ultralytics YOLO (AGPL-3.0) are excluded. Every dependency's licence is listed in
[configs/dependency_licences.yaml](configs/dependency_licences.yaml) and checked by the tests.
The one exception is the Qt binding, PySide6, which is LGPL-3.0: no permissively licensed
Python binding for Qt exists.

Bundled assets:

- Inter and JetBrains Mono fonts — SIL Open Font License 1.1
  (`app/resources/fonts/OFL-*.txt`)
- Lucide icons — ISC (`app/resources/icons/LICENSE-lucide.txt`)

Ported code:

- The CNN Explainer page is a port of [CNN Explainer](https://github.com/poloclub/cnn-explainer)
  by Zijie J. Wang et al., Polo Club of Data Science, Georgia Tech — MIT
  (`labs/computer_vision/explainer/LICENSE-cnn-explainer.txt`). Its Tiny ImageNet example images
  and pretrained weights are not included: they derive from ImageNet.
- Built-in digit samples come from the UCI Optical Recognition of Handwritten Digits data set
  (CC BY 4.0), read from the copy that ships with scikit-learn.

Physics: the Lattice tool's exchange-correlation uses the exact exchange of the two-dimensional
electron gas and the correlation energy fitted by Attaccalite, Moroni, Gori-Giorgi and Bachelet
(Phys. Rev. Lett. 88, 256601, 2002); the numbers come from the paper, the code is the lab's own.

Datasets: the Classical ML lab reads Iris, Wine, Breast Cancer and Digits (UCI, CC BY 4.0) and
Diabetes (Efron et al. 2004, no licence published) from scikit-learn's own copies; nothing is
downloaded. An imported CSV file keeps the licence category `unspecified`.
