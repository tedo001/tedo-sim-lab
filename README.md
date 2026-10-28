# TEDO AI Research Lab

A desktop workbench for machine-learning and computer-vision research, built with
Qt for Python (PySide6). Pick a dataset, a model, hyperparameters and a device; run the experiment
in the background; track it in MLflow and SQLite; compare runs; reproduce any run
from its saved folder.

**V0.1, and since then:** a Windows installer, a Simulation Lab for quantum physics and computer
vision (including electrons in a crystal lattice, solved self-consistently and drawn in 3-D), 3-D
views of datasets and clusters, real terminals, Git and GitHub, a debugger in the Code page, a
Logs page and settings you edit in the app; and now every research lab: **NLP / LLM** (text
classifiers, a local chat model, tokenizers, embeddings), **Audio** (spectrograms, filters, pitch,
speech recognition, spoken-digit models), **Multimodal** (CLIP zero-shot labels and image search,
BLIP captions), **Quantum ML** (variational circuits and quantum kernels, simulated exactly) and
**Model Testing**, where any trained or pretrained model is used: one input, what-if sweeps, batch
tests, object detection, and a look inside every layer of a trained network. Every page works end
to end. See [ROADMAP.md](ROADMAP.md) for what comes next and [CLAUDE.md](CLAUDE.md) for the
architecture and the project's rules.

## What it does

| Area | Pages | What you get |
| --- | --- | --- |
| Research labs | Computer Vision, CNN Explainer, Classical ML, Deep Learning, NLP / LLM, Audio, Multimodal, Quantum ML, Simulation Lab | Text classification with TF-IDF models (seconds on a CPU, the telling words listed) or a fine-tuned DistilBERT/BERT/MiniLM; chat with SmolLM2 or Qwen2.5 on your own computer, streamed token by token; tokenizers and embeddings (similarity, a 3-D map of texts); sounds as waveforms, spectrograms, mel spectrograms and MFCCs, Butterworth filters, pitch, Whisper or wav2vec 2.0 transcription, spoken digits trained as spectrogram images; CLIP zero-shot labels and folder search by sentence (with a 3-D map), BLIP captions; variational quantum classifiers (angle or ZZ encoding, parameter-shift gradients) and quantum-kernel SVMs with their decision boundaries, circuits and kernel matrices; presets for MNIST, Fashion-MNIST, CIFAR-10/100 with SimpleCNN, LeNet-5, ResNet-18 and TinyVGG; a layer-by-layer explainer of a trained TinyVGG; eleven scikit-learn/XGBoost models on built-in tables or your CSV (CV, search, ROC/PR, permutation importance, SHAP); a layer-stack network designer with live shapes and generated PyTorch code; quantum circuits you build by clicking (exact states, shots, Bloch spheres, OpenQASM), wave packets that tunnel and oscillate, electrons in a crystal lattice (a Kohn–Sham density-functional calculation: the lattice potential, the electrons moving on it and the density they average to, in 3-D, with V_eff = V_H + V_ion + V_xc taken apart), an image-processing pipeline, synthetic COCO datasets and a camera model, each saved and exported as a script |
| Data & models | Dataset Hub, Model Zoo, Model Registry | 39 datasets of every kind (images, audio, text, tables, synthetic sets, image–text pairs) and 40 models with licences checked at their sources and downloads only where the terms allow; a downloaded dataset shows examples, opens in its lab and starts an experiment in the builder; pretrained models download from the Model Zoo or their lab (licence first, gated ones refused); any table, MNIST-style image set or spoken-digit set explored in 3-D (three columns or the first three principal components, coloured by class); the ImageNet-1k classes in WordNet (names only); a COCO annotation inspector; versions of your trained models, mirrored into MLflow's registry |
| Experimentation | Experiment Builder, Training, Evaluation, Model Testing, Benchmarking, Compare, MLflow | Image, audio, text, tabular and quantum tasks in one builder; Model Testing uses any finished run or downloaded model (an image or drawing, a WAV file, a text, a row of values; what-if sweeps; batches from a folder or CSV, scored and exported; detection, captions, transcripts) and shows inside a trained network layer by layer with Grad-CAM and saliency. An experiment is an `experiment.yaml`; it runs in a background worker (cancel, resume, early stopping, AMP, checkpoints), is tracked in SQLite and MLflow, snapshotted for reproduction, re-scored on any split, timed by batch size, compared eight at a time and exported as CSV, JSON, Markdown or PDF |
| Tools | Code (an IDE for a project folder), Terminal, Git & GitHub, Jupyter Notebook, Google Colab, Plugin Store, Hardware Monitor, Logs, Documentation | Edit, run and debug a whole project (pdb with breakpoints in the gutter; the editor follows); real terminals in tabs: bash, PowerShell, Command Prompt, Python, vim, Neovim, Docker; commits, history, branches, sync and your GitHub repositories; notebooks started from runs and Jupyter Lab on request; Colab notebooks whose results import back as runs; Kaggle, Hugging Face and Roboflow with keyring credentials and licence-first downloads; live CPU/RAM/GPU use; the lab's logs, live and filtered |

The 3-D views are drawn by the lab itself (Qt's painter; no OpenGL and no GPL 3-D module): drag to
turn, the wheel zooms, a double-click resets, and two views side by side turn together.

Language, speech and vision-language models run in an *inference worker*: one background process
under the experiment Python that loads a model once and keeps it (two at most, so a 4 GB GPU does not
fill up), streams its answers and never touches the window's process.

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

Transformers (Apache-2.0) is installed with the lab for the pretrained models; the installed app keeps
it in the experiment Python. Optional extras: `[cv]` (ONNX, ONNX Runtime), `[ocr]` (PaddleOCR,
EasyOCR, Tesseract), `[nlp]` (Hugging Face Hub client), `[integrations]` (Jupyter Lab, nbformat,
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

## Every page

| Page | What it does |
| --- | --- |
| Home | Live resources, the active job, recent work and the state of this workspace. |
| Computer Vision | Pick a vision task and start from a preset experiment. |
| CNN Explainer | See how a convolutional network turns an image into a prediction, layer by layer. |
| Classical ML | Classical models on tabular data, tracked like neural networks. |
| Deep Learning | Design a network layer by layer and train it like any other model. |
| NLP / LLM | Text classifiers (TF-IDF or a fine-tuned transformer), a local chat model, tokenizers and embeddings. |
| Audio | Sounds as waveforms, spectrograms and MFCCs; filters, pitch, speech recognition and spoken-digit models. |
| Multimodal | CLIP zero-shot labels and image search by sentence, BLIP captions. |
| Quantum ML | Variational quantum classifiers and quantum-kernel SVMs, simulated exactly; boundaries, circuits, kernels. |
| Simulation Lab | Quantum circuits and wave mechanics; image processing, synthetic datasets and a camera model. |
| Dataset Hub | Find datasets, check their licences, download the ones that allow it; explore ImageNet classes in WordNet; inspect COCO annotations. |
| Model Zoo | Models with their licence, weights' terms, size and hardware needs; copyleft ones are kept out. |
| Model Registry | Models trained in this lab and the runs they came from. |
| Experiment Builder | Describe an experiment as a reproducible experiment.yaml and queue it. |
| Training | Queued, running and finished runs, live. |
| Evaluation | Scores, confusion matrix and per-class report; re-score a checkpoint on a split. |
| Model Testing | Use trained and pretrained models: try an input, what-if sweeps, batch tests, detection, captions, speech, and a look inside every layer. |
| Benchmarking | Inference latency and throughput by batch size and device. |
| Compare Experiments | Runs side by side, curves overlaid, exported as CSV, JSON, Markdown or PDF. |
| MLflow | Runs tracked in MLflow, and the MLflow UI. |
| Code | A project folder as an IDE: files, editor, run, terminal and git. |
| Terminal | Real terminals: bash, PowerShell, Command Prompt, Python, vim, Neovim and Docker. |
| Git & GitHub | Changes, commits, history, branches and sync for a project; your GitHub repositories. |
| Jupyter Notebook | Notebooks in the workspace, started from runs and linked to them; Jupyter Lab on request. |
| Google Colab | Write an experiment as a Colab notebook; import its results as a run. |
| Plugin Store | Integrations: licence, status, install, account; browse Kaggle, Hugging Face and Roboflow. |
| Hardware Monitor | CPU, memory and GPU: what this machine has and how busy it is. |
| Logs | What the lab records, live: the app log, warnings and errors, each run's log. |
| Documentation | The lab's own documentation. |

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
downloaded. An imported CSV file keeps the licence category `unspecified`. Downloads happen only
when you ask, each with its licence shown: Palmer Penguins (CC0), the Free Spoken Digit Dataset
(CC BY-SA 4.0), Movie Review Data and 20 Newsgroups (no licence published), the SMS Spam Collection
(CC BY 4.0). The synthetic sets (two moons, circles, XOR) are made on your computer.

Pretrained models (downloaded on request into `models/huggingface/`): SmolLM2, Qwen2.5,
all-MiniLM-L6-v2, DistilBERT, BERT, Whisper, wav2vec 2.0, RT-DETR and YOLOS (Apache-2.0), CLIP
(MIT) and BLIP (BSD-3-Clause). Detectors and BLIP were trained on COCO, whose images keep their own
Flickr licences.
