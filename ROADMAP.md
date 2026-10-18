# Roadmap

What the lab does today is described in the [README](README.md). This page lists what comes
next, in the order it is planned. Every planned item already appears in the app, marked with
the release that brings it (a sidebar tag, a "Planned" pill, or a disabled control that says why).
The rules in [CLAUDE.md](CLAUDE.md) apply to all of it: permissive licences only, nothing that
blocks the window, nothing that pretends to work.

## V0.1: the workbench (done)

Ten build phases:

- the shell;
- the catalogue and licence policy;
- the hardware monitor;
- image classification with the CNN Explainer;
- MLflow, reproducibility and evaluation;
- the Classical ML lab;
- the Dataset Hub, Model Zoo and Model Registry;
- benchmarking, comparison and reports;
- the Deep Learning builder, Plugin Store, Terminal, Jupyter and Colab;
- search across pages, datasets, models and runs.

## After V0.1 (done)

- **Windows installer** (`packaging/`, see [packaging/INSTALLER.md](packaging/INSTALLER.md)): a
  one-folder cx_Freeze build of the window in a per-user `.msi`, MIT/Apache/PSF build tools only;
  the app makes its own experiment Python on request.
- **Projects.** The Code page opens a folder as a project: files, an editor, Run, Run in terminal,
  a debugger (pdb, breakpoints in the gutter, the editor follows), terminals and git status.
- **Real terminals** in tabs: bash, zsh, fish, PowerShell, Command Prompt, Git Bash, WSL, Python,
  vim, Neovim and Docker, with full-screen programs.
- **Git & GitHub:** commit, history, branches, remotes, sync; your GitHub repositories, clone,
  publish, pull requests and issues.
- **Simulation Lab:** quantum circuits, wave mechanics, an image lab, synthetic COCO datasets and a
  camera model, saved and exported as scripts.
- **Logs page** (live, filtered, with a problem count in the title row), **settings edited in the
  app**, and an **update check** that asks GitHub when you want it to.

## Next

- **Code signing**, so Windows SmartScreen does not warn about the installer (needs a code-signing
  certificate from the owner).
- **Projects:** the project's experiments and notebooks listed beside its files.

## V0.2: audio

- Audio datasets with licences checked at their sources.
- Spectrogram pipelines and their augmentation.
- Speech-recognition and audio-classification models through Hugging Face Transformers (Apache-2.0).
- Model cards for Whisper-style models with their weights' own terms.

## V0.3: NLP and language models

- Text classification and tokenisation experiments.
- Hugging Face models and datasets as runners. Gated models need your token and accepted terms; the lab never downloads them for you.

## V0.4: multimodal

- Zero-shot classification and captioning with CLIP-style models.

## V0.5: detection, segmentation, OCR

- The experimental runners for these tasks are already registered:
  - object detection;
  - segmentation;
  - OCR;
  - pose;
  - tracking.

  They become real runners on COCO-format data, which the Dataset Hub's COCO inspector already reads.
- RT-DETR through Hugging Face Transformers, Apache-2.0. Ultralytics YOLO stays excluded because it is AGPL.
- PaddleOCR (Apache-2.0) as a plugin.

## Later

- **Deep Learning builder:**
  - sequence layers (RNN, LSTM, GRU);
  - autoencoders and Transformers;
  - skip connections (a graph rather than a stack).
- **Benchmarks:**
  - ONNX Runtime;
  - TensorRT, where the licence allows.
- **Simulation Lab:** noise and density matrices for circuits, two-dimensional wave packets,
  scenes with more kinds of objects and occlusion levels.
- **Quantum ML:** its scope is not decided yet.
