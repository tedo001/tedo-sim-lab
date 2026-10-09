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

## Next: shipping

- **Windows installer.** PyInstaller and Inno Setup, the way `tedo001/sentra` ships:
  - one `.exe` setup, with the app in Program Files and the data in a workspace folder you choose;
  - first-run setup of an "experiment Python": the installed app cannot pip-install into itself, so PyTorch and plugins go there (`python_executable` is already the switch).
- **Projects.** Open a local folder as a project and manage everything in it:
  - files and an editor;
  - the terminal, which is already per workspace;
  - git, experiments and notebooks.

  The workspace layout and relative paths were designed for this.
- **Settings page editing.** Change `settings.yaml` from the app (today it is read-only there).

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
- **Quantum ML and the Simulation Lab:** their scope is not decided yet.
