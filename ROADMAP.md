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

- **Windows installer.** Built (`packaging/`, see [packaging/INSTALLER.md](packaging/INSTALLER.md)).
  - A one-folder PyInstaller build of the window, wrapped in an Inno Setup wizard.
  - The app makes its own experiment Python on request.
  - Next: code signing, so Windows SmartScreen does not warn, and an in-app update check.
- **Projects.** The Code page already opens a folder as a project. It has:
  - files, an editor and run;
  - a terminal;
  - git status.

  Still to come: committing from git, a debugger, and the project's experiments and notebooks
  listed beside its files.
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
