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
- **3-D:** electrons in a crystal lattice (a self-consistent Kohn–Sham calculation, the potential
  and density as surfaces, the electrons moving on them); datasets in the Dataset Hub and clusters
  and projections in the results as point clouds you turn with the mouse.
- **Every research lab** (what V0.2–V0.4 planned, and Quantum ML):
  - *NLP / LLM*: text classification with TF-IDF models or a fine-tuned transformer (DistilBERT,
    BERT, MiniLM), tracked like every run; a local chat model (SmolLM2, Qwen2.5) streamed token by
    token; tokenizers; embeddings with similarity and a 3-D map.
  - *Audio*: waveforms, spectrograms, mel spectrograms and MFCCs; filters, pitch; Whisper and
    wav2vec 2.0 transcription; the Free Spoken Digit Dataset trained as spectrogram images.
  - *Multimodal*: CLIP zero-shot labels and image search by sentence, BLIP captions.
  - *Quantum ML*: variational quantum classifiers and quantum-kernel SVMs, simulated exactly, with
    decision boundaries, circuits and kernel matrices.
- **Model Testing:** any finished run or downloaded pretrained model, used: one input, what-if
  sweeps, batch tests with CSV export, object detection with pretrained RT-DETR / YOLOS, and a
  layer-by-layer look inside trained image and spectrogram networks (feature maps, Grad-CAM,
  saliency). Runs open there from Training, the Model Registry and the labs.
- **Datasets of every kind** in the Dataset Hub (39 cards: images, audio, text, tables, synthetic
  sets, image–text pairs) with examples, the lab for each and a starter experiment in the builder.
- **Research rigour:**
  - reproducibility reports and bundles for every run, reproduced in a clean workspace by
    `core.experiment_engine.reproduce`; the full package footprint (lock file, CUDA toolkit, GPUs);
  - data checks before training (duplicates, leakage between splits, label conflicts, missing
    labels, balance, licence metadata) with a warn / fail rule, and *Validate* in the Dataset Hub;
  - COCO mAP and IoU for detections and masks, and checks of COCO annotation files.
  - seed studies (mean, sd, t-interval, Welch's test, matched-budget checks), saved baselines,
    hyperparameter sweeps with parallel coordinates and parameter importance (*Studies & Sweeps*);
    benchmark methodology (repeats, intervals, timer, synchronisation, hardware) recorded;
  - quantum noise (depolarizing, T1/T2, readout, shots) in Quantum ML and the Simulation Lab, with
    classical baselines on the same split, resource counts, noise comparisons and no advantage claims.
  - explanation records (input, checkpoint hash, target, method, settings, outputs) for every
    Grad-CAM, saliency and layer view, listed and re-verified on Model Testing.
- **Remote compute:** runs dispatched to SSH targets (Settings → Remote targets, builder *Run on*).
- **Model-performance radar** on Compare Experiments (shared 0–1 scores, absolute or zoomed).
- **Dependency isolation:** a lighter core experiment Python; Transformers, SHAP and ONNX as environment
  packs layered over it, installed and removed from the Plugin Store, picked per runner and for the
  inference worker.

## Next

- **Code signing**, so Windows SmartScreen does not warn about the installer (needs a code-signing
  certificate from the owner).
- **Projects:** the project's experiments and notebooks listed beside its files.
- **Lattice:** band structures along a path through the Brillouin zone (k-points), spin
  polarization, and forces on the ions so the lattice can relax.
- **3-D views of trained networks:** the features a trained CNN gives each test image, in 3-D
  (computed by the worker, which has PyTorch).

## Next for the labs

- Audio: recording from a microphone and playback inside the lab (needs Qt Multimedia, which the
  lab does not install yet; today a sound opens in your own player); Speech Commands and ESC-50
  loaders.
- NLP: IMDB and AG News loaders; text datasets from your own CSV files.
- Multimodal: fine-tuning CLIP on your own image folders.
- Quantum ML: more feature maps; idle-qubit noise and crosstalk in the noise model.

## V0.5: detection, segmentation, OCR

- The experimental runners for these tasks are already registered:
  - object detection;
  - segmentation;
  - OCR;
  - pose;
  - tracking.

  They become real runners on COCO-format data, which the Dataset Hub's COCO inspector already reads.
- Training RT-DETR (Hugging Face Transformers, Apache-2.0) on your data; the pretrained one already
  runs on Model Testing. Ultralytics YOLO stays excluded because it is AGPL.
- PaddleOCR (Apache-2.0) as a plugin.

## Later

- **Deep Learning builder:**
  - sequence layers (RNN, LSTM, GRU);
  - autoencoders and Transformers;
  - skip connections (a graph rather than a stack).
- **Benchmarks:**
  - ONNX Runtime;
  - TensorRT, where the licence allows.
- **Simulation Lab:** two-dimensional wave packets,
  scenes with more kinds of objects and occlusion levels.
