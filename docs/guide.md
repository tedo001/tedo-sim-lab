# A tour of the lab

A short walk through every page, in sidebar order. Ctrl+K jumps to a page, a dataset, a model
or a run by name. Ctrl+B folds the sidebar. Ctrl+\ opens a second page beside the first.

## Home

Shows:

- live CPU, memory and GPU use;
- the run in progress;
- recent work;
- what is in this workspace.

The meters in the title row stay visible on every page.

## Research labs

**Computer Vision.** Start from a preset:

- MNIST with SimpleCNN or TinyVGG;
- Fashion-MNIST with LeNet-5;
- CIFAR-10 with ResNet-18.

You can also download a dataset. The lab asks before downloading anything whose terms need
your agreement. It never downloads ImageNet, COCO or anything gated.

**CNN Explainer.** Shows what a convolutional network computes, layer by layer:

- every feature map;
- a convolution, ReLU, pooling and the softmax, worked through on real numbers;
- a playground for kernel size, stride and padding.

Feed it the built-in digits, an image file, or one you draw. A TinyVGG you trained yourself
appears at the top of the list.

**Classical ML.** Eleven scikit-learn and XGBoost models run on the built-in tables or on a CSV
you import. Each preset runs in seconds. The results show:

- scores;
- ROC and precision-recall curves;
- a confusion matrix;
- feature importance;
- SHAP values, when `shap` is installed.

**Deep Learning.** Design a network layer by layer. As you edit, the page:

- checks each layer against the dataset's image size;
- shows every layer's output shape and parameter count;
- draws the network;
- writes it out as PyTorch code.

*Train now* queues it. *Open in builder* lets you change the training settings first.

## Data and models

**Dataset Hub** has three views:

- **Catalogue:** every dataset with its licence (checked at the source), access requirements, citation and a download button where the terms allow one.
- **Ontology:** the ImageNet-1k classes in WordNet. Names only, never images.
- **COCO:** inspect an annotation file you already have.

**Model Zoo.** Every model with:

- its licence, and its weights' own terms;
- its size and the hardware it needs.

It also lists what the licence policy keeps out, and why.

**Model Registry.** A finished run can become a numbered model version. Each version has:

- a stage and notes;
- the metrics of the run it came from;
- a matching entry in MLflow's registry, when the run was tracked there.

## Experimentation

**Experiment Builder.** Describe an experiment as `experiment.yaml`:

- data, splits and augmentation;
- model;
- optimiser and schedule;
- checkpoints, early stopping and device.

Then save it or queue it. The same file reproduces the run later.

**Training.** Each run shows:

- live progress, ETA and curves;
- its log;
- Cancel, then Resume from the last checkpoint.

Finished runs offer:

- Reproduce;
- Register model;
- Export report;
- Open in CNN Explainer, for TinyVGG.

**Evaluation.** Shows each run's:

- scores;
- confusion matrix;
- per-class report.

You can re-score the best or last checkpoint on the test or validation split.

**Benchmarking.** Times a run's checkpoint, or an untrained model, at several batch sizes. It
reports latency, throughput and peak GPU memory, measured in the worker process.

**Compare Experiments.** Up to eight runs side by side:

- every measure in one table;
- one metric's curves overlaid;
- the comparison exported as CSV, JSON, Markdown or PDF into `reports/`.

**MLflow.** Every run is mirrored to MLflow: parameters, per-epoch metrics, artifacts and the
environment snapshot. Open the MLflow UI from here.

## Tools

**Terminal.** Runs commands in the workspace:

- PowerShell on Windows, bash elsewhere;
- the experiment Python comes first on `PATH`;
- Up and Down recall earlier commands;
- `cd` moves the working folder, and Stop ends the running command.

You cannot type into a running command, and full-screen programs do not work.

**Jupyter Notebook.** Notebooks live in `notebooks/`. A new notebook can start from a run: it
reads that run's files through a relative path and is linked to the run. Jupyter Lab is
installed on request and opens in your browser with a fresh access token.

**Google Colab.** Three steps:

1. Write any experiment as a Colab notebook.
2. Upload it to Colab and run every cell. It trains with the lab's own worker on Colab's GPU and downloads a zip.
3. Import the zip here. It becomes a run like any other.

**Plugin Store.** Shows each integration's status, licence and requirements.

- Connect Kaggle, Hugging Face or Roboflow. Credentials go to the OS keyring and are never shown again.
- Then search and download. Each item shows its licence first, and you confirm it before the download starts.
- Gated or private items are never downloaded: the page links to them instead.

**Hardware Monitor.** Shows:

- what the machine has: CPU, memory, GPU, CUDA, PyTorch;
- the last two minutes of use.

**Documentation.** This guide, the README, the roadmap and the architecture notes.

## Where things are

Everything a project needs lives in its workspace folder:

| Folder | Contents |
| --- | --- |
| `datasets/` | downloads |
| `experiments/<name>-<id>/<run>/` | one folder per run |
| `models/` | models |
| `notebooks/` | notebooks |
| `reports/` | exported reports |
| `results/benchmarks/` | benchmark results |
| `database/` | the lab's database and MLflow's |
| `mlruns/` | MLflow artifacts |
| `logs/` | logs |

Paths are stored relative to the workspace, so the folder can move. Choose another workspace
with `--workspace` or `TEDO_LAB_WORKSPACE`.
