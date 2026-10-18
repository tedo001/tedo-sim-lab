# A tour of the lab

A short walk through every page, in sidebar order. Ctrl+K jumps to a page, a dataset, a model
or a run by name. Ctrl+B folds the sidebar. Ctrl+\ opens a second page beside the first.
Ctrl+` opens a terminal under any page.

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

**Simulation Lab.** Try ideas in quantum physics and computer vision without writing code first.
Each tool saves what you set up (*Save*, then the *Saved* list) and writes it out as a Python
script that runs on its own (*Export script*).

- **Quantum circuits.** Pick a gate, then click a wire to place it (for a CNOT, click the control,
  then the target); right-click removes a gate. You see the exact probabilities, shots sampled
  with a seed, each qubit on its Bloch sphere (shorter than 1 when it is entangled) and the state
  vector with phases. Start from a preset: Bell and GHZ states, teleportation, Deutsch–Jozsa,
  Grover search, the quantum Fourier transform. Circuits export as OpenQASM 2.0.
- **Wave mechanics.** A wave packet in a potential you choose (free, barrier, step, well, box,
  harmonic, double well), moving in time: watch it tunnel, reflect or oscillate, with its norm,
  position, momentum, energy and how much got through. *Compute* finds the potential's lowest
  energies and draws each stationary state.
- **Image lab.** Build a pipeline of image operations (blur, sharpen, edges, corners, thresholds,
  morphology, colour, noise, geometry, Fourier filters, a kernel of your own) on a test pattern or
  your own PNG; every change runs again in the background, with timings and a histogram.
- **Synthetic scenes.** Circles, squares and triangles on varied backgrounds, the same for the
  same seed. *Make the dataset* writes PNG images and COCO annotations into
  `datasets/synthetic/<name>/` (the same button stops it; nothing half-made is kept), and
  *Inspect in the Dataset Hub* opens them there. Shapes may cover each other: the annotations then
  describe only what is visible.
- **Camera model.** A pinhole camera with lens distortion looking at a cube, a ground grid and
  axes. Change its angles, position, focal length and distortion; the matrices K, R, t and
  P = K [R | t] are shown.

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

**Code.** A small IDE for a whole project. Open any folder (*Open folder…*, or *Recent*).

- The tree shows its files. Right-click to make a file or folder, rename, move to the trash, or show it in the file manager.
- The editor colours Python, YAML, JSON, Markdown and TOML, numbers the lines, keeps indentation, and has Ctrl+/ to comment, Ctrl+F to find and replace, and Ctrl+S to save. Saving a Python file checks its syntax and marks the line of an error.
- **Run** (F5) runs the current file: Python with the experiment Python (the project folder is the working folder and is on `PYTHONPATH`, so its own modules import), shell, PowerShell and batch scripts with their own interpreters. Add arguments in the box beside it. Unsaved changes are saved first.
- **Run in terminal** (Ctrl+F5, the terminal button) runs it in a terminal tab instead, for a program that asks you something: Run's output panel has no input.
- **Debug** (Shift+F9) runs a Python file under pdb, Python's own debugger, in a terminal tab. Click the gutter beside a line number, or press F9, to set a breakpoint (a red dot). With breakpoints the program runs to the first one; without, it stops on the first line. The editor follows: the line pdb stopped at opens, marked. Type pdb's commands in the terminal: `n` next line, `s` step into, `c` continue, `p name` print a value, `l` list, `q` quit. A breakpoint set while pdb waits for a command takes effect at once.
- The bottom panel has the run's output (double-click a traceback line to open that file at that line), terminals in the project folder, and git: the branch and the changed files.

**Terminal.** Real terminals in tabs, in the workspace (or the Code page's project):

- PowerShell, Command Prompt, Git Bash and WSL on Windows; bash, zsh and fish elsewhere; the
  experiment Python; vim and Neovim (an install command is offered when one is missing); Docker,
  a shell in a running container or a new container from an image with the workspace mounted;
  or a command of your own. *New* (+) lists them; missing ones say why.
- Full-screen programs work (vim, top, htop, less), with colours, the mouse where a program asks
  for it, selection and copy, paste, scrollback, Ctrl+click on a link or a `file:line`, and zoom.
- Every key goes to the program, the window's shortcuts too, except Ctrl+` (the terminal panel).
- A finished program starts again with Enter. A tab still running something is asked about
  before it closes, and before the app quits.

The same tabs open under any page with Ctrl+` (or the terminal button in the title row).

**Git & GitHub.** Version control for the Code page's project, or any folder:

- **Changes:** stage, unstage or discard files, read each diff, commit (or amend), and push
  straight after committing.
- **History:** the commits, each with its diff.
- **Branches and remotes:** switch branches or make new ones, add remotes, set who commits.
- Fetch, pull and push, with how far ahead and behind you are. A plain folder can become a
  repository; any URL can be cloned.
- **GitHub:** connect a token once (it is checked, then kept in the OS keyring), then filter and
  clone your repositories, see this project's pull requests and issues, or publish the project
  as a new repository.

Every git and GitHub call runs in the background; git never waits for a password.

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

**Logs.** What the lab records, as it happens: the app log, warnings and errors (kept in their
own file, so they outlast the app log's rotation), older logs, and each run's log. Filter by
level, by source (the part of the lab that wrote it) and by text; *Follow new lines* keeps the
newest in view. Choose a record to see it whole; double-click a traceback line to open that file
at that line. *Record … and above* changes how much the lab records, at once. Secrets are masked
before anything is written.

When something goes wrong, a red count appears in the title row; clicking it opens the Logs page
on warnings and errors.

**Documentation.** This guide, the README, the roadmap and the architecture notes.

## Settings

The gear in the title row. Change every setting here: the log level, MLflow tracking and its
store, the terminal shell, how many runs train at once, and the device new experiments start
on. *Save* checks the values together (nothing is saved if one is wrong), writes
`settings.yaml` in the workspace, and says what takes effect now and what from the next run.
*Edit settings.yaml* opens the file in the Code page.

Also here: the workspace's folders, the experiment Python (made by the app, or one you choose),
which credentials are set and where from (never their values), and the lab's version.
*Check for updates* asks GitHub for the latest release; nothing is downloaded. Tick *Check when
the app starts* to be told in the title row.

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
| `results/simulations/` | saved simulations |
| `datasets/synthetic/` | synthetic datasets from the Simulation Lab |
| `database/` | the lab's database and MLflow's |
| `mlruns/` | MLflow artifacts |
| `logs/` | `app.log`, `errors.log` (warnings and errors) and their older copies |

Paths are stored relative to the workspace, so the folder can move. Choose another workspace
with `--workspace` or `TEDO_LAB_WORKSPACE`.
