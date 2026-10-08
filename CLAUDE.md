# CLAUDE.md — TEDO AI Research Lab

A Qt (PySide6) desktop workbench for ML and computer-vision research. An experiment is a
file (`experiment.yaml`); it runs in the background, is tracked in SQLite and MLflow,
and leaves a run folder it can be reproduced from. V0.1 is being built in phases
(see **Build status** at the end).

## Licence rule (from the owner — always follow)

**Only permissively licensed tools: MIT or Apache-2.0** (BSD, ISC and PSF count as
equivalent). **Never copyleft** such as AGPL or GPL; Ultralytics YOLO is excluded for this
reason. Enforced in code and tests:

- `configs/dependency_licences.yaml` lists every dependency's licence;
  `tests/test_dependency_licences.py` fails on a missing or non-permissive one.
- Third-party model cards and plugins are refused at load time unless their licence is
  permissive (`tool_licence_problem`; `origin: builtin` = this project's own code).
- The one documented exception is **PySide6 (LGPL-3.0)**: no permissive Qt binding exists.
  PyQt6 (GPL) is not used. Keep Qt a separate, replaceable shared library.

## Git rules (from the owner — always follow)

- Work and push on branch **`tedo`** only.
- Commits are authored and committed as **`tedo001 <durgamani.d.e.c.e.50@gmail.com>`**
  (set in this repo's local git config). Use that email for every author field
  (pyproject, plugin manifests, docs).
- Commits carry **no** `Co-Authored-By` or session trailers; PR descriptions carry no
  "Generated with" line. Every push is the owner's.
- Do not open pull requests unless asked.

## Commands

```bash
python -m app.main                                   # run the app
python -m app.main --workspace D:/lab                # runtime data elsewhere
QT_QPA_PLATFORM=offscreen python -m app.main --smoke-test [--screenshots DIR]
pytest                                               # full suite (offscreen Qt)
ruff check .                                         # lint
```

Install: CPU-only machines install PyTorch first
(`pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu`),
then `pip install -e ".[dev]"`. Linux needs Qt's system libraries:
`libegl1 libgl1 libxkbcommon0 libfontconfig1 libdbus-1-3`.

## Architecture

```
app/      Qt (PySide6) lives only here: shell, pages, widgets, theme, Qt services
labs/     domain logic per lab (dataset adapters, model builders, runners) — no Qt
plugins/  integrations, one folder each with plugin.yaml — no Qt, lazy optional deps
core/     contracts and engines — no Qt, never imports app/labs/plugins
```

`tests/test_architecture.py` enforces the import rules and the 400-line limit.

| Path | What it is |
| --- | --- |
| `app/main.py` | Entry point: args, workspace, settings, logging, theme, window, smoke test |
| `app/navigation.py` | `NAV`: every page in sidebar order. Drives sidebar, search, placeholders, smoke test |
| `app/main_window.py` | Top bar + sidebar + splitter (main page stack, optional split pane); Ctrl+B / Ctrl+\\ |
| `app/ui/shell/` | `TopBar`, `Sidebar` (collapses to an icon rail), `SplitPane` (second page, own instances) |
| `app/services/ui_state.py` | `UiState`: remembered layout via `QSettings` (per person, never in a project) |
| `app/ui/pages/` | One module per built page; `PAGE_FACTORIES` maps id → page class |
| `app/ui/widgets/` | Kit: `Page`, `PageHead`, `Card`, `Pill`, `KeyValues`, `ElidedLabel`/`PathLabel`, `DataTable`, `MarkdownView` |
| `app/ui/theme/` | `tokens.py` (all colours/sizes), `style.qss` (template), fonts |
| `app/resources/` | Bundled fonts (Inter, JetBrains Mono — OFL) and Lucide icons (ISC) |
| `app/services/context.py` | `AppContext` (paths, config, credentials, catalogue, store, jobs, navigate); `build_context()` |
| `app/services/jobs.py` | `JobQueue`: runs via `QProcess` worker (FIFO, `max_concurrent_runs`), tasks via `QThreadPool` |
| `app/services/hardware.py` | `HardwareService`: probe in its own process, 1 s psutil/NVML sampler, 2-min history |
| `app/ui/widgets/charts.py` | `TimeSeriesChart` (small-multiple live chart, hover), `MeterBar` (title-row gauges) |
| `app/ui/pages/resources.py` | `ResourceStats` / `ResourceCharts` shared by Home and the Hardware Monitor |
| `labs/computer_vision/explainer/` | CNN Explainer engine (numpy, no torch): `Architecture`/`tiny_vgg`, `ExplainerNet` (state_dict-named weights, `.npz`), `run()` → `Trace`, `conv_step`/`pool_step`/`softmax_terms`/`linear_contributions`, `map_limits`, `ConvGeometry`, samples |
| `labs/computer_vision/models.py` | PyTorch builders: SimpleCNN, LeNet-5, ResNet-18, TinyVGG (`to_torch` keeps layer names, so a `state_dict` loads into `ExplainerNet`) |
| `labs/computer_vision/datasets.py` | MNIST, Fashion-MNIST, CIFAR-10 adapters (torchvision layout under `datasets/<id>/`, own downloader in `labs/common/download.py`) |
| `labs/computer_vision/classification.py` | `TorchClassificationRunner` (+ `training/`: data splits and transforms, loop, metrics, checkpoints); `export.py` hands TinyVGG to the explainer |
| `labs/computer_vision/presets.py` | Preset experiments shown in the Computer Vision lab |
| `app/services/experiments.py` | `ExperimentService`: launch (experiment row + run folder + queue), record worker events in `LabStore`, cancel, resume, builder drafts; `runs.py`: `RunView`, `LiveState` (ETA) |
| `labs/classical_ml/` | `SklearnRunner` (`runner.py`): `tables` (target/features/subset, seeded split), `pipeline` (impute, one-hot, scale, `SelectKBest`, grid/random search), `models` (`BUILDERS`), `report` (metrics, ROC/PR, permutation importance, clusters, projections), `explain` (optional SHAP), `datasets` (scikit-learn tables, `CsvAdapter`, `import_csv`), `presets` |
| `app/services/downloads.py` | `DownloadService`: one background download per dataset, licence acknowledgement; `import_csv` (card + file into `datasets/imported/`, `imported` signal) |
| `app/ui/pages/builder/` | Experiment Builder (`form.py` ⇄ `ExperimentSpec`, switching `torch_section.py` / `sklearn_section.py` by task; `fields.py` controls and YAML boxes; `choosers.py` dataset/model + downloads) |
| `app/ui/pages/classical_ml.py` | Classical ML lab: presets (Run / Open in builder), `csv_import.py`, tabular datasets and models |
| `app/ui/pages/sklearn_results.py` | `SklearnResults` from `results.json`, on the Training and Evaluation pages; charts in `app/ui/widgets/plots.py` (`CurveChart`, `ScatterChart`, `BarList`) |
| `core/experiment_engine/snapshot.py` | Reproducibility snapshot (`snapshot.json`, `code.diff`): Python, platform, packages, git, dataset fingerprint; `compare()` |
| `core/experiment_engine/recording.py` | What a worker records around a run: snapshot + `MlflowTracker` (`core/tracking/mlflow_tracker.py`) |
| `app/ui/pages/evaluation.py` | Evaluation: scores, `ConfusionMatrixView`, per-class report, re-score best/last on test/val (`worker --evaluate`) |
| `app/ui/pages/mlflow_page.py` | MLflow page: tracking store, runs read in a quiet task, `MlflowUi` service (`app/services/mlflow_ui.py`) |
| `app/ui/pages/training.py` | Training page (`training_run.py`: progress, ETA, `EpochChart` curves, results, log, cancel/resume) |
| `app/ui/pages/explainer/` | CNN Explainer page: `overview` (all maps, links), `detail_*` (conv, ReLU, pool, input, softmax), `playground`, `inputs` (samples, open image, draw pad), `article` |
| `core/hardware/` | `info.probe_hardware()` (run as `python -m core.hardware.info`), `sampler`, `devices` |
| `core/common/` | `AppPaths`, `AppConfig`, logging + masking, `CredentialStore`, licensing, vocab, cards, cancel |
| `core/catalog.py` | `load_catalog()`: dataset, model, plugin and runner registries in one object |
| `core/dataset_registry/` | `DatasetCard`, `DatasetRegistry` (live status), `DatasetAdapter` contract |
| `core/model_registry/` | `ModelCard` (+ separately licensed `WeightsInfo`), `ModelRegistry` |
| `core/plugin_api/` | `PluginManifest`, `Plugin`, `ManifestPlugin` (status/actions from manifest), `PluginRegistry` |
| `core/experiment_engine/` | `ExperimentSpec` (strict, canonical YAML), runner contract, events, `worker` |
| `core/tracking/` | SQLite: `Database` + `migrations/NNNN_*.sql`, `LabStore` (relative paths) |
| `configs/` | Shipped cards (`datasets/`, `models/`) and `runners.yaml` (runner entries) |
| `plugins/<name>/` | `plugin.yaml` manifest + `plugin.py` entry (a `ManifestPlugin` subclass) |

### Adding a page

1. Add a `PageSpec` to `NAV` (`planned_for=None` once it is built).
2. Write `app/ui/pages/<page>.py` (subclass `Page`, take `ctx: AppContext`).
3. Register it in `PAGE_FACTORIES`. Tests fail if a built page has no factory or vice versa.

### Adding a dataset, model, runner or plugin

- Dataset/model: add a card to `configs/datasets/*.yaml` or `configs/models/*.yaml`. Licence
  facts must come from the source; no licence published → `unspecified`. Cards start as
  `maturity: planned`; flip to `stable` only when the loader/builder exists (a test checks).
- Runner: subclass `ExperimentRunner` in `labs/…`, add its `"module:Class"` to `configs/runners.yaml`.
- Plugin: copy `plugins/custom/`, edit `plugin.yaml` (name = folder) and `plugin.py`.

## Conventions

- **No god files**: every `.py` ≤ 400 lines.
- **No fake functionality**: a control works, or it is absent/disabled with a reason
  (tooltip or pill: `Experimental`, `Not connected`, `Planned for vX.Y`). Unbuilt
  adapters raise `NotImplementedError` with a clear message.
- **The UI never blocks**: training/evaluation run in a subprocess
  (`python -m core.experiment_engine.worker <run_dir> [--runner m:C] [--cancel-on-eof]`):
  one JSON event per stdout line, everything else printed goes to stderr; exit 0 done,
  1 failed, 2 cancelled. Short tasks use `QThreadPool`. Cancel is cooperative ("cancel" on
  stdin, `CancelToken`), then kill after a grace period. Runner discovery is by
  `configs/runners.yaml` strings, so `core` never imports `labs`.
- **Experiment Python**: workers run under `config.python_executable` (empty = the app's own
  interpreter) with the code root on `PYTHONPATH` and `TEDO_LAB_WORKSPACE` set to the workspace
  (the app process sets it too, so runner checks and workers see the same datasets); plugin
  installs go there too. PyTorch downloads (pretrained weights) go to `<workspace>/models/torch-hub`.
- **Runs**: a run folder is `experiments/<name>-<experiment id>/<run id>/` with `experiment.yaml`,
  `checkpoints/{last,best}.pt` (`last.pt` every epoch = the resume point), `metrics.jsonl`,
  `run_info.json`, `test_confusion.json`, `snapshot.json` (+ `code.diff`), `run.log`, `evaluations/`
  and, for TinyVGG, `explainer.npz/json`. scikit-learn runs (`RunView.tabular`) are one fit: one
  "epoch", stages as steps, `checkpoints/model.joblib` (a pickle: only load the lab's own),
  `results.json` (everything the results views draw) and `test_confusion.json` for
  classification; unsupervised tasks fit every row. The app records events in `LabStore` (metrics per
  epoch); the worker mirrors each run to MLflow (`mlflow_tracking` setting; `TEDO_LAB_MLFLOW=0`
  switches it off, which the test suite does). Tracking never fails a run. "Reproduce" queues the
  same `experiment.yaml` as a child run and lists environment differences first.
- **Quiet jobs**: housekeeping tasks (`submit_task(..., quiet=True)`) are not recorded, not
  "active", and never ask before quitting. Never delete a `QProcess` inside its own signal:
  keep it alive until the event loop is back (see `JobQueue._release`, `MlflowUi._release`).
- **Database**: one SQLite file per workspace (`database/lab.db`), WAL mode, one connection
  per thread. Schema changes are new numbered files in `core/tracking/migrations/`; never edit
  an applied one. Paths inside the workspace are stored relative (projects can move).
- **Secrets**: only from env vars or the OS keyring (`CredentialStore`, service
  `tedo-ai-lab`). `Secret` values mask themselves; every handed-out value is
  registered with `core.common.masking`, and every log handler has `SecretMaskingFilter`.
  `settings.yaml` rejects credential-like keys.
- **Licences**: every dataset/model card has a licence category (`open-source`,
  `research-only`, `non-commercial`, `commercial`, `gated`, `proprietary`, `unspecified`).
  Never auto-download gated/restricted items; downloads are always user-initiated.
  Never download or redistribute ImageNet images. Tool licences: see the licence rule above.
- **Optional dependencies**: check with `core.common.optional.is_installed()` (package
  metadata), never `find_spec` — the workspace's `datasets/` and `models/` folders
  would look like importable packages.
- **Text faces**: Inter for words people wrote, JetBrains Mono for anything a machine
  wrote (paths, URIs, versions, hashes, metrics, logs).
- **No CUDA in the UI process**: never call `torch.cuda.*` from the app process (it pins a CUDA
  context and ~300 MB of VRAM). Hardware facts come from the probe subprocess; live GPU use
  from NVML only.
- **Charts** follow the dataviz rules: one series per chart (small multiples), one axis, 2 px
  lines, recessive grid, values in text colours, gaps for missing data, hover crosshair.
  Series colour `series_1` is validated against the dark surface. Scatter groups: `series_1..3` pass
  all-pairs checks; with more than three groups one is highlighted and the rest fold into
  `series_other`.
- **CNN Explainer** is a port of poloclub/cnn-explainer (MIT): keep its notice in
  `labs/computer_vision/explainer/LICENSE-cnn-explainer.txt` and the credit on the page. Never copy
  its Tiny ImageNet images or pretrained weights (ImageNet-derived). Its numpy engine must match
  PyTorch exactly (`tests/core/test_explainer.py`); activations use the diverging `heat_*` tokens.
- **Narrow panes**: pages must work at half width (split view). Key/value rows are single
  line and elide (full text in tooltip); tables stretch a text column, never a pill column.
- **Look**: dark, restrained, flat. Colours only from `tokens.py`; one accent (muted
  blue) for anything pressable; status shown with tinted pills. No gradients or glow.
- **Workspace**: runtime folders (`models/ datasets/ experiments/ notebooks/ results/
  reports/ database/ mlruns/ logs/`) are git-ignored; tests use a temporary workspace.
- **MLflow** defaults to `sqlite:///<workspace>/database/mlflow.db`, artifacts in `mlruns/`.

## Product direction (from the owner)

After V0.1 the lab ships as a **Windows installer** (PyInstaller + Inno Setup, the way
`tedo001/sentra` does it) and grows **IDE-like project handling**: open a local folder as a
project and manage everything in it (files, editor, terminal, git, experiments, notebooks).
Design for it now:

- Never write into the code/install folder; all data goes to the workspace (= project folder).
- Keep everything a project needs inside its folder; store paths relative to it.
- The installed app cannot pip-install into itself: experiments and plugins use a separate
  "experiment Python" environment (`python_executable`).
- Resources are found relative to package files (works in a PyInstaller bundle).

## Environment notes

- Development container: Linux, CPU only, Python 3.13. Target: Python ≥ 3.11, primary
  user machine Windows with a 4 GB GPU (keep batch-size/precision defaults within 4 GB).
- In the cloud dev container, `download.pytorch.org` and `cdn.jsdelivr.net` are blocked;
  PyPI and the npm registry work. CI (GitHub Actions) uses the CPU PyTorch index.

## Build status (V0.1)

- [x] Phase 0 — plan approved (defaults: repo-root layout; Windows/4 GB target;
      Simulation Lab and Quantum ML as placeholders; `unspecified` licence category;
      MLflow on SQLite; CI on GitHub Actions)
- [x] Phase 1 — skeleton, theme, sidebar, every page (placeholders), logging with
      masking, settings, credentials, smoke test, CI
- [x] Phase 2 — catalogue (10 dataset, 18 model cards (19 with TinyVGG), 8 plugins, 5 experimental runners),
      licence policy, SQLite schema + migrations + LabStore, strict spec with canonical YAML,
      worker protocol, Qt JobQueue
- [x] Phase 3 — hardware probe (subprocess) + live sampler, Hardware Monitor, Home dashboard,
      title-row meters; also: licence policy (PySide6, no AGPL), collapsible sidebar, split view
- [x] Phase 4 — dataset loaders (MNIST, Fashion-MNIST, CIFAR-10) with licence-checked downloads,
      SimpleCNN / LeNet-5 / ResNet-18 / TinyVGG builders, TorchClassificationRunner (splits,
      augmentation, schedules, AMP, checkpoints, early stopping, cancel, resume, test metrics),
      ExperimentService, Experiment Builder, Computer Vision lab with presets, Training page;
      trained TinyVGG runs open in the CNN Explainer
- [x] Phase 4b (built early) — CNN Explainer (port of poloclub/cnn-explainer via tedo001, MIT):
      layer overview, convolution / ReLU / pooling / softmax views, hyperparameter playground,
      article, samples / open image / draw a digit; lists trained TinyVGG runs first
- [x] Phase 5 — MLflow tracking from the worker (params, per-epoch metrics, artifacts, tags,
      never fails a run), reproducibility snapshot, Reproduce (child run, differences, result
      comparison), Evaluation page (confusion matrix, per-class report, re-score checkpoints),
      MLflow page with the MLflow UI
- [x] Phase 6 — Classical ML lab: SklearnRunner (classification, regression, clustering, PCA;
      imputation, one-hot, scaling, SelectKBest, CV, grid/random search, permutation importance,
      optional SHAP), 11 scikit-learn/XGBoost models, Diabetes card, CSV import, presets, builder
      scikit-learn section, results on the Training and Evaluation pages
- [ ] Phase 7 — Dataset Hub, Ontology Explorer, COCO inspector, Model Zoo, Model Registry
- [ ] Phase 8 — Compare, benchmarking, report export
- [ ] Phase 9 — Plugin Store, Terminal, Notebook, Jupyter, Colab, Deep Learning builder
- [ ] Phase 10 — hardening, README, ROADMAP
