# CLAUDE.md — TEDO AI Research Lab

A PyQt6 desktop workbench for ML and computer-vision research. An experiment is a
file (`experiment.yaml`); it runs in the background, is tracked in SQLite and MLflow,
and leaves a run folder it can be reproduced from. V0.1 is being built in phases
(see **Build status** at the end).

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
app/      PyQt6 lives only here: shell, pages, widgets, theme, Qt services
labs/     domain logic per lab (dataset adapters, model builders, runners) — no Qt
plugins/  integrations, one folder each with plugin.yaml — no Qt, lazy optional deps
core/     contracts and engines — no Qt, never imports app/labs/plugins
```

`tests/test_architecture.py` enforces the import rules and the 400-line limit.

| Path | What it is |
| --- | --- |
| `app/main.py` | Entry point: args, workspace, settings, logging, theme, window, smoke test |
| `app/navigation.py` | `NAV`: every page in sidebar order. Drives sidebar, search, placeholders, smoke test |
| `app/main_window.py` | Top bar + sidebar + `QStackedWidget`; pages built on first visit |
| `app/ui/pages/` | One module per built page; `PAGE_FACTORIES` maps id → page class |
| `app/ui/widgets/` | Kit: `Page`, `PageHead`, `Card`, `Pill`, `KeyValues`, `PathLabel`, `DataTable`, `MarkdownView` |
| `app/ui/theme/` | `tokens.py` (all colours/sizes), `style.qss` (template), fonts |
| `app/resources/` | Bundled fonts (Inter, JetBrains Mono — OFL) and Lucide icons (ISC) |
| `app/services/context.py` | `AppContext` handed to every page (paths, config, credentials, navigate) |
| `core/common/` | `AppPaths`, `AppConfig`, logging + secret masking, `CredentialStore`, optional imports |

### Adding a page

1. Add a `PageSpec` to `NAV` (`planned_for=None` once it is built).
2. Write `app/ui/pages/<page>.py` (subclass `Page`, take `ctx: AppContext`).
3. Register it in `PAGE_FACTORIES`. Tests fail if a built page has no factory or vice versa.

## Conventions

- **No god files**: every `.py` ≤ 400 lines.
- **No fake functionality**: a control works, or it is absent/disabled with a reason
  (tooltip or pill: `Experimental`, `Not connected`, `Planned for vX.Y`). Unbuilt
  adapters raise `NotImplementedError` with a clear message.
- **The UI never blocks**: training/evaluation run in a subprocess
  (`python -m core.experiment_engine.worker <run_dir>`, JSON-lines events on stdout);
  short tasks use `QThreadPool`. Cancel is cooperative ("cancel" on stdin), then kill.
- **Secrets**: only from env vars or the OS keyring (`CredentialStore`, service
  `tedo-ai-lab`). `Secret` values mask themselves; every handed-out value is
  registered with `core.common.masking`, and every log handler has `SecretMaskingFilter`.
  `settings.yaml` rejects credential-like keys.
- **Licences**: every dataset/model card has a licence category (`open-source`,
  `research-only`, `non-commercial`, `commercial`, `gated`, `proprietary`, `unspecified`).
  Never auto-download gated/restricted items; downloads are always user-initiated.
  Ultralytics is AGPL-3.0. Never download or redistribute ImageNet images.
- **Optional dependencies**: check with `core.common.optional.is_installed()` (package
  metadata), never `find_spec` — the workspace's `datasets/` and `models/` folders
  would look like importable packages.
- **Text faces**: Inter for words people wrote, JetBrains Mono for anything a machine
  wrote (paths, URIs, versions, hashes, metrics, logs).
- **Look**: dark, restrained, flat. Colours only from `tokens.py`; one accent (muted
  blue) for anything pressable; status shown with tinted pills. No gradients or glow.
- **Workspace**: runtime folders (`models/ datasets/ experiments/ notebooks/ results/
  reports/ database/ mlruns/ logs/`) are git-ignored; tests use a temporary workspace.
- **MLflow** defaults to `sqlite:///<workspace>/database/mlflow.db`, artifacts in `mlruns/`.

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
- [ ] Phase 2 — registries (dataset/model/plugin cards), SQLite schema, spec, job queue
- [ ] Phase 3 — hardware monitor and dashboard
- [ ] Phase 4 — Experiment Builder, Computer Vision lab, Torch classification runner, Training page
- [ ] Phase 5 — MLflow tracking, reproducibility snapshot, reproduce run, Evaluation
- [ ] Phase 6 — Classical ML / XGBoost lab
- [ ] Phase 7 — Dataset Hub, Ontology Explorer, COCO inspector, Model Zoo, Model Registry
- [ ] Phase 8 — Compare, benchmarking, report export
- [ ] Phase 9 — Plugin Store, Terminal, Notebook, Jupyter, Colab, Deep Learning builder
- [ ] Phase 10 — hardening, README, ROADMAP
