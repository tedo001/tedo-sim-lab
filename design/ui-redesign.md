# UI redesign: system flow, navigation and screens

A proposal for a cleaner, production-level interface. Nothing here is built yet; the current app
is unchanged. The clickable design (system flow, navigation map, foundations and five screens) is
the canvas *TEDO Lab UI Redesign* on claude.ai:
https://claude.ai/artifact/Gjqpcjw2x7xMW24w48DqyX (private to the owner until shared).

## What is wrong today

Seen in the smoke-test screenshots of every page:

- **The sidebar is a list of features, not a path.** 28 rows, four of them placeholders
  (NLP / LLM, Audio, Multimodal, Quantum ML). At 900 px tall it needs a scroll bar.
- **One run, five pages.** Training, Evaluation, Benchmarking, Compare and MLflow all act on a run,
  and each makes you pick the run again.
- **Home answers "what is installed?", not "what next?".** Half of it is paths, schema version and
  Qt version; with no GPU, two charts say "No NVIDIA GPU detected" in large empty boxes.
- **Empty states are sentences.** "No runs yet. Describe one in the Experiment Builder" has no
  button.
- **The Experiment Builder is one long form.** Problems are listed at the top right, but nothing
  offers to fix them.
- **Contrast.** `text_faint` (#6B737E) is 3.7:1 on `surface`; white on `accent` (#4C7BD0) is
  4.2:1; `fail` (#D2594F) text is 4.4:1. All three are under 4.5:1.

## The system flow

One loop from an open folder to a shipped model. Every screen belongs to one step.

| Step | What happens | Where |
| --- | --- | --- |
| 1 Project | A folder is the workspace; data, runs and code stay inside it | Project switcher in the top bar, setup checklist on Home |
| 2 Data | Find a dataset, read its licence, download it or import a CSV | Library › Datasets / Models |
| 3 Design | Start from a lab preset or five short steps; writes `experiment.yaml` | New experiment, lab presets, Deep Learning builder |
| 4 Run | Preflight passes, the run queues and trains in the worker | Preflight panel, Jobs tray, the run page |
| 5 Analyze | Curves, evaluation, benchmark, explainer; compare up to 8 runs | Run tabs, Runs › select › Compare |
| 6 Ship | Register a version, export a report, reproduce | Run actions, Model registry |

Analyze loops back to Design ("iterate or reproduce"). Always one keystroke away: Ctrl+K
commands, the Jobs tray, the problem count, the terminal panel (Ctrl+\`), split view (Ctrl+\\)
and the live meters. Under the hood nothing changes: Qt app → `JobQueue` → worker subprocess (one
JSON event per line) → `LabStore` → live pages; the worker writes the run folder and MLflow.

## Navigation: 28 rows to 15

```
Home
WORKFLOW    Library · New experiment · Runs · Model registry
LABS        Computer Vision · Classical ML · Deep Learning · CNN Explainer · Simulation Lab
            Coming soon (4)  → expands to Audio v0.2, NLP / LLM v0.3, Multimodal v0.4, Quantum ML TBD
WORKSPACE   Code · Terminal · Git & GitHub · Notebooks
footer      Plugins · Hardware · Logs · Docs · Settings   (icon buttons)
```

| New place | What moved in |
| --- | --- |
| Home | Home, minus paths and schema (to Settings › Workspace) |
| Library | Dataset Hub and Model Zoo as tabs: Datasets, Models, Ontology, COCO inspector |
| New experiment | Experiment Builder as five steps; lab presets open it pre-filled |
| Runs | Training, Evaluation, Benchmarking, Compare, MLflow: a list plus a run page with tabs |
| Model registry | Model Registry |
| Notebooks | Jupyter and Google Colab as tabs |
| Footer | Plugin Store, Hardware Monitor, Logs, Documentation, Settings |

Rules: a row is a place, not a feature; every empty state has a next step; a blocker says how to
fix it; old names still work in Ctrl+K and open the matching tab.

## Screens

- **Home**: Running now (progress, ETA, val accuracy, VRAM, Open run / Cancel, what is next in
  the queue); a setup checklist (workspace, experiment Python, a dataset, MLflow) that collapses
  to "All set"; recent runs; this machine as four meters, not empty charts; three presets.
- **Library**: tabs, filter chips (task, licence, status), a table and a detail panel. The licence
  is shown before anything else; Download stays off until the terms are accepted, then shows
  progress. Catalogue-only and gated datasets explain why the lab does not download them.
- **New experiment**: steps Task, Data, Model, Training, Review, each a button; a summary and a
  preflight panel beside them. Each problem has a fix button (for example "Batch 256 may not fit
  in 4 GB" → Use 128). Run stays off until preflight passes. Review shows the `experiment.yaml`.
- **Runs**: status filters with counts, a table (status pill, model and data, best result,
  epochs with a bar, duration, start), checkboxes and a bar for Compare (2–8 runs) and Export.
- **Run page**: breadcrumb, status, actions (Cancel, Reproduce, Export report, Register model,
  off with a reason while running), a facts strip with progress, and tabs: Overview (four scores,
  validation accuracy and training loss as small multiples), Evaluation (confusion matrix on the
  `seq_*` ramp, per-class report, re-score), Benchmark, Logs, Files and reproduce.

## Foundations

The lab's own tokens stay: dark, flat, one accent, Inter for words and JetBrains Mono for what a
machine wrote. Changes:

| Token | Today | Proposed | Why |
| --- | --- | --- | --- |
| `text_faint` | #6B737E | #7F8792 | 3.7 → 4.9:1 on `surface` |
| `accent` (button fill) | #4C7BD0 | #3B67B8 | white text 4.2 → 5.5:1 |
| `accent_text` (new) | — | #6E9BE6 | links and selected marks, 6.3:1 |
| `fail` (text) | #D2594F | #E06B61 | 4.4 → 5.4:1 |
| `radius` | 4 everywhere | 4 field, 6 button, 8 card, 12 dialog | clearer hierarchy |

Spacing follows 4 · 8 · 12 · 16 · 24 · 32. Type: page title 22/28 600, card heading 15/22 600,
body 13/20, help 12/16, group labels 11/16 caps, numbers 12/18 mono. Buttons are 36 px with one
primary per view. Run states are filled pills, licences outlined pills, so they never read as one
scale. Charts keep the existing rules (one series per chart, 2 px lines, `series_1`).

## Building it

In order, each step shippable on its own and keeping `tests/ui/test_responsive.py` (600 px) green:

1. **Tokens** (`app/ui/theme/tokens.py`, `style.qss`): the four colours, `accent_text`, the radius
   scale. Re-check `series_*` and `heat_*` stay untouched.
2. **Sidebar** (`app/navigation.py`, `app/ui/shell/sidebar.py`): sections `workflow`, `labs`,
   `workspace`; planned pages folded into one "Coming soon" row; footer icons for the system pages.
   Ctrl+K (`search_index.py`) maps old page ids to their new homes.
3. **Empty states**: one `EmptyState` widget (icon, title, text, up to two buttons) in
   `app/ui/widgets/`, used wherever a card says "No … yet".
4. **Runs page**: a run list plus a run page whose tabs host the existing Training, Evaluation,
   Benchmarking and MLflow views (`training_run.py`, `evaluation.py`, `benchmarking.py`); Compare
   opens from a selection.
5. **New experiment**: split `app/ui/pages/builder/form.py` into steps without changing
   `ExperimentSpec`; preflight problems carry a fix action.
6. **Library**: Dataset Hub and Model Zoo under one page with tabs, sharing `catalog_common.py`.
7. **Home**: the checklist and the four meters; workspace facts move to Settings › Workspace.
