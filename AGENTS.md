# AGENTS.md - Hexaboard Visual Inspection Current Spec

## Current Project Summary

This repository trains reconstruction models and inspects preprocessed CMS HGCAL
hexaboard NumPy tensors. It provides pixel-wise reference comparison, autoencoder
reconstruction scoring, calibration and analysis commands, and a vanilla
HTML/CSS/JavaScript inspection and labeling interface.

The supported workflow starts with an existing board `.npy` file. Scanner
acquisition and cropping are placeholders, not an operational scan-to-inspection
path. Reflection mitigation is planned in the dedicated section below; the
current production path still uses one reference and whole-segment scores.

| Contract | Current implementation |
| --- | --- |
| Stored board | BGR `uint8`, shape `(8, 5, 1060, 1882, 3)` |
| Board grid | 40 positions; 32 inspected with the canonical eight-position mask |
| Model input | RGB float `[B, 3, 1060, 1882]`, normalized to `[0, 1]` |
| Models | CNN and ResNet autoencoders returning reconstruction logits |
| Training CLI | `CNNAutoencoder`, configured by `configs/train_CNNAutoencoder.yaml` |
| Training YAML architecture | Latent dimension 32, initial filters 128, layers `[2, 2, 2]` |
| Inspection YAML architecture | Latent dimension 64, initial filters 128, layers `[2, 2, 2]`, for `run_01.pt` |
| Training loss | `BCEWithLogitsLoss` on raw logits |
| Pixel score | `1 - SSIM(reference, inspected)` |
| AE score | Whole-segment MAE after sigmoid |
| Hybrid | Intersection of pixel and AE flags |
| UI server | Python `ThreadingHTTPServer`, default `127.0.0.1:3000` |

Training and inspection configurations describe different checkpoints; do not
copy the training YAML's latent dimension into inspection settings without
checking the weights. The inspection loader can infer CNN dimensions when
`inspection.model` is `{}`. ResNet exists and is tested, but the current shared
inspection loader and training CLI instantiate the CNN.

---

## Implementation Boundaries

Keep data loading, models, training, calibration, inspection, and HTTP presentation
in their existing packages. Extend reusable APIs instead of duplicating logic in
scripts or notebooks.

- `scripts/` is execution-only; application code must not import from it.
- `configs/inspection.yaml` drives calibration, inspection, analysis, evaluation,
  and server startup. `configs/train_CNNAutoencoder.yaml` drives training.
- Scanning, cropping, tensor loading, and model execution belong server-side.
  Scanner/cropper work remains separate from HTTP handlers.
- The browser receives metadata, flag grids, and requested segment PNGs; it does
  not load board tensors. Preserve the vanilla frontend and Python serving path.
- Keep generic training infrastructure in `Trainer`; autoencoder-specific tensor
  dispatch belongs in `AutoencoderTrainer`, which inherits the shared loop.
- Keep the server package split and `python -m scripts.server`. There is no
  monolithic `web/server.py` to refactor and no frontend Node build requirement.
- Keep notebooks as clients of repository APIs with parameters declared in cells.
- Preserve unrelated working-tree changes, existing labels, datasets, and weights.
  Generated-artifact cleanup and Git tracking changes require a scoped decision.
- The reference agents file is a formatting example only. Requirements for other
  projects, datasets, or model families do not belong in this repository.

Store board inputs and inspection sidecars under the configured data paths,
calibration arrays/manifests under `calibrations/`, reports and plots under
`outputs/`, and training logs/weights/checkpoints/run snapshots under `logs/`.
Reusable YAML configurations belong under `configs/`; masking manifests, templates,
and annotation JSON belong under `calibrations/`. Documentation images belong in
`assets/`. Keep raw sensor dumps and bulky datasets out of commits.

---

## Current Implementation Status

### Repository verification: 2026-09-28

This is a source-and-tests snapshot, not a claim of detection accuracy or a clean
installation on every machine. It replaces the obsolete migration-gap checklist.

Verified in the existing local environment:

- Python 3.12.3, PyTorch 2.13.0, TorchVision 0.28.0, NumPy 1.26.4.
- Full pytest suite: 37 passed. Five warnings concern CPU pin-memory use and the
  test that wraps an optimizer step to exercise graceful interruption.
- Ruff checks, Bash job syntax, and frontend JavaScript syntax pass.
- The train, evaluate, calibrate, analyze, inspect, and server CLI help paths were
  checked; command examples below follow their current arguments.
- Headers of the 124 boards in the five configured data folders have canonical
  shape and `uint8` dtype: 41 train, 3 validation, 20 test, 59 bad, 1 bad example.
  These are file counts, not proof of unique physical boards or label quality.
- Six local calibration arrays have shape `(8, 5)`; damaged-label JSON has 123
  entries. Header/count checks do not establish artifact or annotation validity.
- `calibrations/manifest.json` exists, but `validate_calibration` rejects the
  current default settings with "Baseline, checkpoint, or skip mask changed."
  Regenerate calibration before using current default inspection/analysis.

Earlier work reported real-board calibration, inspection, a live HTTP job, and a
one-update training/checkpoint smoke run. Those were not repeated in this refresh,
and `outputs/pipeline_smoke/` is not present in this checkout. Do not cite absent
reports as reproducible current validation. No full calibration, training, live
server, GPU/DDP, scanner, or Docker image build was run for this documentation
refresh.

### Implemented and maintained

- Strict board/dtype and skip-coordinate validation, BGR-to-RGB conversion,
  single-file/directory datasets, and memory-mapped segment access.
- CNN/ResNet model shape and gradient checks; portable CNN checkpoint loading,
  architecture inference, and compatibility validation for inspection.
- Shared trainer loop with AE tensor dispatch, configurable scheduler intervals,
  epoch/step validation, callbacks, interruption handling, and checkpoint state.
- Streaming calibration, unique normal samples per input path/coordinate,
  deterministic anomalous-board selection, and threshold provenance.
- Inspection result serialization beside board files with signature-based reuse.
- Analysis over all non-skipped positions on boards excluded by calibration ID;
  streaming reconstruction evaluation with bounded retained examples.
- Server path/coordinate validation, label persistence, on-demand segment PNGs,
  configured initial board, and inspection subprocess execution.
- Reversed board display/navigation matching notebook 04 while preserving source
  coordinates and individual image orientation.

### Remaining work and limits

1. Recalibrate current artifacts before inspection. A passing test suite does not
   make stale thresholds compatible with a checkpoint or baseline.
2. Complete the reflection plan below. Run 1 schemas/storage for batch metadata,
   region templates, and annotations are implemented separately from inspection.
   Runs 2–4 add mask geometry, annotation APIs, and the browser mask editor.
   Run 5 adds regional review with explicit status, visibility, and completion.
   Run 6 adds preview-first legacy migration and template reconciliation. Multiple
   reference aggregation remains future work. Runs 7–8 provide opt-in local
   thresholds, persisted decisions, and inspection uncertainty UI.
3. Validate normal/defective review completeness and duplicate physical-board
   identities. Legacy labels do not distinguish unreviewed from normal, and
   legacy split logic relies on filenames rather than a reviewed scan manifest.
   Regional workflows have explicit review and split metadata requiring real-data validation.
4. Validate accuracy on representative held-out boards of both appearances.
   Reconstruction loss and operational smoke tests are not defect-detection metrics.
5. GPU/DDP behavior and exact shuffled mid-epoch resume determinism remain
   unverified. Checkpoints do not preserve every random/sampler state.
6. Implement scanner/cropper integration separately; `jobs/main.sh` currently
   inspects only a preprocessed board.
7. Treat the server as a local inspection tool. Path and body validation exists,
   but it is not a complete authenticated multi-user deployment design.
8. Dependency security updates and clean-install compatibility remain separate
   work. `requirements.txt` includes Linux CUDA and notebook/development packages;
   a successful local environment is not proof of cross-platform installation.
9. Local `docker/Dockerfile` now uses Ubuntu 24.04, Python 3.12, and the shared
   requirements in `/opt/venv`; it expects a checkout mounted at
   `/workspace/visual-inspection`. An image build has not been verified here.
   No `.github` CI workflow or `requirements-dev.txt` is present.
10. `AGENTS.md` and `docker/` remain Git-ignored. Some generated arrays and weights
    are already tracked; ignore rules do not untrack them. Do not delete or
    silently change tracking as part of another task.

---

## Target Code Surface

| Location | Responsibility |
| --- | --- |
| `src/utils/data/` | Board contracts/loading, datasets, skip masks, labeling, image helpers |
| `src/models/` | CNN and ResNet autoencoders |
| `src/configs/`, `configs/` | Model/train settings and shared inspection configuration |
| `src/engine/` | Generic training loop and AE tensor adapter |
| `src/loss/`, `src/optim/` | Loss, optimizer, wrapper, scheduler registries |
| `src/utils/metrics.py` | Shared SSIM/AE score computation |
| `src/utils/get_threshold.py` | Score collection and threshold selection |
| `src/inspection/` | Checkpoint loading, calibration, inspection, analysis, evaluation, artifacts |
| `src/utils/get_results.py` | Result grids, metadata, serialization |
| `src/utils/metrics.py`, `src/utils/viz/` | Metrics and plot helpers |
| `scripts/` | Execution-only CLI entry points |
| `web/server/`, `web/app/` | HTTP handling and inspection/label/job coordination |
| `web/configs/`, `web/utils/` | File-derived paths and image/HTTP support |
| `web/cache/` | Existing compatibility cache helpers; current app uses persisted inspection signatures |
| `web/interface/` | Static HTML, CSS, JavaScript |
| `scanner/`, `jobs/` | Acquisition placeholders and preprocessed-board orchestration |
| `tests/`, `notebooks/` | Automated checks and interactive API clients |

---

## Data and Board Indexing Contract

The canonical stored shape is `(8, 5, 1060, 1882, 3)` with BGR `uint8` pixels.
`open_board` memory-maps arrays with `allow_pickle=False` and validates shape and
dtype. Explicit alternate spatial dimensions are allowed for synthetic tests;
do not silently accept legacy production shapes.

```text
board[row, col] -> BGR segment[height, width, channel]
flat_grid_index = row * 5 + col
model segment -> RGB float tensor[3, 1060, 1882] in [0, 1]
```

`HexaboardDataset` accepts one `.npy` file or a directory of `.npy` files, validates
all discovered boards, and yields non-skipped segments in row-major order.
Its segment index runs over usable positions, so it is not the full-grid flat
index above. Without a transform it returns normalized RGB tensors; a custom
transform receives RGB `uint8` and owns conversion to the model-input contract.

`calibrations/skipped_segments.json` is authoritative. Current physical skips are:

```text
(0, 0), (0, 4), (1, 0), (1, 4),
(6, 0), (6, 4), (7, 0), (7, 4)
```

Reject non-integral or out-of-bounds coordinates. Validate compatible reference,
board, model, and threshold shapes before processing. Normalize once and preserve
consistent color/range conventions across scoring and training.

The UI displays source `(row, col)` at `(7 - row, 4 - col)`. It reverses both grid
axes, not the pixels within a segment. Navigation follows display order. API and
stored coordinates remain zero-based source coordinates; visible segment numbers
are one-based. Do not transpose data to reproduce this presentation.

Damaged labels use a `files` mapping keyed by board paths, with board shape and
`{'row', 'col'}` entries. Web labels use repository-relative paths; current
calibration/analysis also derive identities from filename stems. Preserve those
identities and reject conflicting labels. Stronger physical-board/scan identity
and review completeness are part of the planned reflection manifests.

---

## Model and Trainer Contract

Both autoencoders return logits at the configured spatial shape. Use logits for
`BCEWithLogitsLoss`; apply sigmoid for images and MAE scoring. The inspection
model loader accepts CNN weights or a compatible trainer checkpoint and verifies
explicit architecture settings against the checkpoint.

The generic trainer accepts supervised `(X, y)` batches. `AutoencoderTrainer`
overrides `_prepare_batch` to use one normalized tensor as both input and target;
it inherits the shared training loop, device handling, loaders, optimizer,
schedulers, callbacks, logging, interruption guard, and checkpoint management.
Its `evaluate()` retains the three-value `(loss, targets, predictions)` API.
Use the streaming evaluation CLI for full-resolution datasets; trainer evaluation
collects returned image arrays in memory.

`TrainConfig.scheduler_interval` supports `optimizer_step`, `epoch`, and
`validation`; its default is `validation`, while the training YAML explicitly
uses `epoch`. `ReduceLROnPlateau` requires `validation`. Step-based validation also
runs at epoch boundaries if that boundary was not already evaluated.

The training CLI seeds with 42, loads the model/train YAML, uses available CUDA
or CPU, and launches multiple ranks when multiple CUDA devices are visible.
Missing/incompatible resume checkpoints fail instead of silently restarting.
Best weights and resumable checkpoints are separate. Run snapshots record model,
training, and dataset information; do not claim complete experiment provenance or
exact resume determinism without checking the required fields and random state.
The first Ctrl-C requests a completed-update checkpoint; a second interrupts.

The YAML batch size of 8 and 100 epochs are experiment settings, not a portable
smoke-test recipe. Use bounded data and a separate short-run YAML before long
training. Verify target-device memory and distributed behavior independently.

---

## Calibration and Inspection Contract

The current default uses whole-segment scalar scores:

```text
pixel_score = 1 - SSIM(baseline, inspected)
ae_score = MAE(sigmoid(reconstruction_logits), inspected)
flagged = score >= per_segment_threshold
hybrid = pixel_flagged AND autoencoder_flagged
```

`configs/inspection.yaml` currently selects baseline
`data/train/320XLF4CQH00001.npy`, example board
`data/bad_example/320XLF4CQH00445.npy`, and checkpoint
`logs/CNNAutoencoder/best/run_01.pt`. Paths resolve against the repository root;
run CLI commands from that root. Update the checkpoint and architecture together,
or use `model: {}` to infer the supported CNN dimensions.

Calibration defaults to normal boards in train/validation folders, excluding the
reference path and IDs with damaged labels. It selects half of annotated bad
boards deterministically with seed 42. Normal samples are scored once per input
path/coordinate, not repeatedly per defect entry. Distinct paths can still refer
to copies of one physical board; review duplicates explicitly.

Threshold selection with both classes maximizes F1, then favors lower false-positive
rate, higher true-positive rate, and a higher threshold. Positions without defect
examples use the configured normal-score quantile (default 0.99), nudged upward
to avoid flagging an equal normal score. Calibration requires normal boards and
some annotated anomalous boards overall. Do not confuse physical skips with
uncomputed positions or the lower-level utility's `fill_mode` behavior.

Calibration writes six `(8, 5)` arrays and then `manifest.json`. The manifest
records config, sources, dependencies, and threshold signatures. Inspection
rejects missing/stale provenance. Current signatures use paths, file sizes, and
modification times; they are not content hashes or a complete provenance system.
Recalibrate after changes to the reference, model, skips, score semantics, or data
policy. Do not manually modify signatures to bypass incompatibility.

`inspect_board` saves `<board>.inspection.json` beside each board and reuses it
when the source/settings signatures match. `--force` recomputes results, but does
not bypass calibration validation. Analysis excludes calibration and reference
IDs from test/remaining-bad boards and evaluates every physically unskipped
position. It currently treats unmarked positions as normal, so label completeness
is essential; batch/review-aware evaluation is planned, not implemented.

Reports and plots go to `output_dir`, normally `outputs/inspection`. Reconstruction
evaluation reports BCE loss for `test_dir` and the configured example board and
saves one selected example; its index defaults to 0. That loss is not detection
accuracy. Planned local scores and appearance-specific calibration require new
artifacts and are specified in the reflection section below.

---

## HTTP and Frontend Contract

Maintain `web.server.run_server` and `python -m scripts.server`. Server config is
file-derived and shared with inspection. The application validates `.npy` paths
inside `data_root`, bounds-checks coordinates, and validates label payloads.

| Endpoint | Current behavior |
| --- | --- |
| `GET /api/config` | Configured initial board path |
| `GET /api/board?path=...` | Metadata, labels, skips, inspection grids; computes inspection if cache is stale/missing |
| `GET /api/segment?path=...&row=...&col=...` | One source-coordinate RGB segment as base64 PNG |
| `POST /api/label` | Persist `{path, row, col, damaged}`; `damaged` is boolean |
| `POST /api/run` | Force inspection of `{path}`, returning status, stdout, stderr |

Results expose `pw_flags`, `ae_flags`, `hybrid_flags`, metadata, counts, and flagged
coordinates. Encoding remains `-1 = skipped`, `0 = OK`, `1 = flagged`. There is
no quality/uncertainty status in legacy mode. Opt-in local mode uses schema version
2 and authoritative `local.decision_grid` values: skipped/ok/flagged/uncertain.
Zero compatibility flags must never be interpreted as passes in this mode.

On POSIX, `/api/run` invokes `jobs/main.sh`; Windows invokes the same inspection
CLI directly. The job uses the server interpreter and a temporary snapshot of its
resolved config. The resulting sidecar is reused on subsequent board retrieval;
the job is not followed by a redundant inference run. Shell execution does not
scan or crop a new board. Keep images and inference server-side, and preserve
source coordinates across labels, status grids, and reversed display navigation.

---

## Code Implementation Plan

The original geometry/trainer/server migration is implemented and covered by the
current checks. Do not repeat completed refactors based on an obsolete gap list.
Use this order for subsequent work:

1. Keep the current data, model, calibration, persistence, and HTTP contracts
   passing; resolve stale local calibration before data-dependent inspection.
2. Implement reflection batches 1 and 2 with synthetic fixtures while the data
   team prepares reviewed board/reference/region manifests. Keep new paths opt-in.
3. Add optional correction and training support in batch 3, without claiming
   real-data effectiveness from synthetic tests.
4. Validate the pilot handoff, masks, splits, and fixed operating-point metrics;
   compare baseline and reflection experiments before promoting a configuration.
5. Run short target-device training and artifact-consumer checks before any longer
   run. Validate CUDA/DDP separately when used.
6. Treat scanner integration, dependency remediation, Docker build verification,
   and CI setup as distinct tasks with explicit validation.

Updating documentation or passing unit tests does not authorize a long training
run or establish production accuracy. The detailed reflection contracts and data
team responsibilities below govern that next phase.

---

## Reflection Mitigation Plan and Ownership

### Run 1 update: 2026-09-30

The data team confirmed a common manufacturer; batches and material revisions,
not manufacturer classification, organize the appearance metadata. Reflection
may persist across future revisions. Keep appearance separate from batch/layout.

Run 1 implements JSON schemas under `calibrations/schemas/`, synthetic examples
under `calibrations/examples/masking/`, `MaskingConfig`, validators in
`src/utils/data/region_contracts.py`, and `RegionAnnotationStore`. JSON storage is:

- `calibrations/manifests/boards.json`
- `calibrations/masks/<template_id>/v001.json`
- `calibrations/annotations/<board_id>/<scan_id>.json`
- `calibrations/thresholds/<calibration_id>/` reserved for later calibration

No real metadata, masks, labels, or thresholds were fabricated. Existing inspection
settings, `damaged_segments.json`, and threshold artifacts are unchanged.
Template geometry and rasterization are implemented; local scoring is pending.
Run 3 adds annotation HTTP endpoints; Run 4 adds the mask editor. Run 1 does not
complete the broader Batch 1 below.

Run 1 validation: 86 tests pass (49 new contract/storage checks), with the same
five existing trainer warnings. Ruff and documentation example checks pass.

Version-one board records have `batch_id`, `material_revision`, `appearance_group`,
`layout_id`, explicit split (including `unassigned`), duplicate-scan grouping,
whole-board assurance with evidence/reviewer/date, and optional template assignment.
Use `unknown` for missing metadata. Source template/reference boards require
`guaranteed_good` assurance; this never asserts that all pixels are visible.
Partial review must not be represented as a whole-board guarantee.

Region IDs are unique across each template. `component` and optional `interconnect`
regions are inspection targets; substrate/alignment/exclusion regions are support
regions. The electrical function of a visible line need not be known. Templates
may omit interconnect regions entirely. If included, they require review just like
components; removing them later requires a new template version and explicit label
reconciliation. Do not delete old findings to remove a category.

Labels separate hardware `status` from `visibility`. Omitted targets are unreviewed;
review completion requires explicit status for every target, but can still contain
uncertainty. Defects require a type, uncertainty a reason, and reviewed labels a
reviewer/date and visibility assessment. Segment findings allow damage outside
existing regions. Legacy label migration is implemented by the Run 6 CLI.

Create records in dependency order: save a manifest with null assignments, save a
template, update assignments, then save annotations. Storage validates references,
layouts, geometry, and split separation; saves are atomic per file and template
versions are immutable. Manifest edits that invalidate saved labels are rejected.
Run 6 publishes separate validated annotation snapshots for migration; in-place
multi-file transactions remain unsupported.
Run 4 integrates pixel coverage and physical skip checks into the editor. See README for the working temporary-directory example.

The agreed run sequence is: (1) schemas/storage, (2) rasterization and geometry
mapping, (3) annotation-only APIs, (4) browser editor, (5) regional review UI,
(6) legacy migration/version reconciliation, (7) local calibration/scoring,
(8) inspection integration. Runs 1–8 are implemented; real-data validation remains required.

### Run 2 update: 2026-09-30

`src/utils/data/region_masks.py` provides Boolean rectangle/polygon rasterization,
segment mask iteration, compact template coverage reports, reversed board-grid
mapping, and source/display image-coordinate transforms. Polygons include pixel
centers inside or on edges; adjacent polygons can share boundary pixels. Rectangles
match half-open NumPy slices. All stored geometry remains original-resolution,
zero-based source coordinates. Display transforms use the actual image rectangle
after zoom/pan, excluding letterbox padding; individual images are never rotated.

Exclusions are unioned and removed from component/interconnect targets. Coverage
reports use `valid`, `empty`, and `insufficient_coverage` geometry statuses, not
inspection decisions. No annotated regions means no masks. The default coverage
cutoff is zero, but empty masks are always reported as empty. Mask generation
rejects populated physical skips and substrate overlaps with targets/exclusions.
Call the mask utilities in addition to Run 1 store validation for pixel checks.

`src/utils/viz/mask_viz.py` supplies RGB overlay rendering and labeled preview plots.
Masks are generated one segment/region at a time; callers must avoid collecting a
full corpus of arrays. README contains a synthetic preview example. Run 3 adds
HTTP access and Run 4 adds the editor; regional scoring, thresholds, and inspection
integration remain pending.
Run 2 validation: 116 tests pass (30 new geometry/overlay checks), with the same
five existing trainer warnings. Ruff passes for `src web scripts tests`, and a
full-resolution synthetic overlay renders successfully. Repository-wide Ruff
also reports 41 existing notebook E402 import-order errors. These checks do not
establish reflection mitigation accuracy on real boards.

### Run 3 update: 2026-09-30

`web/app/annotation_app.py` provides annotation-only coordination, exposed under
`/api/annotation/` by the existing server. GET/POST manifest, template, and
annotations endpoints use the Run 1 store and Run 2 coverage checks. Metadata-only
board reads and full-resolution RGB segment/overlay PNGs do not invoke inspection,
checkpoint loading, or calibration validation. The current browser page still
uses the legacy inspection endpoints; Run 4 supplies a separate mask editor page.

Document responses carry a content revision. POST requires the last revision,
or null for creation; stale edits return 409. Immutable template changes require
new versions. A per-service lock serializes reads and writes within the running
server, including revision checks. This does not coordinate external file writers
or multiple server processes. Missing documents return null document/revision,
never synthetic good labels. Existing documents with missing dependencies fail.

Manifest saves validate actual NumPy headers and server data-root confinement.
Templates validate raster coverage/skips; overlays require compatible dimensions
and registered layout. Labels reject skipped positions and retain Run 1 review
semantics. Geometry coverage status is separate from defect status. Body limits
are 4 MiB for annotation routes; duplicate JSON keys are rejected. JSON artifacts
remain in the configured calibration directory; no real records are fabricated.
See README for endpoint names, request payloads, and bootstrap order.

Run 3 validation: 124 tests pass, including eight new HTTP/service tests and the
five existing trainer warnings. Ruff passes for `src web scripts tests`; diff
whitespace checks pass. Synthetic requests verify operation without inference,
stale/concurrent edit rejection, persistence, path/layout/skip validation, full
image resolution and RGB conversion, duplicate keys, and body limits. No live
server or real annotation/calibration write was performed for this run.

### Run 4 update: 2026-09-30

`web/interface/mask-editor.html` is linked from the inspection page and uses only
annotation APIs. It lists registered guaranteed-good reference-role scans, creates
or loads templates from their original source, and preserves reversed board-grid
presentation with source-coordinate region geometry. It has rectangle/polygon
click drawing, integer coordinate entry, draggable draft vertices, zoom/scroll,
region editing/removal, undo/redo, coverage reports, and draft JSON export/restore.
No browser build or new frontend dependency is required.

Saving creates a new immutable version and leaves assignments and existing labels
unchanged. Conflicts retain the local draft. Pending region edits require Apply
before validation/save/export, and navigation/unload guards protect unsaved work.
The UI does not fabricate board assurance or register unknown boards. A reviewed
manifest and existing NumPy reference file are prerequisites. Empty templates and
partial coverage are drafts, never automatic passes.

`POST /api/annotation/preview` validates an unsaved `{document: template}` and
returns coverage without writing. Browser vector overlays use exclusions for
component/interconnect fills; server rasterization remains authoritative. The
regional good/defective review UI is implemented in Run 5; local scoring remains Run 7.
README documents the complete editor workflow. Geometry and editor behavior tests
run with `node tests/test_mask_editor.mjs`; these use a minimal DOM/API harness,
not a real browser rendering engine.

Run 4 validation: 125 Python tests pass with the five existing trainer warnings,
and three Node test groups cover geometry and editor workflows. Python Ruff, JS
module syntax, and diff whitespace checks pass. No browser runtime is installed
here, so real-browser rendering was not verified. No production board records,
labels, thresholds, or template assignments were changed.

### Run 5 update: 2026-09-30

`web/interface/region-review.html` provides annotation-only regional review, linked
from inspection and the mask editor. Boards require manifest template assignments.
The page shows the board and original template reference side by side, retains
source coordinates and reversed grid navigation, and supports zoom, target outline
selection, explicit hardware/visibility labels, and next-unreviewed navigation.
Existing labels display reviewer/date. Omitted labels remain unreviewed regardless
of whole-board assurance. No inference or calibration artifacts are required.

Only component/interconnect targets receive regional defect labels. Defects require
a type, uncertainty a reason, and reviewed statuses a reviewer/date and visibility.
Completion requires all targets explicitly reviewed; uncertainty/obscurity and
defects remain visible in summaries and map markers. Untemplated segments cannot
be completed. Region and additional-finding edits reopen segment review. Geometry
coverage diagnostics never become hardware decisions. Additional findings support
issues outside masks and preserve legacy provenance when edited.

Saves reuse the Run 3 revision contract and validated atomic storage. Stale saves
retain the draft; the reviewer can download the document/revision envelope, reload,
and reconcile manually. Pending edits must be applied before save/download, and
navigation/unload guards protect unsaved work. No automatic merging, label migration,
review-draft import, or production inspection integration is added. Run 6 remains
responsible for migration/version reconciliation. README documents the workflow.

Run 5 validation: 125 Python tests pass with the five existing trainer warnings.
Two new Node review test groups and all three mask-editor test groups pass, along
with Python Ruff, JS module syntax, and whitespace checks. The review tests use a
DOM/API harness for completion rules, explicit labels, findings, save/reload, and
conflict retention; real-browser rendering remains unverified. No production
labels, assignments, or inspection/calibration artifacts were modified.

### Run 6 update: 2026-10-01

`src/utils/data/annotation_migration.py` implements reusable legacy migration and
single-scan template reconciliation plans. `python -m scripts.migrate_annotations`
previews by default. `--apply --output-dir calibrations/migrations/<name>` publishes
a separate validated annotation store through staging and rename. Source stores,
legacy labels, datasets, and weights are never overwritten. Existing destinations
and stale source signatures are rejected. `migration.json` archives full before/
after documents, provenance, and changes. No real dataset migration was run here.

Legacy damaged segments become additional defective findings with path/coordinate
and document-hash provenance. They never imply regional defect localization or
normality for unmarked positions. Importing reviewer/date identifies the import.
Unregistered scans, missing assignments, skipped damage, or contradictory good
assurance block publication; malformed labels and conflicting aliases are rejected.
Identical imports are idempotent; conflicting historical evidence needs resolution.

Reconciliation preserves labels only when identity, source coordinates, full region
definition, template source, and segment support masks are unchanged. Changed,
added, moved, renamed, or removed targets require renewed review; affected segment
completion is reset. Removed/reset defective or uncertain evidence remains visible
as findings, while every original label remains archived. No inferred ID mappings
or automatic evidence deletion are allowed. Layout/shape mismatches are rejected.

The server accepts `--annotation-dir` for opening a snapshot independently of
inspection calibration settings. Stop annotation editing during snapshot creation;
source signatures are checked before publication. Roll back by reopening the old
annotation directory, preserving newer edits separately. No active-store promotion,
multi-process writer coordination, automatic merge, or recalibration is implied.
README contains preview/apply/serve examples. Runs 7–8 now supply local calibration and inspection integration.

Run 6 validation: 141 Python tests pass, including 16 migration checks, with the
five existing trainer warnings. All five frontend test groups pass. Python Ruff,
whitespace checks, and migration/server CLI help checks pass. Synthetic tests
cover repeat imports, conflicts, changed/removed targets, provenance, stale inputs,
failed publication, and the annotation-directory override. No live server or
production migration was run.

### Run 7 update: 2026-10-01

`LocalScoreConfig`, `configs/local_calibration.yaml`, and
`python -m scripts.calibrate_local` add opt-in regional SSIM/AE calibration.
`spatial_error_maps` and `aggregate_error` expose mean/tail-mean scoring.
`regional_scores` applies component/interconnect masks and exclusions, checks
coverage per metric, and excludes SSIM borders and windows touching exclusions.
Diagnostic patch scoring is available separately, without inferred component labels
or threshold reuse. Maps are retained only for the current segment.

Calibration uses explicit appearance metadata and each template's fixed,
guaranteed-good reference-role source. Unknown/mismatched appearance does not route.
Only calibration-role scans enter fitting. Physical-board/duplicate groups stay in
one stage with one deterministic representative scan. Regional thresholds and the
final segment maximum-margin thresholds use disjoint partitions. Clear explicit
good/defective regional labels are required; segment fitting also requires complete
review with no outside-mask findings. Assurance does not invent regional visibility.
Normal-only quantile fallback is opt-in; missing support remains uncalibrated.

Artifacts live under the annotation store's `thresholds/<id>/local.json`. They record
sample counts, scope keys, expected targets, references, disjoint board identities,
exclusions, policies, and provenance. Existing artifacts are not overwritten.
The loader rejects changed score settings, source documents, shapes/skips, or file
signatures. JSON uses content digests; board/checkpoint signatures retain the existing
path/size/mtime limitation. `local_decisions` exposes regional evidence and uncertain
pixel/AE/hybrid decisions when calibration or coverage is incomplete.

Legacy calibration arrays, scalar scoring, checkpoint architecture, and production
inspection behavior are unchanged. No production calibration was run. Multi-reference
aggregation, appearance inference, correction/alignment, saturation quality checks,
and inspection UI integration are not claimed here. Run 8 is the next implementation
step. README documents the data prerequisites, API boundaries, and CLI usage.

Run 7 validation: 151 Python tests pass, including ten new local scoring/calibration
checks, with the five existing trainer warnings. All five frontend test groups,
Python Ruff, whitespace checks, and the new CLI help path pass. Fixtures cover
local defects, SSIM exclusion neighborhoods, empty masks, patch coverage, explicit
normal-only fallback, physical-board separation, reference-role leakage rejection,
uncertain labels, complete synthetic calibration/decisions, and stale artifacts.
No real-board calibration, training, or detection-accuracy claim is made.

### Run 8 update: 2026-10-01

Run 8 completes the agreed eight-run implementation sequence. `InspectionConfig`
adds optional `local_calibration_path` and `annotation_dir`, plus
`local_scoring_path`. With a local artifact configured, `inspect_board` dispatches
through `src/inspection/local_inspection.py`; legacy settings retain their current
whole-segment behavior. The shared server/job config carries these settings, and
annotation APIs use the same configured store. No production artifacts were changed.

Local inspection validates Run 7 provenance before cache reuse or scoring, selects
the template's fixed appearance-compatible reference, streams one segment at a
time, and saves schema-version-two sidecars. Results retain regional scores,
thresholds, coverage, pixel/AE segment decisions, hybrid decisions, uncertainty
reasons, and reference/template/calibration identity. Missing targets, routing,
coverage, or threshold support cannot become an automatic pass. Stale artifacts
reject inspection even with `--force`. Result loading checks decision grids,
counts, skip consistency, and complete segment evidence. Existing flag grids remain
compatible; zero means no calibrated flag, not an OK decision in local mode.

The inspection page shows explicit uncertain cells, authoritative local counts,
regional evidence and thresholds, and optional reference-template mask overlays.
Source coordinates and reversed grid presentation are preserved. The UI explicitly
limits OK decisions to annotated targets and quality checks to mask coverage.
Automatic glare/saturation detection, alignment, appearance inference, multi-reference
aggregation, illumination correction, and held-out accuracy evaluation are not
implemented by this run. The legacy analysis command still evaluates legacy scores.

`python -m scripts.demo_inspection` starts an isolated synthetic server on port 3000
with small explicit test dimensions, generated labels/masks/thresholds, and a constant
reconstruction model. It demonstrates 30 OK, one flagged, one uncertain, and eight
skipped segments. Mask editing and regional review use the same temporary store.
Files and edits disappear on exit; no real labels, boards, or calibration artifacts
are fabricated or replaced. `--check` builds and inspects without starting a server.
This demonstrates software behavior only. README contains demo and real-data opt-in
configuration instructions.

Run 8 validation: 158 Python tests pass with the five existing trainer warnings.
Six frontend test groups (mask editor, regional review, and local result handling),
Python Ruff, JavaScript module syntax, and whitespace checks pass. The synthetic
`--check` command produces the expected counts. Tests cover local persistence,
cache reuse and forced recomputation, stale calibration rejection, fail-closed
sidecar validation, opt-in dispatch, HTTP results/overlays/run requests, and
unregistered-board uncertainty. Server job snapshots preserve local configuration.
HTTP tests use an in-memory socket; real-browser rendering, a live listening server,
real-board calibration, and real defect-detection accuracy were not verified.

### Workflow and held-out evaluation update: 2026-10-14

`web/app/workflow_app.py` and `web/interface/boards.html` add board registration,
verified-reference forms, assignment preview/apply, and workflow readiness. Existing
NumPy headers are checked inside the configured data root. Unknown appearance,
batch, and material metadata is allowed during preparation; reference status still
requires actual evidence, reviewer/date, and explicit whole-board good confirmation.
No physical identity or verification evidence is inferred. Existing identities cannot
be renamed in the form. Registration preserves assignments and the annotation store's
validation/revision contracts.

Assignments without existing labels update the manifest only after preview. Boards
with saved annotations use the existing reconciliation planner and publish a separate
snapshot under `calibrations/migrations/`; active labels and the running store stay
unchanged. Preview tokens are bounded in memory and stale revisions are rejected.
The UI retains form values on conflicts and carries selected-board context into mask
editing, review, and inspection. Readiness reports missing targets, review completion,
visibility, mask geometry, and calibration support; it is never a hardware decision.
All workflow endpoints share the annotation service lock, not a multi-process lock.

`evaluate_local`, exported from `src.inspection`, and `python -m scripts.evaluate_local`
apply fixed local thresholds to explicit evaluation-role boards. Connected physical
board/duplicate identities select one representative scan and cannot overlap other
roles, fitted scans, or references. Only explicit clear good/defective region labels
supply ground truth. Segment metrics additionally require complete clear target
review without outside-mask findings. Reports under `outputs/` include confusion
counts, conditional detection rates, decision coverage, uncertainty, reviewed-target
coverage, appearance/layout groups, exclusions, and input/artifact provenance. Unknown
predictions remain uncertain. Missing reviews yield no accuracy estimate. Evaluation
may save inspection sidecars but never fits or replaces thresholds.

Current artifact provenance includes evaluation annotations; editing them requires
regenerating compatible calibration even though their labels never enter fitting.
No production data, metadata, labels, weights, or thresholds were changed. The existing
untracked baseline manifest was preserved. These additions remain subject to real-data
accuracy validation and the existing limitations on automatic glare/alignment checks.

### Reflection roadmap

The remaining roadmap is not evidence of improved detection accuracy. Keep the current single-reference inspection path
as the default and as an experimental baseline. New reflection-related options
remain disabled until explicitly configured and validated. This plan does not
authorize a long training run or replacement of production calibration artifacts.

Some boards have different surface appearances: reflections expose background
traces and texture more strongly on some boards than on others. A whole-segment
average can therefore rank a good reflective segment as more anomalous than a
segment with a small missing component. Appearance alone does not establish
batch or material identity. The notebook's current standard-deviation reflection
filter selected all 123 boards in the earlier check; do not use it as a reliable
group label or evidence of a particular batch split.

Implement reference selection and local scoring before increasing model size or
retraining. Software development can proceed with small synthetic boards and
explicit test dimensions while the data team prepares real inputs.

### Division of responsibilities

| Work | Software team can implement now | Data team supplies or verifies |
| --- | --- | --- |
| Board manifest | Schema, validators, split loading, duplicate and leakage checks | Board identity, batch/material evidence, appearance and assurance/review status |
| Inspection regions | Mask format, bounds checks, overlays, coordinate transforms | Where components, stable substrate, fiducials, and exclusions are |
| Reference collection | Loading, explicit group routing, fixed aggregation, cache signatures | Verified-good examples representing each appearance and layout |
| Local scoring | Error maps, region/patch aggregation, bounded-memory processing | Reviewed defect examples and useful inspection coverage |
| Calibration | Group/region thresholds, sample counts, missing-calibration handling | Representative normal and damaged calibration boards |
| Image correction | Shared correction functions, bounds, synthetic checks | Acceptable correction settings and real alignment/glare review |
| Training | Group-balanced sampling, augmentation, optional losses | Verified normal training data from both appearances |
| Evaluation | Group-aware reports, experiment configuration, integration tests | Held-out boards and complete review of evaluated regions |

Software readiness means the contracts work end to end on fixtures. It does not
mean the pipeline is robust to real reflection. Production thresholds, parameter
choices, defect recall, and false-positive rates require the prepared data.

### Contracts to implement

#### Board, split, and reference manifests

Keep JSON manifests, templates, and findings in their dedicated `calibrations/`
subdirectories, with schema versions and repository-relative board paths. YAML
experiment settings remain under `configs/`. Run 1 storage consumes the new JSON
contracts; existing inspection/calibration commands do not yet consume them. Preserve current label files
through explicit adapters rather than silently changing their meaning.

Record stable board ID, scan ID/path, duplicate-scan group, batch/material revision
or `unknown`, appearance group or `unknown`, layout ID, whole-board assurance with
evidence/reviewer/date, and split assignment. Track regional review scope separately
in annotations. Separate appearance
from batch/material identity and separate unreviewed from verified-normal. A partly reviewed
board cannot be treated as verified-normal outside its reviewed regions.

Split roles are training, calibration, evaluation, and reference. Keep all scans,
segments, and derived patches of a physical board in one role for the initial
experiment. Validate reference/evaluation separation and reject duplicate leakage.
Freeze evaluation membership before tuning. Include both appearances where data
allows; explicitly report missing coverage instead of inventing balance.

Reference entries identify their board/scan, group, layout, reviewed coverage,
and version. Route by supplied metadata first. Later appearance-based routing
must use reviewed substrate regions, never suspected component-defect scores.
Select the group once per board. Unknown or ambiguous routing yields an explicit
review/uncertainty status; do not search groups for the lowest anomaly score.
Within a group, use a fixed documented reference aggregation rule, such as a
median spatial error map, and compare it with a fixed representative reference.

#### Region masks and coordinate mapping

Each template records schema version, template/layout ID, source board and segment
shape, region IDs, region purpose, source `(row, col)`, and geometry. Supported
purposes are component, optional interconnect, substrate, alignment, and exclusion. Start with rectangles;
add polygons when they materially improve coverage. Use stable component IDs when
available; descriptive region IDs are acceptable otherwise.

Use zero-based source coordinates. Within a segment, `x` increases rightward and
`y` downward. Store rectangles as `[x_min, y_min, x_max, y_max]`, with exclusive
maximum bounds, corresponding to `image[y_min:y_max, x_min:x_max]`. Require
`0 <= x_min < x_max <= width` and `0 <= y_min < y_max <= height`. Validate polygon
geometry, empty regions, shape compatibility, and effective coverage after masking.
An exclusion takes precedence for component scoring; flag empty or inadequately
covered component regions rather than silently accepting them. Substrate regions
must not include expected components or variable stickers.

Masks describe expected component positions even when components are missing.
Reuse templates across boards with the same layout. If alignment is applied,
record the transform and explicitly map regions and overlays between template
and source coordinates. Prefer bounded rigid alignment on stable geometry; do
not use flexible warps that can remove a defect from the comparison.

The UI reverses board positions for presentation only: source `(row, col)` appears
at display `(7 - row, 4 - col)`. Individual images are not rotated. Labels, masks,
HTTP requests, and serialized results retain source coordinates. Existing UI
segment numbers are one-based; annotation tools must clearly identify their
coordinate convention and export zero-based source coordinates.

#### Local scores and calibration

Extend `src/utils/metrics.py` to expose spatial `1 - SSIM` maps and
channel-averaged `abs(sigmoid(reconstruction_logits) - input)` maps. Preserve the
existing scalar scores for baseline comparison. Process one segment at a time;
do not retain all full-resolution maps for a corpus in memory.

Aggregate maps within component regions or configurable fixed overlapping patches
when component templates are unavailable. Patch fallback does not establish
component-level localization. Evaluate predefined aggregations such as regional
mean and upper-tail mean. Record patch size, overlap, valid-pixel policy, SSIM
window/border treatment, and aggregation parameters in the experiment settings.

Calibrate by appearance group, region/position, and score definition. Retain sample
counts, reference identities, split identities, and provenance. Mark insufficient
samples or missing thresholds as uncalibrated; do not silently substitute an
unvalidated threshold from another group. A normal-only quantile fallback may be
configured explicitly, with its sample count and limited evidence recorded.

Calibrate the final segment decision as well as local scores: taking the maximum
or flagging any of many regions changes the false-positive rate. Keep existing
pixel, AE, and hybrid-intersection decisions available for comparison. Local score
changes require fresh calibration and must not reuse the legacy `(8, 5)` arrays
as if they were local thresholds.

#### Preprocessing, quality, and training

Use shared server-side preprocessing for calibration and inference. Estimate
bounded brightness/color corrections only from stable, valid substrate pixels.
Evaluate local contrast normalization and gradient comparisons as separate
experiments. Do not enable aggressive histogram equalization or learned glare
removal by default. Additional trace edges mean gradient comparison alone is not
assumed to solve appearance differences.

Record alignment confidence, visible component coverage, and glare/saturation
checks. Saturation can destroy evidence; uncertain or uninspectable regions must
not be counted or displayed as automatic passes. Numerical quality limits remain
configuration choices requiring real-data validation.

Initially keep the existing AE's input preprocessing unchanged while evaluating
pixel-wise improvements. Changed AE preprocessing must match across training,
calibration, and inference and requires a matching checkpoint and thresholds.
Keep the architecture initially unchanged. Add optional balanced sampling using
supplied groups and bounded brightness, gamma, color, and smooth illumination
augmentation. Use the same augmented tensor as input and reconstruction target.
A corrupted-input/clean-target denoising experiment needs an explicit separate
batch contract; do not silently change `AutoencoderTrainer`'s tensor contract.
Evaluate optional structural/gradient losses alongside reconstruction loss only
after the scoring baseline is established.

#### Results, HTTP, and provenance

Extend `InspectionResults` with a versioned schema carrying group, reference IDs,
region scores/thresholds, calibration status, quality status, and reason codes.
Preserve the meaning of legacy flags: `-1` remains physically skipped. Add a
separate quality/decision representation and update the frontend together so an
uncertain result cannot be rendered or counted as a green pass. Retain available
defect evidence even when another region is uncertain.

Serve local overlays/images on demand. Persist source coordinates and reload
results safely. Cache and calibration signatures must cover manifests, templates,
references, split policy, preprocessing, scoring/aggregation settings, quality
policy, checkpoint, and threshold artifacts. Reject incompatible artifacts clearly.
Keep scripts execution-only, reuse inspection/data/calibration APIs, and keep
notebooks as clients with explicit parameters. Store generated reports/plots under
`outputs/` and immutable training-run snapshots under `logs/`.

### Software implementation batches

#### Batch 1: Contracts and reference handling

Implement schemas, validators, explicit group routing, reference loading,
coordinate overlays, provenance, and cache invalidation. Extend existing config,
data, and inspection modules; do not duplicate logic in notebooks or scripts.

Acceptance checks: reject invalid geometry, conflicting IDs, split leakage, wrong
layouts, and missing required references. A synthetic component defect cannot
change metadata/substrate-based group selection. Verify known source/display
coordinates and round trips through serialization. Defer learned appearance
classification until the data team provides reviewed examples.

#### Batch 2: Local inspection, calibration, and UI

Implement spatial errors, region/patch aggregation, local calibration, final
segment decisions, and explicit uncalibrated/uncertain status. Extend threshold
utilities, analysis, result serialization, and web rendering together.

Acceptance checks: synthetic local defects remain visible in local scores; region
bounds, SSIM borders, exclusions, and patch coverage behave as configured. Missing
calibration or invalid coverage cannot become a pass. Verify API persistence,
overlays, cache refresh, and backward compatibility of the default pipeline.
Fixtures establish behavior, not real detection accuracy.

#### Batch 3: Optional correction and training support

Implement bounded shared correction, alignment/quality hooks, optional balanced
sampling and augmentation, configurable extra losses, and experiment reports.

Acceptance checks: identity settings preserve baseline behavior; synthetic smooth
illumination is handled as designed without numerical errors; known artificial
defects are not used to fit correction. Verify deterministic seeded transforms,
group sampling, matching preprocessing contracts, finite gradients, and bounded
memory. Keep experimental defaults disabled pending real-data validation.

### Data team guide: practical responsibilities

No machine-learning knowledge is needed for these tasks. A "reference" is a
board that a person has checked and considers good in the areas being inspected.
A "mask" is a marked area telling the software where to look. A "calibration"
set is a collection used to choose the error level that should trigger a warning.
An "evaluation" set is a separate collection used only to check the final result.

#### 1. Identify boards and their surface appearance

Use production or assembly records to record the batch and material/layout
revision. All boards are reported to share a manufacturer; no manufacturer
classification task is required. If batch or revision is unavailable, write
`unknown`; do not guess from how bright the board looks. Separately record its appearance using agreed descriptions
such as darker surface or stronger reflection, with example images. Flag ambiguous
cases. Record board ID, file path, layout revision if known, and any repeat scan
of the same board. The software team can provide a spreadsheet/import template;
team members do not need to edit JSON by hand.

#### 2. Select and review reference boards

Choose a few candidate good boards from each appearance group. Check expected
components against assembly information or an agreed known-good example. Check
that the areas needed for inspection are visible and not hidden by strong glare,
cropping, or blur. Record who reviewed each board, when, and which areas were
checked. A low software anomaly score does not prove that a board is good.
If only some segments were reviewed, mark that scope explicitly.

Include normal variation within each appearance group. Do not choose references
only because they happen to match the current model or golden board closely.
Mark uncertain candidates for review rather than certifying them as good.

#### 3. Draw component regions on a reusable template

Choose a reviewed board for each distinct physical layout. Open each selected
segment in an annotation viewer that shows source coordinates. Begin with segments
containing known missing-component examples; the first delivery need not cover
all 32 inspected segments.

For each component or small functional group, draw a rectangle around its expected
position. Include a modest margin for placement variation, pads, solder, and
nearby evidence, but avoid large amounts of unrelated green board surface. Use
names such as `connector_J3` or a descriptive stable name when assembly IDs are
unavailable. Start with rectangles; exact outlines are usually unnecessary.

Keep the rectangle in the same expected place on a defective board even if the
component is absent. For example, a capacitor's inspection region includes the
empty pads left behind when it is missing. Drawing regions only around visible
components would exclude the defects we want to find.

Also mark the following areas, with their separate purposes:

| Area to mark | How to choose it | What it is used for |
| --- | --- | --- |
| Stable bare board surface | Component-free areas without stickers, critical routing, or unstable glare | Estimate appearance and brightness differences |
| Optional interconnect | Visible routing or a bundle to inspect; its electrical function may be unknown | Record local visual findings separately from background |
| Stable geometry | Clear mounting holes or designated alignment marks | Check that component regions line up |
| Exclusions | Variable ID stickers, outside-board background, or intentionally uninspected content | Keep unrelated variation out of component scores |

Do not exclude a difficult or reflective component merely to make scores better.
Flag poor visibility instead. Stickers can be checked separately if required.
Avoid substrate regions containing components: a missing component there could
otherwise change how the software corrects the entire image.

Reuse the template on several other boards of the same layout, including both
appearances. View overlays and check that each rectangle still covers its intended
component. If many rectangles shift together, report an alignment issue to the
software team rather than moving every rectangle on every board. Batches
can share a template when their geometry and expected component population are the same; create separate templates
only for genuine layout differences. The software team owns coordinate conversion
and file-format validation.

#### 4. Record actual defects separately from inspection regions

An inspection region says "look here for capacitor C17." A defect annotation says
"C17 is missing on board X." They are different records. For a damaged board,
record board/scan ID, source segment, affected component/region when known, defect
type, and review notes. A defect rectangle or polygon is helpful when the affected
region is unclear, but existing segment-level labels remain useful.

Record whether evaluated areas were checked and found good, checked and found
damaged, not yet reviewed, or too obscured to decide. An empty list of defects
must not automatically mean that someone checked the board. Existing labels need
an explicit review-completeness field or manifest adapter before using this rule.

#### 5. Check the NumPy files and hand off a small pilot

Once the annotation viewer is implemented, use it with the validation helpers to
check file identity, expected shape,
color display, known segment positions, and missing/cropped content. Compare
component positioning across boards. Report wrong orientation, swapped segments,
repeat scans, unreadable files, or glare-obscured areas with the board ID and an
example coordinate. Do not silently transpose arrays or overwrite originals.
If a file is corrected, preserve its original identity and record the correction
and new version so existing labels can be checked against it.

The first handoff should contain:

- Board IDs/paths, batch/material evidence or `unknown`, appearance and layout labels,
  plus repeat-scan relationships.
- A few reviewed-good reference candidates covering both appearances, with review
  scope and reviewer/date recorded.
- One reusable region template for a small set of segments containing known
  defects, plus overlays checked on multiple boards.
- Reviewed normal and defective examples for those segments, including explicit
  unreviewed/obscured status and defect locations where available.
- A short list of unresolved file, alignment, visibility, or labeling problems.

The software team imports and validates this handoff, checks that overlays agree
with the reviewers, and proposes board-level split assignments for joint review.
Only after the pilot is usable should the team expand annotation to the full
board. Do not require annotation of every pixel or redrawing templates per board.

### Real-data validation after the handoff

Run a controlled sequence: current pipeline; matched references; matched references
plus local scoring; add illumination correction; finally add balanced/augmented
AE training. Tune on calibration data, freeze settings, then evaluate held-out
boards. Do not repeatedly tune on the final evaluation set.

Report defect recall (how many actual defects were found), false-positive rate
(how often good regions were flagged), uncertainty rate, and inspected coverage,
separately for each appearance group and at segment and board level where useful.
Track model size, inference time, and memory. Uncertain cases stay in the report;
do not remove them to make accuracy look better. Compare recall at a fixed
false-positive operating point chosen before final evaluation. Lower average MAE
alone is not an acceptance criterion.

Promote an experimental configuration only after reviewed real examples show
reduced reflection-related false alarms without unacceptable loss of defect
recall or coverage. Document limitations and update commands, notebooks, and
provenance together. No particular preprocessing, loss, patch size, or appearance
classifier is assumed successful before that evaluation.

---

## Development Commands and Testing Expectations

### Environment and quick checks

Run from the repository root. The current local environment is `venv` with Python
3.12.3. For a new Linux environment, use Python 3.12 and the shared requirements;
those pins include CUDA libraries, notebook tools, tests, and Ruff. Dependency
security fixes and a clean install are not validated by the examples themselves.
There is no separate `requirements-dev.txt` in this checkout.

```bash
# New environment only; do not overwrite an existing environment unnecessarily.
python3.12 -m venv venv
source venv/bin/activate
python -m pip install -r requirements.txt
python -m pip check

# Current verification commands; these do not train or recalibrate a model.
PYTHONPATH=. MPLBACKEND=Agg MPLCONFIGDIR=/tmp/hexaboard_mpl OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -m pytest -q
ruff check src scripts web tests scanner
bash -n jobs/main.sh
node --check web/interface/app.js

# Optional focused checks while editing relevant modules.
PYTHONPATH=. python -m pytest -q tests/test_pipeline_contracts.py tests/test_calibration_pipeline.py tests/test_web.py
```

Node is needed only for the JavaScript syntax check, not for serving the UI.
Model tests use full-resolution tensors; loader, trainer, calibration, and HTTP
checks use synthetic/temporary fixtures. The CPU thread limits above bound test
resource use. Report the exact checks run and distinguish passing, warnings,
unrun target-device paths, and data-dependent blockers.

### Generate thresholds and inspect a board

First review checkpoint, model, paths, device, and calibration split settings in
`configs/inspection.yaml`. Existing thresholds must match that configuration; the
local manifest failed compatibility validation during the 2026-09-28 refresh.
Calibration replaces arrays/manifest in the configured destination. Use a copied
YAML and a separate `calibration_dir`/`output_dir` for experiments that should
preserve existing artifacts.

```bash
source venv/bin/activate

# Generate thresholds before using the default inspection workflow.
python -m scripts.calibrate --config-path configs/inspection.yaml

# Optional direct inspection; saves a sidecar beside the board.
python -m scripts.inspect --config-path configs/inspection.yaml \
    --new-hexaboard-path data/bad_example/320XLF4CQH00445.npy

# Start the browser interface at http://127.0.0.1:3000.
python -m scripts.server --config-path configs/inspection.yaml --host 127.0.0.1 --port 3000
```

Open the page, enter a board path within `data_root`, and choose Load Board or Run
Inspection. Loading reuses a current sidecar or computes one; Run Inspection
forces recomputation. Neither action repairs stale calibration automatically.

Use the same reusable APIs in Python:

```python
from src.configs import InspectionConfig
from src.inspection import inspect_board
from src.inspection import calibrate
from web.server import run_server

config = InspectionConfig.from_yaml('configs/inspection.yaml')
calibrate(config)
result = inspect_board(config)
print(result.summary())
run_server(config=config)
```

The last call runs the server until stopped. Notebook experiments should declare
`InspectionConfig` parameters directly in cells rather than loading YAML; see the
notebook conventions below. Do not import `scripts` modules for programmatic use.

### Analysis and reconstruction evaluation

```bash
# Bounded held-out integration check: at most one board from each class.
python -m scripts.analyze --config-path configs/inspection.yaml --max-boards 1

# Full configured held-out analysis, after labels/splits have been reviewed.
python -m scripts.analyze --config-path configs/inspection.yaml

# Reconstruction BCE loss and example plot; no threshold generation.
python -m scripts.evaluate --config-path configs/inspection.yaml --display-segment-idx 0
```

Analysis writes `analysis.json` and `confusion_matrices.png`; reconstruction
evaluation writes `reconstruction.png`, all under `output_dir`. Evaluation's
index is the usable dataset segment index, not a full-grid coordinate. `analyze`
requires compatible thresholds and held-out examples of both classes.

### Training and resume

Inspect arguments without starting a run:

```bash
python -m scripts.train --help
python -m scripts.evaluate --help
python -m scripts.calibrate --help
python -m scripts.analyze --help
python -m scripts.inspect --help
python -m scripts.server --help
```

The following is the actual training entry point, not a smoke test; only run it
when the configured experiment and device budget are appropriate:

```bash
python -m scripts.train --config-path configs/train_CNNAutoencoder.yaml \
    --train-data-dir data/train --val-data-dir data/val
```

For a bounded smoke run, create a separate YAML with reduced epochs and batch size
and use explicitly bounded input files/directories with temporary logging/output
locations. Resume by adding `--checkpoint-path` with an existing trainer checkpoint
under `logs/<model>/checkpoints/`; a best-weight file alone is not resumable state.
After training, update inspection settings to the new compatible checkpoint,
recalibrate, and validate artifact consumers. Do not reuse thresholds from the
previous model.

### Container workflow

The local `docker/Dockerfile` builds an environment image, not a bundled dataset
or application checkout. Build from repo root and mount the checkout for use:

```bash
docker build -f docker/Dockerfile -t hexaboard-inspection .
docker run --rm -it --gpus all --shm-size=2g \
    --user "$(id -u):$(id -g)" \
    --env MPLCONFIGDIR=/tmp/matplotlib \
    --mount "type=bind,source=$(pwd),target=/workspace/visual-inspection" \
    --publish 127.0.0.1:3000:3000 \
    hexaboard-inspection \
    python -m scripts.server --config-path configs/inspection.yaml --host 0.0.0.0
```

The container uses `/opt/venv`; do not activate the mounted host environment.
GPU use needs compatible host drivers and Docker GPU support; omit `--gpus all`
for CPU execution and adjust shared memory for the workload. The container host
binding enables port forwarding; the published host port remains localhost-only.
The image consumes current dependency pins and does not itself resolve their
security advisories. A root build context can include large artifacts unless a
Docker ignore file is supplied; review that before building. Docker is unverified
in this environment, and `docker/` is currently ignored by Git.

### Change validation and review

Add meaningful focused tests when modifying board/color/skip contracts, checkpoint
compatibility, scores and thresholds, masks, source/display coordinates, result
serialization, cache invalidation, labels, HTTP failures, or trainer dispatch and
scheduler timing. Use temporary outputs. Preserve tests that distinguish generic
supervised batches from AE reconstruction tensors. Reflection acceptance checks
are specified in the implementation batches above.

Keep synthetic checks separate from real-data/GPU validation. Do not execute
notebooks merely to update them. Use short sentence-case commit titles and explain
changed behavior, validation, and limitations in PRs. Include screenshots for
UI-visible changes when available. Keep bulky data out of commits; preserve
unrelated working-tree changes and check CI when a workflow exists.

---

## Additional Python Conventions

The portable version of these preferences and Ruff enforcement guidance lives in
[docs/STYLES.md](docs/STYLES.md). Follow it alongside the repository-specific
contracts in this file.

### Formatting and alignment

- Do not align equals signs.
- Do not put an unnecessary trailing comma after the final parameter,
  argument, collection element, or import name. This applies to definitions,
  calls, class instantiations, lists, tuples, dictionaries, sets, and imports,
  whether written on one line or multiple lines. Preserve commas only when
  required to retain syntax or meaning, such as a singleton tuple `(value,)`
  or a singleton tuple index `tensor[index,]`. Multiple-element tuples do not
  need a trailing comma.
- Use four-space indentation and ordinary spacing between tokens; do not
  enforce a fixed line-length limit. Wrap by logical structure and visual
  density, not solely to shorten a line. Keep readable expressions, messages,
  and short conditional assignments together rather than isolating individual
  integers or arguments.
- Keep simple `if` and `elif` conditions compact. When wrapping helps, use
  parentheses and break before each `and` or `or`, preserving logical grouping
  and keeping each individual comparison intact.
  Tensor masks use `&` and `|` with parenthesized comparisons; do not replace
  those operators with Python `and` and `or` during formatting.
- Function, method, and class definitions may keep their parameters on one
  line. If the parameter list is multiline, keep the opening parenthesis on
  the definition line, put every parameter on its own continuation line, and
  put the closing parenthesis on a separate line. This includes `self`, `cls`,
  class bases, and class keyword arguments; put `*` and `/` separators on their
  own lines too. Keep short tuples and other expressions within a parameter
  together; do not split their internal commas into separate parameters.
- Function calls and class instantiations may stay on one line. If a call
  is multiline, keep the opening parenthesis with the callable name, put every
  argument on its own continuation line, and put the closing parenthesis on
  a separate line. This applies even to short related arguments and keyword
  arguments. Keep nested tuples, shapes, and other individual argument
  expressions together when readable; their elements are not separate call
  arguments.
- Expand a crowded inner call without unnecessarily expanding a simple outer
  wrapper; `Model(ModelConfig(...))` may keep both opening calls together
  and close with `))`, with the inner arguments on separate lines. Wrapping
  an outer expression does not require recursively expanding readable nested
  calls, comprehensions, or small dictionaries.
- Keep calls with a few short arguments compact. Wrap calls with many arguments
  or long expressions rather than stretching the one-line allowance; use one
  argument per line and explicit keyword names when they clarify meaning.
  Use the actual callable parameter names. Readability, not a fixed character
  count, determines when to wrap.
- Keep shapes, coordinate tuples, and short matrix rows together. Expand larger
  collections by logical entry, without a trailing comma after the final element.
  Use one key/value pair per line for substantial dictionary records. Lists
  and tuples may group short related elements on a line; give longer expressions
  their own lines, as with compact CSV headers versus complex CSV values.
- For a multiline method chain, keep `.method(` attached to its receiver or
  the preceding call's closing `)` on the same line; do not start a new line
  with `.method`. Place arguments on the following indented line and the
  closing `)` on the next line. Apply this layout to the initial call too.
- Within a multiline chain, put each argument of a multiline call on its
  own continuation line, including short arguments. Do not add a trailing
  comma after the final argument. Continue the chain with `).next_method(`,
  using the same layout for its arguments.
  Argument-free calls such as `.clone()` may stay inline. Short chains that
  fit entirely on one line need not be expanded.
- When a conditional expression contains a crowded call, wrap that call first.
  A short `if ... else ...` suffix may remain attached to its closing parenthesis
  when readable. Do not split simple expressions or introduce temporary variables
  solely to meet a line-length threshold.
- Wrap long message strings with adjacent literals inside parentheses, without
  commas, and preserve spaces between fragments. Formatting must not change a
  string into a tuple or alter the resulting message or report schema.
- Short, related loop or conditional blocks may be grouped without blank lines
  between them, especially consecutive edge-case checks or guard clauses.
  Always leave one blank line after the final block in the group before the
  following statement, including a fallback `return`.
- Leave one blank line after a completed control-flow block (`if`, loops,
  `with`, `try`, or `match`) before a following statement at the same indentation.
  The grouped short-block exception above applies only between related blocks,
  never after the final block. Do not insert blank lines between a block and
  its continuation clauses (`elif`, `else`, `except`, or `finally`), or add
  trailing blank lines merely because a block ends its enclosing scope.

Examples of wrapping long conditions and function calls:

```python
if (
    images.ndim != 4
    or images.shape[1] != 3
    or images.shape[0] != expected_batch_size
):
    raise ValueError("Unexpected segment image dimensions.")

figure = plot_reconstructions(
    y_true=targets,
    y_pred=predictions,
    num_images=4,
    show=False
)
```

Example of a multiline method chain:

```python
images = (
    images.to(
        device=device,
        dtype=torch.float32
    ).permute(
        0,
        3,
        1,
        2
    ).contiguous()
)
```

### Docstrings

- Do not add module-level docstrings or descriptive file-header docstrings.
  Start Python files with their imports (after any required interpreter or
  encoding directive); keep documentation on relevant classes and public APIs.
- In docstrings and user-facing messages, surround tensor/array shapes and
  inline variable references with one backtick on each side, for example
  `[B, T, C, H, W]` and `history_valid_mask`. Do not use double backticks or
  leave shapes unquoted. Apply this to interpolated shapes in error messages
  too. Keep NumPy-style parameter/type declarations unquoted.
- Do not add docstrings to Python files outside `src/`, including scripts
  and tests. This applies to module, class, and function docstrings.
- Inside `src/`, keep NumPy-style docstrings on major reusable classes such as
  trainers and their variants, models, datasets, and configuration objects,
  and on APIs intended for direct use from notebooks or scripts.
- Do not add docstrings to internal helpers, including functions or methods
  whose names start with a single `_`, nested implementation helpers, or
  internal-only functions even when their names have no underscore prefix.
  A public-looking name or package re-export alone does not require a docstring.
- Public entry points implemented through special methods, such as `__init__`,
  `__call__`, and `__getitem__`, may retain docstrings as part of a documented
  class interface. These are not private helpers merely because of their names.
- Put opening and closing triple-double-quote delimiters on separate lines.
  Start the description on the line after the opening delimiter, even for
  a one-line description.
- In NumPy-style parameter declarations, put one space before and after
  the colon: `param_name : Type`.
- Use annotation-style collection names in docstring parameter and return
  fields: `List[Dict]`, `Dict`, `Tuple[Tensor, Tensor]`, and `Sequence[str]`,
  rather than prose such as `list of dict` or `tuple of Tensor`. Keep scalar
  types such as `str`, `int`, and `float` lowercase. Include element types
  where useful and accurate.
- Express optional types and alternatives in docstrings using `, optional`
  and `or`: `List[Tensor], optional`, `Tensor or Tuple[Tensor]`, and
  `str or Path`. Combine them as `Callable or List[Callable], optional`.
  Use this notation in parameter and return fields instead of `Optional[...]`
  and `Union[...]`. Actual Python type annotations retain `Optional[...]`
  and `Union[...]`; this rule changes documentation notation only.
- Do not leave trailing spaces after closing docstring delimiters.
- In functions and methods, start the body immediately after the closing
  docstring delimiter, without a blank line. Leave one blank line after a
  class docstring before its attributes or methods.
- Use ordinary comments where an internal implementation needs explanation.

```python
"""
Description goes here.

Parameters
----------
param_name : Type
    Description.

Returns
-------
ReturnType
    Description.
"""
```

### Quotation rules

- Use single quotes for:
  - dictionary keys
  - stable internal identifiers
  - strict parameter values
  - configuration constants
  - file modes and encodings
- Use double quotes for:
  - error messages
  - print and log messages
  - user-facing text

### Naming

- Use `snake_case` for functions and variables, `PascalCase` for classes, and
  `UPPER_SNAKE_CASE` for constants. Prefer descriptive names.

### Typing rules

- Annotate parameters and return types.
- Do not inline type-annotate local variables inside function bodies.
- Prefer `typing` generics such as `List`, `Dict`, `Tuple`, and `Optional`.
- Prefer `Tensor` instead of `torch.Tensor`, so let's put Tensor in the statement
  `from torch import ...` along with other modules.

### Import conventions

- For imports across folders, import project symbols from the owning package
  namespace, not implementation filenames: use
  `from src.utils.data import open_board`, not
  `from src.utils.data.board_contract import open_board`.
- For imports from a module in the same folder, use its implementation filename
  directly. For example, in `src/engine/ae_trainer.py`, use
  `from .trainer import Trainer`, not
  `from . import Trainer`. Prefer relative imports inside `src/`.
  These rules also apply to scripts, tests, notebooks, and documentation examples.
- Direct imports from explicitly private modules whose filename starts with `_`
  are allowed, for example `from tests._http import request`. An underscore on
  the imported symbol alone does not make its source module private.
- Define package exports in `__init__.py`; implementation-module imports there
  establish the package API for cross-folder consumers. Within the import groups
  below, sort imports alphabetically by module, ignoring case. Within each
  `from ... import ...` statement, order symbols as `CONSTANT -> ClassObject -> func`:
  constants first, classes second, functions third. Alphabetize names within each
  category, ignoring case. This applies to package exports and private symbols too;
  for example, `PreflightSession` precedes `_preflight_result`. Use the symbol's
  actual role when its spelling is ambiguous. Depart from this order only when
  required by a concrete dependency or initialization constraint, such as a
  circular import; document the reason beside the affected imports.
  Same-folder consumers import directly from the defining
  module; cross-folder consumers import from the owning package namespace.
  Do not rename modules solely to evade this rule.
- This rule governs repository modules, not third-party library APIs. Keep
  low-level dependencies acyclic; avoid dynamic imports as a workaround.
- Inspection score functions live in `src/utils/metrics.py` and are exported
  through `src.utils`. Do not reintroduce `inspection_scores.py` or a forwarding shim.
- Group related tests by behavior: region contracts/geometry, annotation
  persistence/API/migration, and local calibration/inspection. Reuse private
  helpers instead of importing another test module; preserve all test coverage.

- Group imports in this order:
  1. standard library
  2. third-party libraries other than PyTorch
  3. PyTorch imports in their own group to make the deep-learning dependency clear
  4. first-party or relative imports, with relative imports ordered by increasing
     depth (fewer leading `.` characters first)
- Separate each import group with a blank line.
- Expand crowded import symbol lists using parentheses and one symbol per line;
  keep short imports compact and preserve the applicable ordering rules.
- Preserve required initialization order, such as selecting a Matplotlib backend
  before importing pyplot. Keep deliberate deferred or type-checking imports
  where they are needed rather than moving them mechanically to module scope.
- Prioritize `import` before using `from` in each import groups.
- Use relative imports inside `src/` packages when practical.
- Use absolute `src...` imports in `tests/` and `scripts/`.
- Do not mutate `sys.path`.

### File organization

- After every work session, include a list of all files edited or added in the
  final response, alongside the usual summary and verification results.
- Update `AGENTS.md` for durable conventions, contracts, or plan changes, and
  `README.md` for repository-wide setup, interfaces, and workflows, not after
  every run. Keep detailed run results in their report artifacts; README is
  an onboarding guide, not an experiment log.
- Do not add incremental smoke-test, diagnostic, or development-run results
  or test-count snapshots to `AGENTS.md`. Keep those in run artifacts. Add a
  concise evaluation milestone only after the required held-out evaluation and
  artifact validation are complete.

- CLI entry points belong in `scripts/`.
- `scripts/` files are execution-only and must not be imported elsewhere.

### Experiment artifacts

- Treat `logs/` and `outputs/` as repository-relative directories.
- Use `outputs/` for visualizations, result tables, final diagnostic reports,
  and decoded prediction results intended for inspection.
- Create output directories only when writing artifacts, not during trainer
  initialization. Tests that export figures must use temporary output paths.
- Use YAML for reusable experiment configuration files consumed by scripts or
  jobs, including training in tmux, with named sections such as `data`, `model`,
  `loss`, and `train`. Notebooks declare parameters directly in their cells.
  JSON remains appropriate for board manifests and resolved run snapshots.
- Store reusable YAML configuration files under `configs/`, grouped by experiment
  when useful. Masking JSON manifests, templates, and regional findings belong
  under `calibrations/`, as requested for this project. Do not put reusable inputs
  in `logs/setup` or a run's logging folder.
- Store training logs, checkpoints, model weights, resolved per-run configuration
  snapshots, manifest snapshots, environment records, and source provenance
  under `logs/`. Shared input configs and immutable run snapshots have different
  roles; retain run snapshots for reproducibility.
- Keep corresponding experiment names and run identifiers in both trees, and
  record the result directory in the run metadata. Resume from the run in
  `logs/`; do not mix resumable state into `outputs/`.

### Visualization helpers

- Public visualization functions display the completed figure with `plt.show()`
  by default. Close it afterward and return None so a notebook does not display
  the same figure twice through its final expression.
- Accept `savefig=None`. A supplied str or Path saves a PNG at 300 DPI before
  display; create its parent directory when necessary. Use `outputs/` for plots.
- Support `show=False` for composition, tests, and headless callers. Return the
  figure (or named figure dictionary) in that mode; the caller owns closing it.
  Composed helpers must not display incomplete nested figures.
- For multi-figure helpers, document how the supplied path generates filenames.
- Leave explicit vertical space between image rows so lower-row titles never
  overlap upper-row images. Account for figure headings in the layout.
- Preserve segment image aspect ratios in comparison plots.

### Training notebooks

- Instantiate Config classes with explicit parameter values in notebook cells
  so readers can see the setup. Do not load experiment parameters from YAML;
  reserve YAML inputs for scripts and jobs. Persist resolved run snapshots
  through the trainer for reproducibility.
- Use the configured segment dimensions and training/validation board folders
  by default. Do not silently reduce these for demonstration convenience;
  synchronize Markdown and code configuration.
- Display results using a named variable as the final cell expression when
  practical, including named DataFrames or Series. Put independently useful
  outputs in separate cells when needed.
- Otherwise use `print()`, including for labeled status messages, multiple
  scalar results, or live training progress. Do not construct anonymous
  dictionaries or artificial summaries merely to avoid `print()`.
- Call visualization helpers directly; avoid redundant `display()`, `savefig()`,
  and `close()` boilerplate around helpers that already handle those operations.
- Keep logging/checkpoint artifacts in `logs/` and exported figures/result tables
  in `outputs/`. Do not execute notebooks merely to edit or statically check them.

### Test invocation

- Run pytest from repo root with `PYTHONPATH=.`
