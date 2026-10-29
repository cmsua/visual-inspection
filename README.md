# UA CMS: Hexaboard Visual Inspection

A deep learning-based visual inspection system for detecting defects in hexaboard segments from the High Granularity Calorimeter (HGCAL) detector of the CMS experiment at CERN.

## Description

The High Granularity Calorimeter (HGCAL) is a key component of the CMS detector upgrade for the High Luminosity Large Hadron Collider (HL-LHC). HGCAL consists of silicon sensors arranged in hexagonal modules called hexaboards, which provide unprecedented spatial resolution for particle detection in the forward region of the CMS detector.

![Cutaway diagram of CMS detector (retrieved from https://cds.cern.ch/record/2665537/files/)](assets/cms_160312_02.png)
*Cutaway diagram of CMS detector (retrieved from <https://cds.cern.ch/record/2665537/files/>)*

Hexaboards are critical silicon sensor modules that form the active detection layers of the HGCAL endcap calorimeter. These hexagonal-shaped boards contain arrays of silicon pad sensors that measure the energy deposits from electromagnetic and hadronic showers. Each hexaboard must meet strict quality standards, as defects can significantly impact the detector's performance in measuring particle energies and positions with high precision.

This project implements an automated visual inspection system that combines:

- **Autoencoder-based anomaly detection**: A convolutional autoencoder trained on reference images to detect reconstruction anomalies
- **Pixel-wise comparison**: Traditional image comparison using Structural Similarity Index Measure (SSIM) between baseline and test images
- **Method-specific flags**: Pixel and autoencoder flags are shown separately; hybrid flags indicate their intersection

## Setup

The current environment was validated with Python 3.12.3, PyTorch 2.13.0,
TorchVision 0.28.0, and NumPy 1.26.4. Use a compatible CPU or CUDA PyTorch build.

```bash
python -m venv venv
source venv/bin/activate
python -m pip install -r requirements.txt
```

Run module commands from repository root. Activate the environment before
starting the server so its inspection subprocess uses the same interpreter.
The frontend is plain HTML/CSS/JavaScript served by Python; no Node build is needed.

## Calibrate and start inspection

Edit `configs/inspection.yaml` to select the checkpoint, baseline, calibration
folders, labels, and device. The current saved `run_01.pt` uses 64 latent channels,
128 initial filters, and `[2, 2, 2]` blocks. The training YAML is a separate
experiment configuration. Set `inspection.model: {}` to infer CNN dimensions
from another checkpoint, or explicitly set matching architecture constraints.

```bash
# Generate new thresholds and their provenance manifest.
python -m scripts.calibrate --config-path configs/inspection.yaml

# Start the UI at http://127.0.0.1:3000.
python -m scripts.server --config-path configs/inspection.yaml
```

Open the page, enter a processed board path inside the configured `data_root`,
and choose **Load board** for manual review without inference or calibration.
Unscored segments show **Not inspected**. **Run inspection** forces a fresh run via
`jobs/main.sh` on POSIX, using the server's Python and YAML. Windows uses the
same inspection CLI directly.

The folder/board picker beside the path controls defaults to `data/bad` (or the
configured data root when no `bad` subfolder exists). Enter another folder within
`data_root` and click **List boards**. The dropdown lists NPY files recursively,
with subfolder names to distinguish files. Selecting one updates the path and
loads it for manual review; it does not run inference. Folder names are just file
locations, not training splits or defect labels. Use **List boards** again after
adding files. Empty folders and missing paths are reported in the picker.


The shared Python entry points are:

```python
from src.configs.inspection_config import InspectionConfig
from src.inspection.calibration import calibrate
from web.server import run_server

config = InspectionConfig.from_yaml('configs/inspection.yaml')
calibrate(config)
run_server(config=config)
```

For a single inspection without the browser:

```bash
python -m scripts.inspect --config-path configs/inspection.yaml \
    --new-hexaboard-path data/bad_example/320XLF4CQH00445.npy
```

The server and command save `<board>.inspection.json` beside each processed
`.npy` file. It contains `pw_flags`, `ae_flags`, `hybrid_flags`, board metadata,
and source signatures. Status values are `-1` skipped, `0` OK, and `1` flagged.
Results are reused only while the board, baseline, checkpoint, thresholds, mask,
and model settings match. Changing the baseline, weights, or mask requires
recalibration. Existing threshold files without `manifest.json` must be regenerated.

Calibration, inspection, analysis, and reconstruction evaluation now take one
`--config-path`; their former separate model/path arguments have been replaced
by the shared YAML. Training retains its training configuration and data arguments.

## Data and calibration contract

Stored boards must be BGR `uint8` tensors shaped `(8, 5, 1060, 1882, 3)`.
Production loading checks the full shape and dtype. Segment extraction converts
to RGB and normalizes to `[0, 1]`. Dataset loading accepts a single file or a
directory, validates all boards, and memory-maps each board instead of copying
it for every segment. Smaller fixtures must explicitly supply their shape.

The canonical `calibrations/skipped_segments.json` defines eight skipped corners,
leaving 32 inspected positions. Grid coordinates are zero-based and traverse
rows first. The image tensor remains on the server; the UI requests one PNG
segment at a time.

The UI minimap reverses both board axes, matching notebook 04: source `(row, col)`
appears at display `(7 - row, 4 - col)`. This rotates the segment arrangement
180 degrees without rotating individual images. Navigation follows display order;
labels, flags, image requests, and the displayed segment numbers retain source
coordinates (segment numbers in the UI are one-based).

Calibration scores each normal board/coordinate once and uses labeled anomalous
segments from a deterministic subset of `bad_dir`. Defaults use `data/train`
and `data/val` for normal calibration and half the annotated bad boards, selected
with seed 42. `data/test` and remaining bad board identities are held out for
analysis. Keep board identities unique across acquisition splits.

- Pixel score: `1 - SSIM(reference, inspected)`.
- Autoencoder score: `MAE(sigmoid(model(inspected)), inspected)`.
- Flagging: score greater than or equal to the position's threshold.
- Hybrid: intersection of the two methods.
- Positions with both normal and defect examples use F1 threshold selection.
- Positions without defect examples use the configured normal-score quantile,
  nudged upward so an equal normal score is not flagged. This fallback still
  needs enough normal boards to estimate a useful operating point.

Threshold arrays and a provenance manifest are written to `calibration_dir`.
Do not use calibration performance as held-out model quality. Low reconstruction
loss alone does not establish defect-detection accuracy.

## Evaluate and analyze

```bash
# Confusion matrices over all unmasked positions on held-out boards.
python -m scripts.analyze --config-path configs/inspection.yaml

# Limit each evaluation class for an integration check.
python -m scripts.analyze --config-path configs/inspection.yaml --max-boards 1

# Reconstruction loss and an example image, without retaining every image in RAM.
python -m scripts.evaluate --config-path configs/inspection.yaml --display-segment-idx 0
```

Reports and plots go to the configured `output_dir`, normally `outputs/inspection`.
Analysis excludes calibration board IDs and the reference board, including copies
in different directories. Its label contract treats unmarked positions as normal;
complete damage labeling is therefore necessary for meaningful metrics.

## Training

```bash
python -m scripts.train --config-path configs/train_CNNAutoencoder.yaml \
    --train-data-dir data/train --val-data-dir data/val
```

Use `--checkpoint-path` for a resumable trainer checkpoint. Missing/incompatible
checkpoints fail instead of silently starting another run. Autoencoder training
uses a tensor as both input and reconstruction target; the generic trainer keeps
its supervised pair interface. Both share scheduling, validation, callbacks,
checkpointing, distributed reductions, and graceful interruption.

Set `scheduler_interval` to `optimizer_step`, `epoch`, or `validation` explicitly.
The training YAML uses `epoch`; `ReduceLROnPlateau` requires `validation`.
With step-based validation, the pasted base trainer also validates at each epoch
boundary when that boundary was not already evaluated. The first Ctrl-C finishes
an update and saves resumable state; a second forces interruption.

Logs, weights, checkpoints, and resolved run metadata go under `logs/`.
Figures go under `outputs/`. Exact shuffled mid-epoch resume determinism and
multi-GPU behavior still require separate validation. Do not start a long run
until the data, calibration, and short training checks pass on the target device.

## Server API

| Endpoint | Purpose |
| --- | --- |
| `GET /api/config` | Configured initial board path |
| `GET /api/board?path=...` | Shape, labels, skips, and inspection grids |
| `GET /api/segment?path=...&row=...&col=...` | One RGB segment as base64 PNG |
| `POST /api/label` | Persist `{path, row, col, damaged}` |
| `POST /api/run` | Force inspection for `{path}` |

Only `.npy` boards inside `data_root` are accepted. Labels and segment coordinates
are bounds-checked. The supported operational input is a preprocessed board;
scanner acquisition and cropping remain hardware integration work under `scanner/`.

## Codex / HEPTAPOD preflight chat demo

This backend connection demo lets Codex call HEPTAPOD's existing MCP preflight
tool and explain the result. It does not add a browser chat tab yet. The existing
inspection UI and whole-segment pipeline are unchanged, and no HEPTAPOD source
changes or toolbase activation commands are needed.

Requirements: Linux, an installed Codex with `app-server`, a working Codex login,
and HEPTAPOD's `visual-inspection` branch with its existing environment and toolbase
configuration. App Server is experimental; this adapter follows the locally
generated protocol schema. Recheck the connection after Codex/toolbase upgrades.
See the [Codex App Server protocol documentation](https://learn.chatgpt.com/docs/app-server)
for the thread, turn, and event interface.
The MCP launcher reuses the `toolbase._toolkit_host` entry point exercised by
HEPTAPOD's served-tool tests; it is an internal toolbase interface, not a promise
of compatibility with every future release.

Review `configs/agent_preflight.yaml` first. Paths resolve relative to that YAML,
including the optional board override. The example points to the sibling
HEPTAPOD checkout and its existing `main-demo-v1` copied dataset/configuration.
It does not create that dataset. Adapt the paths to your installation. The
`state_config` is the existing toolbase HEPTAPOD YAML containing
`visual_inspection_*` settings, not the example CLI's differently shaped
`bridge.json`. Its upstream repository, interpreter, read roots, and writable
roots are preserved. Runtime diagnostics are redirected to this repository's
`outputs/agent_preflight/`, without modifying the source state file. Keep the
inspection YAML consistent with those upstream paths and authorized read roots.

Run from the visual-inspection repository root:

```bash
# Uses the existing Codex authentication; does not print credentials.
codex login status

# Start Codex and discover the one exposed MCP tool; no model conversation.
venv/bin/python -m scripts.agent_preflight --check-connection

# One real preflight call, then one short follow-up using its captured evidence.
venv/bin/python -m scripts.agent_preflight \
    --follow-up "Based only on the captured checks, what prevents inspection, if anything?"
```

The second command tests adapter/MCP readiness. The last command uses the model
configured for your Codex login unless the YAML specifies one; model turns use
low reasoning effort. Codex credentials and ordinary Codex runtime state remain
managed by Codex. The UI below also keeps those credentials server-side.
Conversation text and tool evidence are sent to the configured model provider;
board tensors and weights are not attached to chat requests.

Only `VisualInspectionPreflight` is served to the agent, with explicit approval
for that already-requested diagnostic operation. `InspectBoard` and
`ReadInspectionResult` are not exposed by this preflight-only CLI. General shell execution,
plugins, apps, web search, and multi-agent features are disabled for this process,
and the thread requests a read-only sandbox. Unexpected approvals or permission
requests are declined and fail the run. This does not change global Codex settings.
MCP workers are separate processes; their allowed behavior comes from the served
preflight implementation and HEPTAPOD's own path policy, not the Codex sandbox.

Preflight may load the checkpoint to check compatibility, but does not perform
inference or repair inputs. It writes diagnostic artifacts, not board sidecars,
calibration, or labels. A tool envelope with `status: ok` can contain
`result.ready: false`; the CLI reports that Boolean from actual tool evidence.
An assistant answer without a successful, correctly targeted preflight result is
an error, even if the prose claims success. Human review remains necessary before
treating any later inspection flags as physical defects.

Each run prints its evidence directory, containing:

- `run.json`: connection status, model/thread identity, validated preflight
  evidence, answers, and failure reason if applicable.
- `events.jsonl`: completed assistant messages, observed MCP calls/results,
  turn status, and denied approval events; hidden reasoning is not recorded here.
- `codex.stderr.log`, `toolbase.log`, and `heptapod/`: retained runtime and worker
  diagnostics. Do not publish these blindly; they may contain local paths/data.

Exit codes: `0` for connection-only success or a ready preflight, `2` for a
completed preflight with blockers, `1` for setup/protocol/tool failure, and `130`
for Ctrl-C. A blocked preflight can still demonstrate successful agent/tool
integration. Ctrl-C and timeouts interrupt the turn and clean up the local
process tree, including HEPTAPOD workers in separate sessions. Diagnostic writes
already completed are retained. Do not automatically retry a failed run.

Configuration is exported from `web.configs` (`AgentConfig`); the session API
is exported from `web.agent` (`PreflightSession`).
Within a session, `ask()` runs the initial readiness request and
`ask(..., require_preflight=False)` answers from captured evidence. The constructor
accepts an event callback and cancellation event for the web adapter. One
session permits one preflight call and must be used sequentially by one owner.
Threads are ephemeral: follow-ups work while the session is open, but persistent
conversation restoration is provided separately by the web adapter below.

## Inspection and Agent chat demo

Launch both tabs with the existing Codex login and HEPTAPOD checkout:

```bash
codex login status
venv/bin/python -m scripts.server --agent-config configs/agent_chat.yaml
# Open http://127.0.0.1:3000
```

Manual inspection uses `configs/inspection.yaml` by default, including this
repository's `data/` directory. Override it with `--config-path` when needed.
The agent YAML configures the separate HEPTAPOD connection; it no longer replaces
the manual inspection configuration or restricts it to the HEPTAPOD demo folder.
The adapter supplies a per-request HEPTAPOD runtime configuration for this
repository and the selected inspection YAML, with sidecar writes confined to the
configured data root. It does not modify HEPTAPOD's source or saved configuration.
HEPTAPOD's existing path, calibration, model, and write checks still apply.

1. Enter a repository-relative path such as `./data/bad_example/320XLF4CQH00445.npy`
   and click **Load board**. Images and manual labels load without model inference
   or calibration checks. Segments without scores display **Not inspected**.
2. Use **Run inspection** when automated scores are needed. This still requires
   compatible calibration and writes the usual inspection sidecar.
3. Open **Agent chat** and ask for a summary. The conversation starts automatically
   for the selected board. Changing the board path starts a separate conversation
   on the next message, so results from different boards are not mixed.
4. Chat shows the assistant's response and a compact strip of overall statistics.
   Existing saved results are supplied as evidence; their current freshness is
   unverified. If there are no results, run inspection from the Inspection tab.
   Explicit commands such as **Inspect board 445**, **Please inspect board 445**,
   or **Recompute board 445** now authorize the corresponding operation. Ordinary
   questions still only interpret results. You can also use a full filename or
   repository-relative path. ID lookup searches all subfolders under the manual
   inspection configuration's `data_root`. Duplicate IDs are rejected with matching
   paths; specify the full path to select one. Natural variants such as **Inspect
   the board with ID 67** and **Could you please inspect the board with an ID of
   67?** also work. **Inspect a board with an ID higher than 500** chooses the
   lowest available ID strictly above 500. Comparisons also accept above/over,
   below/under/lower than/less than, at least/at most, and `>`, `>=`, `<`, `<=`.
   Selection among several IDs requires “a board” or “any board”; a request for
   “the board” remains ambiguous if several IDs match. Duplicate scans of the
   selected ID still require a full path. The activity display names the actual
   selected file. These are supported request forms, not unrestricted language
   understanding; unsupported descriptions never silently select another board.

During an inspection request, a green activity dot and operation description show
progress in chat. The Inspection tab shows the same job status and disables its
load/run buttons while the job is active. On success it displays the saved result
without another inference call. Blockers, cancellation, and failures remain visible;
no result is presented as a completed inspection when readiness failed. The activity
line describes the operation; it is not a shell command executed by the model.
HEPTAPOD already provides the required inspect/preflight/read tools, so no upstream
implementation or handoff prompt is needed.

The chat panel has no tool dashboard, artifact browser, or review controls. Manual
review stays in the Inspection tab. Enter sends a message; Shift+Enter inserts a
line break. The composer starts at one line, grows to six lines, then scrolls. **New chat** clears the current view, and **Stop** cancels a response.
Transcripts and diagnostic evidence remain saved server-side. The browser does
not currently expose a conversation-history picker.

The board grid retains its reversed arrangement and unrotated segment images.
Agent coordinates are zero-based source coordinates; viewer numbers are one-based.

The browser talks only to this local Python server. The server starts Codex App
Server, which connects to a small MCP gateway implemented here. That gateway
exposes only the tools allowed for the selected action and binds board/config/
force/result arguments server-side. It allows at most one call per tool per
request, and inspection requires successful readiness in that request. The gateway
calls HEPTAPOD's existing tools using HEPTAPOD's Python environment. The gateway's
`mcp` imports therefore resolve in `heptapod_python`, not necessarily the UI's
Python environment. An editor using only this repository's interpreter may flag
those imports even when the gateway works. The current HEPTAPOD environment supplies
MCP 2.2.0; no additional MCP installation in the UI environment is required.
Verify the configured gateway interpreter with:

```bash
../heptapod/venv/bin/python -B -c 'from mcp import Client, StdioServerParameters; from mcp.server.mcpserver import MCPServer; print("MCP imports OK")'
```

Neither an
assistant message nor model-chosen arguments can grant new write permissions.
Shell, general file editing, extra MCP servers, apps, and web tools are disabled.

Conversation text, tool evidence, and local paths are sent to the Codex model
provider using your existing login; board images/tensors and weights are not
attached. The UI renders basic Markdown (bold, italics, lists, headings, code, tables, and HTTP(S) links) and a thinking indicator,
not hidden reasoning. Raw HTML remains inert text; LaTeX is not rendered. Model usage counts toward the configured Codex account.
Each request uses low reasoning effort and the configured timeout (30 minutes
maximum); this is not a spending cap.

Only one agent operation or inspection may execute at a time. **Stop** cancels
the active request and cleans up its worker processes; already completed writes
remain and captured evidence is retained when available. Timeouts and failures
are shown without automatic retries. Server shutdown stops active jobs; jobs do
not resume after restart. Transcripts persist, while recent job status is held in
memory. Conversation/history files are local and are not an authenticated audit
service. This demo is loopback-only, with origin/host checks; do not expose it as
a shared multi-user server.

Artifacts are stored under `outputs/agent_chat/`:

- `conversations/<id>.json`: messages, registered evidence, and snapshot-specific
  human decisions including reviewer and timestamp.
- `<run_id>/`: Codex event/run records and HEPTAPOD diagnostic artifacts.
- `reports/<id>.txt`: readable summaries derived from validated snapshot grids.
- `manual_labels.json`: ordinary UI labels in agent-enabled mode. This separate
  UI store leaves the calibration configuration and HEPTAPOD labels unchanged.

Conversation and report writes do not alter HEPTAPOD source/configuration. The
normal inspection tool may still create or refresh its configured board sidecar.
For real boards, independently validate detection accuracy and calibration; a
working chat demo does not establish defect-detection performance.

Validation commands:

```bash
venv/bin/python -m pytest -q
node tests/test_agent_chat.mjs
node --check web/interface/app.js
node --check web/interface/chat.js
venv/bin/ruff check src scripts web tests
```

The UI test uses a DOM/API harness, not a browser rendering engine. Live local
validation exercised Codex readiness and the HTTP preflight → cached inspection →
saved-result read → image/evidence retrieval path. Forced real inference and
real-browser rendering were not exercised in that smoke test.

## Tests and conventions

```bash
PYTHONPATH=. python -m pytest -q
python -m pip install -r requirements-dev.txt
ruff check src scripts web tests scanner
```

Full-resolution CNN/ResNet tests exercise shape and gradient behavior. Trainer,
loader, threshold, serialization, and HTTP tests use synthetic fixtures and
temporary artifacts. Notebooks are clients of repository helpers; editing them
does not require executing training cells.

Follow [docs/STYLES.md](docs/STYLES.md) for quote choice, type annotations, grouped
imports, NumPy API docstrings, multiline layouts, and Ruff guidance. `AGENTS.md`
defines repository-specific contracts. Plot helpers accept `savefig` and `show`;
`show=False` returns the figure for the caller to close. `save_fig` remains a
compatibility alias. CLI modules are execution-only and are never imported by
application code.
