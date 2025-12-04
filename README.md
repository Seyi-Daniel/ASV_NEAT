# ASV NEAT — COLREGs Crossing Scenario

This repository hosts an experimental NEAT-Python implementation for training a
give-way autonomous surface vessel (ASV) to resolve COLREGs-compliant
encounters. The codebase has been consolidated into the `asv_neat/` project,
which cleanly separates three concerns:

* **Scenario generation** – deterministic construction of fifteen COLREGs
  encounters (five each for crossing, head-on and overtaking) so the stand-on
  vessel always lies within the rule-defined sector and the speed profiles can
  be tuned per situation.
* **Environment simulation** – a lightweight pygame-compatible integrator that
  applies chunked helm commands to simple boat dynamics while tracking both
  vessels’ goals.
* **Evolutionary training** – NEAT-Python integration that evaluates every
  genome across the fifteen scenarios _in parallel_, aggregates the cost
  metrics and feeds the resulting **minimisation** objective back to NEAT.

The give-way vessel receives a 12-dimensional observation vector (position,
heading, speed and goal coordinates for itself and the stand-on craft) and
selects one of nine discrete action combinations (three helm directions × three
throttle commands). Reward shaping encourages quick, collision-free arrivals
while penalising COLREGs violations within a configurable TCPA/DCPA envelope.

---

## Repository layout

```
asv_neat/
├── configs/                # neat-python configuration (input/output counts etc.)
├── scripts/                # CLI entry points for training and scenario previewing
└── src/asv_neat/           # Python package with env, scenarios, NEAT utilities
```

Key modules:

* `src/asv_neat/scenario.py` — deterministic geometry helpers and scenario
  dataclasses (including goal placement offsets).
* `src/asv_neat/env.py` — pygame-ready simulation core that advances both boats
  according to helm/throttle commands while exposing snapshots for NEAT.
* `src/asv_neat/neat_training.py` — parallel evaluation loop, cost function and
  training harness wrapping neat-python.
* `src/asv_neat/hyperparameters.py` — a single source of truth for every
  numeric tunable used by the project, with a CLI-friendly override mechanism.

---

## Hyperparameters

All numeric values live in `HyperParameters` and can be adjusted at runtime. To
inspect them:

```bash
python asv_neat/scripts/train.py --list-hyperparameters
```

Each entry can be overridden on the command line with `--hp NAME=value`. For
example, to test a slower acceleration profile and a tighter COLREGs window:

```bash
python asv_neat/scripts/train.py --hp boat_accel_rate=1.2 --hp tcpa_threshold=60
```

Important groups include:

| Name prefix | Purpose |
|-------------|---------|
| `boat_*`    | Hull geometry and surge acceleration/deceleration limits. |
| `turn_*`    | Chunked helm session behaviour (degrees per command, rate, hysteresis). |
| `env_*`     | Integration settings and render scaling. |
| `scenario_*`| Encounter layout (distance/goal extension) and per-scenario speed profiles. |
| `feature_*` | Normalisation constants applied to the 12-element observation vector. |
| `max_steps`, `evaluation_workers`, `evaluation_executor` | Episode length cap, the maximum parallel scenario workers (defaults to CPU core count with batching; override `evaluation_workers` to pin a specific value), and whether evaluation parallelism uses threads (default) or processes. |
| `step_cost`, `goal_bonus`, `collision_penalty`, `timeout_penalty`, `distance_cost`, `distance_normaliser` | Cost shaping terms for the minimisation objective. |
| `tcpa_threshold`, `dcpa_threshold`, `angle_threshold_deg`, `wrong_action_penalty` | Continuous COLREGs violation penalties (per-step costs when the wrong turn is taken inside the window). |

Invalid overrides raise a friendly error so experiments remain reproducible.

---

## Training workflow

1. Install `neat-python` (and `pygame` if rendering is desired).
   * Optional GPU acceleration: installing [CuPy](https://docs.cupy.dev/en/stable/install.html)
     (e.g. `pip install cupy-cuda12x`) will offload the core kinematics and
     geometry math to the GPU automatically. When present the trainer prints the
     selected numeric backend before evolution begins.
2. Launch training (optionally selecting a specific encounter family):

   ```bash
   # Crossing-only run with a custom reward shaping term.
   python asv_neat/scripts/train.py --scenario-kind crossing --generations 100 --hp goal_bonus=-50
   ```

   * Every genome is evaluated on all fifteen scenarios using a batched pool,
     capping workers at your CPU core count (or `--hp evaluation_workers`)
     to avoid oversubscription while keeping encounter dynamics independent.
     Use threads by default or switch to processes with
     `--hp evaluation_executor=process` to bypass the GIL on CPU-bound runs.
     GPU-backed setups (`cupy`) default to multi-worker evaluation but emit a
     warning; if your CUDA stack stalls under thread fan-out, set
     `--hp evaluation_workers=1` to force serial execution.
   * Fitness values are set to the **negative** average cost, meaning lower cost
     solutions yield higher NEAT fitness while respecting the minimisation
     framing.
   * The default NEAT config (`configs/neat_crossing.cfg`) already expects 12
     inputs and nine outputs, matching the observation/action definitions.

3. The winning genome is saved by default to `winners/<scenario>_winner.pkl` so
   it can be replayed immediately. Override the location with `--save-winner`
   if you prefer a custom path. The demo (`scripts/demo_winner.py`) and LIME
   explainer (`scripts/lime_explain.py`) both look in the same directory when no
   `--winner` path is supplied.
4. After evolution the script prints a scenario-by-scenario summary (steps,
   COLREGs penalty, collision status, average cost) and, if `--render` is used,
   replays each encounter via pygame. Without `--render` the fifteen summaries are
   gathered in parallel so the evaluations finish together before printing.

Checkpoints can be enabled with `--checkpoint-dir` and
`--checkpoint-interval` to capture intermediate states during longer runs. You
can also archive the top genomes for every species and generation by pointing
`--species-archive` at a directory (defaults to `winners/species_archive`). The
archive stores up to `--species-top-n` pickled genomes (default 3) per species,
along with the NEAT config used to train them.

---

## Scenario visualisation

Use the helper script to inspect the deterministic encounters and render random
helm/throttle sequences. By default every crossing, head-on and overtaking case
is replayed in series, but you can focus on a specific family via
`--scenario`:

```bash
python asv_neat/scripts/crossing_scenario.py --render --duration 40 --seed 42
```

CLI flags accepted by the preview script:

| Flag | Description |
|------|-------------|
| `--render` | Enable the pygame viewer so that each scenario is visualised in sequence. |
| `--duration <seconds>` | Number of simulated seconds to replay when `--render` is enabled (default `35`). |
| `--seed <int>` | Seed for the random give-way policy that drives the placeholder helm/throttle inputs. |
| `--scenario {all,crossing,head_on,overtaking}` | Limit the preview to a single encounter family (default `all`). |
| `--hp NAME=VALUE` | Repeatable overrides for any hyperparameter exposed by `HyperParameters`. |

Hyperparameters such as the encounter distances or the six per-scenario speed
settings can be adjusted
here too via `--hp` overrides, ensuring the preview matches the training
configuration. During rendering the give-way and stand-on destinations are now
drawn as colour-coded markers so you can confirm that each vessel has its own
goal beyond the shared crossing point.

**HUD overlay:** when the pygame window is open you can press `H` to toggle the
right-aligned HUD that lists each vessel’s position, heading and speed along
with the TCPA/DCPA estimates. The preview script starts with the HUD hidden so
`H` is the quickest way to bring it into view during a render session.

### Replaying saved winners

Use `scripts/demo_winner.py` to visualise a previously trained genome. By
default it looks for `winners/<scenario>_winner.pkl` so the file emitted by a
scenario-specific `train.py --render` session can be replayed immediately:

```bash
python asv_neat/scripts/demo_winner.py --scenario-kind crossing --render
```

All hyperparameter overrides accepted by the training CLI are available here as
well, ensuring the demo uses the exact same environment conditions as the run
that produced the saved genome.

---

## LIME explanation workflow

The project ships with an end-to-end LIME pipeline (`scripts/lime_explain.py`)
for analysing the controller’s per-step decisions. It couples a small update to
`simulate_episode` (a trace callback hook) with a scikit-learn style wrapper
around the NEAT network so the `lime` package can call `predict_proba` just like
it would on a conventional classifier.

### Prerequisites

1. Install `lime` alongside the existing training dependencies:

   ```bash
   pip install lime
   ```

2. Produce a `winner.pkl` via `scripts/train.py --save-winner` or grab the
   auto-saved file emitted by a rendered training/demo session. Ensure the
   matching NEAT config (defaults to `configs/neat_crossing.cfg`) is available so
   the genome can be re-instantiated.

### Capturing traces

`simulate_episode` now accepts `trace_callback`, which is invoked once per step
with the 12-element observation vector, the raw network outputs, the arg-maxed
action, and the full agent/stand-on state snapshots. The LIME script passes a
recorder that caches these structures verbatim before the environment advances
to the next tick, guaranteeing that the explanation phase sees the exact values
used by the controller.

### Running the explainer

```
python asv_neat/scripts/lime_explain.py \
  --scenario-kind crossing \
  --winner winners/crossing_winner.pkl \
  --output-dir lime_reports \
  --hp max_steps=400  # optional overrides so the replay matches training
```

For every deterministic scenario of the selected encounter family the script:

1. Rebuilds the NEAT network from the pickle/config pair.
2. Replays the encounter without rendering while recording the trace via the new
   callback.
3. Computes summary metrics (steps, collision status, goal distance, COLREGs
   penalties, etc.) and their aggregate cost.
4. Feeds the captured feature matrix through `lime.LimeTabularExplainer` using
   the built-in feature names and action labels, yielding a local explanation for
   every time-step.

### Output artefacts

Results are grouped per scenario inside the requested `--output-dir`:

```
lime_reports/
└── 01_crossing/
    ├── metadata.json      # scenario geometry + per-episode metrics
    ├── trace.json         # ordered list of recorded features/states/actions
    ├── lime_step_000.json # individual per-step explanations (one file each)
    └── lime_summary.json  # array of all per-step explanations in order
```

Each explanation records the selected action, the nine-way softmax probabilities
returned by the NEAT network wrapper, the normalised feature values presented to
the controller, and the LIME-attributed weights for those features. This makes
it straightforward to line up a specific action choice with its causal inputs,
repeatable across all steps (`max_steps`) and across five canonical encounters
per scenario type.

### SHAP explanation workflow

For a global/stepwise view of feature importance, a parallel SHAP pipeline is
available via `scripts/shap_explain.py`. It reuses the same trace capture hooks
as the LIME script but feeds the feature matrix through `shap.KernelExplainer`
so every timestep receives SHAP values, plots and a stitched animation alongside
the environment render frames.

#### Prerequisites

Install `shap` in the same environment used for training/demo:

```bash
pip install shap
```

#### Running the explainer

```bash
python asv_neat/scripts/shap_explain.py \
  --scenario-kind crossing \
  --winner winners/crossing_winner.pkl \
  --output-dir shap_reports \
  --hp max_steps=400  # optional overrides so the replay matches training
```

Artefacts mirror the LIME layout with SHAP-specific filenames:

```
shap_reports/
└── 01_crossing/
    ├── metadata.json       # scenario geometry + per-episode metrics
    ├── trace.json          # ordered list of recorded features/states/actions
    ├── shap_step_000.json  # per-step SHAP attributions (one file each)
    ├── shap_summary.json   # array of all SHAP explanations in order
    └── explanation_animation.gif  # combined render + bar plot per step
```

---

## Cost function overview

The minimisation objective combines several components:

* **Step cost** (`step_cost × steps`) guarantees that slower solutions accrue
  higher penalties (1,000 steps → cost 1,000 by default).
* **Goal bonus** (negative value) rewards fast arrivals, while a configurable
  timeout penalty and distance term apply when the goal is missed.
* **Collision penalty** adds a fixed cost when separation falls below the
  collision distance.
* **COLREGs penalty** accrues `wrong_action_penalty` each step the agent chooses
  a non-starboard helm while the encounter falls within the specified TCPA,
  DCPA and bearing thresholds.

These terms can all be tuned through the shared hyperparameter interface to
support different research experiments without touching the core code.
