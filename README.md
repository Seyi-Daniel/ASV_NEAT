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
| `max_steps`, `step_cost`, `goal_bonus`, `collision_penalty`, `timeout_penalty`, `distance_cost`, `distance_normaliser` | Cost shaping terms for the minimisation objective. |
| `tcpa_threshold`, `dcpa_threshold`, `angle_threshold_deg`, `wrong_action_penalty` | Continuous COLREGs violation penalties (per-step costs when the wrong turn is taken inside the window). |

Invalid overrides raise a friendly error so experiments remain reproducible.

### Scenario-specific overrides

When training all encounter families together you can give each one its own
reward shaping or geometry tweaks. Prefix the hyperparameter with the scenario
name to scope the override:

```bash
# Encourage faster arrivals in crossing encounters without affecting others
python asv_neat/scripts/train.py --hp crossing.goal_bonus=-55

# Apply a harsher collision penalty to overtaking scenarios
python asv_neat/scripts/train.py --hp overtaking.collision_penalty=400
```

The prefixes `crossing.`, `head_on.` and `overtaking.` are supported, and they
can be combined with regular overrides in the same command.

---

## Training workflow

1. Install `neat-python` (and `pygame` if rendering is desired).
2. Launch training (optionally scoping the encounter family):

   ```bash
   # Run only the crossing scenarios with a custom reward shaping profile
   python asv_neat/scripts/train.py --scenario crossing --generations 100 --hp goal_bonus=-50
   ```

   * Every genome is evaluated on all fifteen scenarios using a thread pool so the
     environment dynamics remain independent. Pass `--scenario head_on` or
     `--scenario overtaking` to train against the five head-on or overtaking
     encounters respectively, or keep the default `--scenario all` to evaluate the
     complete 15-scenario suite.
   * Fitness values are set to the **negative** average cost, meaning lower cost
     solutions yield higher NEAT fitness while respecting the minimisation
     framing.
   * The default NEAT config (`configs/neat_crossing.cfg`) already expects 12
     inputs and nine outputs, matching the observation/action definitions.

3. Winners are saved automatically under `artifacts/winners/<scenario>_winner.pkl`
   (for example `artifacts/winners/crossing_winner.pkl`). Override the path with
   `--save-winner custom.pkl` or disable the behaviour with
   `--disable-auto-save`.
4. After evolution the script prints a scenario-by-scenario summary (steps,
   COLREGs penalty, collision status, average cost) and, if `--render` is used,
   replays each encounter via pygame. Without `--render` the fifteen summaries are
  gathered in parallel so the evaluations finish together before printing.

Checkpoints can be enabled with `--checkpoint-dir` and
`--checkpoint-interval` to capture intermediate states during longer runs.

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

---

## Model replay

After training completes you can visualise any saved genome with the dedicated
demo script. This avoids retraining and makes it easy to compare separate
crossing/head-on/overtaking winners:

```bash
python asv_neat/scripts/demo.py --winner artifacts/winners/crossing_winner.pkl --scenario crossing --render
```

The CLI accepts the same hyperparameter overrides as the training script so the
environment, reward shaping, and encounter geometry all match the settings used
when the genome was produced.

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
