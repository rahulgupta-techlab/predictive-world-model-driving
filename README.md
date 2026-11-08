# Predictive World Model for Autonomous Driving

A research-oriented prototype exploring whether predicting surrounding actors' future motion can improve candidate-path selection and support an introspective safety fallback.

> **Status:** Software components are being assembled. The repository includes CARLA data collection, actor-centric trajectory preparation, constant-velocity/LSTM/Transformer forecasting baselines, kinematic candidate-path scoring, and a rule-based fallback gate. No local CARLA run or empirical model/planner evaluation has yet been reported.

## Research question

Can an autonomous-driving agent improve safety in long-horizon and unexpected scenarios by predicting possible future states, evaluating candidate actions, and inspecting its predictions before acting?

## Current implementation

1. **Data collection:** CARLA synchronous collector records ego state, nearby actor state and front-camera images. The ego vehicle uses Traffic Manager autopilot; this is a collection baseline, not a learned driving policy.
2. **Trajectory dataset:** actor tracks are converted to heading-aligned local XY windows. Splits are assigned by episode to reduce leakage.
3. **Forecasting baselines:** constant-velocity evaluation plus PyTorch LSTM and Transformer training scripts.
4. **Prediction inference:** reusable checkpoint loader supports both learned baselines and returns future XY trajectories.
5. **Candidate planning:** constant-steering bicycle-model paths are scored against either constant-velocity actor extrapolations or supplied predicted actor futures.
6. **Safety gate:** a configurable rule-based gate accepts a candidate or requests a stop fallback when clearance/TTC thresholds are violated.

The repository now includes a CARLA experiment harness with three modes: Traffic Manager autopilot, constant-velocity prediction plus candidate planning, and learned LSTM/Transformer prediction plus the same fallback gate. Planner candidates are generated around a lane-waypoint heading reference so equally safe candidates prefer continuing along the current lane. A synchronized scene assembler converts recent CARLA actor tracks into the actor-centric input convention used during training, then transforms forecasts into the ego frame. In learned mode, the controller requests a stop while any visible actor lacks a complete synchronized history, rather than silently omitting that actor. The harness is implemented but has not been run against a local CARLA server in this development session. The current risk model does not model road boundaries, traffic rules, full vehicle dynamics, uncertainty, or calibrated prediction confidence. The fallback is not safety-certified; treat planner modes as simulator research experiments only.

## Repository structure

```text
.
├── .github/workflows/tests.yml
├── configs/
│   ├── collection.yaml
│   ├── planner.yaml
│   └── prediction.yaml
├── data/
│   ├── processed/
│   └── raw/
├── experiments/results/
├── scripts/
│   ├── 00_check_environment.py
│   ├── 01_check_carla.py
│   ├── 02_collect_scenarios.py
│   ├── 03_prepare_trajectory_dataset.py
│   ├── 04_evaluate_constant_velocity.py
│   ├── 05_train_lstm.py
│   ├── 06_train_transformer.py
│   ├── 07_evaluate_learned_forecaster.py
│   ├── 08_run_carla_experiment.py
│   └── 09_compare_carla_runs.py
├── src/pwmd/
│   ├── planning/
│   │   ├── risk.py
│   │   └── safety.py
│   └── prediction/
│       ├── forecasters.py
│       ├── coordinates.py
│       └── scene.py
└── tests/
    ├── test_planning_risk.py
    ├── test_predicted_trajectory_risk.py
    ├── test_prediction_coordinates.py
    ├── test_prediction_scene.py
    ├── test_carla_comparison.py
    ├── test_safety_gate.py
    └── test_trajectory_pipeline.py
```

## Environment

Target environment:
- Windows 11, Python 3.10
- CARLA 0.9.15 server and matching Python API
- 32 GB RAM and an NVIDIA RTX GPU (exact model/VRAM still to be confirmed)
- CUDA-enabled PyTorch installed separately using the official selector: https://pytorch.org/get-started/locally/

Create an environment and install base dependencies:

```powershell
py -3.10 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Install the CARLA wheel matching the server release and Python version. Example path (adjust to the local extraction):

```powershell
pip install "C:\CARLA_0.9.15\PythonAPI\carla\dist\carla-0.9.15-cp310-cp310-win_amd64.whl"
```

Install a compatible PyTorch build from the official selector. Start the CARLA server and check the environment:

```powershell
python scripts/00_check_environment.py
python scripts/01_check_carla.py
```

## Data collection and preparation

Collect at least three distinct episodes to create independent train/validation/test splits:

```powershell
python scripts/02_collect_scenarios.py --scenario mixed_traffic --episodes 3 --seconds 60
python scripts/03_prepare_trajectory_dataset.py --obs-steps 8 --pred-steps 12 --sample-stride 5
```

The defaults represent 2 seconds of observations and 3 seconds of future trajectory at 4 Hz (assuming the collector's 20 Hz simulation rate and stride 5). If fewer than three episodes are available, all windows remain in the training split; do not treat them as held-out evaluation data.

Episode folders contain actor/ego CSV logs, metadata and sampled RGB images. Raw and processed datasets are excluded from Git by default. Do not commit large datasets, CARLA binaries, credentials or trained weights.

## Forecasting baselines

Evaluate constant velocity on the test split:

```powershell
python scripts/04_evaluate_constant_velocity.py --split test --dt 0.25
```

Train the LSTM and Transformer baselines (PyTorch required):

```powershell
python scripts/05_train_lstm.py --epochs 40
python scripts/06_train_transformer.py --epochs 60 --d-model 64 --heads 4 --layers 2
```

Evaluate a trained checkpoint on the held-out test split:

```powershell
python scripts/07_evaluate_learned_forecaster.py --checkpoint experiments/results/transformer.pt --split test
```

Use `--checkpoint experiments/results/lstm.pt` to evaluate the LSTM. The evaluator checks that the dataset window dimensions match the checkpoint. Metrics are ADE, FDE and miss rate; they are not evidence of model quality until the data and split composition have been inspected and the experiment has actually run.

## Connecting predictions to the planner

The reusable inference API is in `src/pwmd/prediction/forecasters.py`; `coordinates.py` converts actor-centric forecasts into the ego frame. The planner accepts one future XY array per surrounding actor, sampled on the planner time grid:

```python
from pwmd.prediction.forecasters import load_forecaster, predict
from pwmd.prediction.coordinates import actor_local_to_ego_frame
from pwmd.planning.risk import generate_candidate_paths
from pwmd.planning.safety import decide_action_from_predictions

model, device, metadata = load_forecaster("experiments/results/transformer.pt")
local_futures = predict(model, device, observed_actor_windows)
# These values must come from one synchronized scene.
actor_futures = [
    actor_local_to_ego_frame(future, actor_position_ego[i], actor_yaw_relative_rad[i])
    for i, future in enumerate(local_futures)
]
times, candidate_paths = generate_candidate_paths(speed_mps=8.0, horizon_s=3.0, dt=0.25)
decision = decide_action_from_predictions(
    candidate_paths, times, actor_futures,
    current_speed_mps=8.0,
    actor_radii_m=[1.0] * len(actor_futures),
)
```

This illustrates the software interface only. The example assumes the observed windows, current ego-relative actor positions and relative headings have been assembled from one synchronized scene. Inputs must share the planner time grid and correct actor radii. The `SceneAssembler` implementation is in `src/pwmd/prediction/scene.py`; it has unit tests for temporal gaps, actor-local history windows and ego-frame transforms. Its behavior still needs validation against real CARLA frame/timestamp data before interpreting closed-loop results.

## Closed-loop CARLA comparison

Run the three modes separately with the same CARLA map, seed, duration and NPC count. The harness records collision-sensor events, distance travelled, mean speed, minimum predicted clearance, fallback interventions and controller decisions. It writes a JSON summary and (for planner modes) a per-frame CSV.

```powershell
# Start CARLA 0.9.15 first, then run from the repository root.
python scripts/08_run_carla_experiment.py --mode autopilot --seed 42 --seconds 60
python scripts/08_run_carla_experiment.py --mode constant_velocity --seed 42 --seconds 60
python scripts/08_run_carla_experiment.py --mode learned --checkpoint experiments/results/transformer.pt --seed 42 --seconds 60

# Compare matching-seed summaries.
python scripts/09_compare_carla_runs.py experiments/results/carla_autopilot_seed42.json experiments/results/carla_constant_velocity_seed42.json experiments/results/carla_learned_seed42.json
```

Repeat all three modes with additional identical seeds (for example 42, 43 and 44); compare modes within each seed before aggregating results. The comparison script rejects mixed seeds/maps/CARLA versions and duplicate modes. A learned checkpoint must have 8 observed samples and 12 future samples; the planner interval is 0.25 seconds to match 20 Hz collection sampled every 5 frames. Start with short, supervised simulator runs. This controller is experimental and is not suitable for a real vehicle.

## Evaluation plan

| Dimension | Candidate metrics |
|---|---|
| Forecasting | ADE, FDE, miss rate |
| Safety | Collision rate, near-collision rate, minimum TTC |
| Driving performance | Route completion, travel time |
| Comfort | Acceleration and jerk |
| Introspection | Risk recall, false-alarm rate, uncertainty calibration |
| Runtime | Inference and planning latency |

The next research milestone is empirical validation: run the harness, inspect the generated logs, verify scenario comparability, and report repeated-seed results. Safety comparisons must account for distance travelled and mean speed so that a system that simply stops is not automatically considered better.

## Tests

```powershell
pytest -q
```

GitHub Actions runs the lightweight unit tests without requiring CARLA or PyTorch. A green unit-test run would validate only the tested software contracts, not CARLA integration or driving safety.

## Research integrity

The repository distinguishes implemented functionality from planned experiments. No performance results are claimed until the corresponding experiments have been run. Earlier research planning is not represented as earlier software implementation.

## License

No open-source license has been added. Standard copyright restrictions apply until a license is selected.
