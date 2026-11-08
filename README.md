# Predictive World Model for Autonomous Driving

A research-oriented portfolio project exploring whether an autonomous-driving agent can improve safety in long-horizon and unexpected scenarios by predicting possible future states, evaluating candidate actions, and inspecting its own predictions before acting.

> **Project status:** Research prototype under active development. The repository now includes CARLA collection utilities, trajectory dataset preparation, constant-velocity and LSTM forecasting baselines, an optional Transformer forecaster, and a transparent kinematic candidate-path risk scorer. These components have not yet been validated in a local CARLA run or evaluated on a collected dataset.

## Research question

Can an autonomous-driving agent improve safety in long-horizon and unexpected scenarios by predicting possible future states, evaluating candidate actions through simulation, and inspecting its own predictions before acting?

## Planned research workflow

1. **Simulation and data collection** — collect synchronized ego-vehicle, surrounding-actor, and front-camera observations in CARLA.
2. **Dataset and baselines** — prepare temporal samples and establish constant-velocity prediction baselines.
3. **World model** — compare the LSTM and Transformer trajectory forecasting baselines.
4. **Predictive planning** — generate kinematic candidate paths and score them against constant-velocity actor extrapolations. This is a first risk-scoring baseline, not a closed-loop MPC controller.
5. **Introspection and safety gate** — add calibrated uncertainty, model disagreement, and a tested fallback policy; these remain future work.
6. **Evaluation** — compare safety, progress, comfort, prediction quality, and runtime, including ablations.

## Initial implementation

The current starter includes:
- CARLA client/server and environment checks.
- A synchronous CARLA data-collection script.
- Front RGB camera capture and ego/nearby actor state logging.
- YAML configuration and a Git ignore policy that excludes collected datasets.

The collector currently uses CARLA Traffic Manager autopilot for the ego vehicle. It is a data-collection baseline, **not** a learned controller or a safety-validated autonomous-driving system. The initial mixed-traffic setup does not yet implement scripted edge-case scenarios.

## Repository structure

```text
.
├── configs/
│   └── collection.yaml
├── data/
│   ├── processed/
│   └── raw/
├── experiments/
│   └── results/
├── scripts/
│   ├── 00_check_environment.py
│   ├── 01_check_carla.py
│   ├── 02_collect_scenarios.py
│   ├── 03_prepare_trajectory_dataset.py
│   ├── 04_evaluate_constant_velocity.py
│   └── 05_train_lstm.py
├── tests/
│   └── test_trajectory_pipeline.py
├── .gitignore
└── requirements.txt
```

## Target development environment

- Windows 11
- Python 3.10
- CARLA 0.9.15 server and matching Python API
- NVIDIA RTX GPU; exact model and VRAM determine feasible simulation and model scale
- 32 GB system RAM
- CUDA-enabled PyTorch installed separately using the official selector

## Setup

1. Download and extract CARLA 0.9.15.
2. Create and activate a Python 3.10 virtual environment:

   ```powershell
   py -3.10 -m venv .venv
   .venv\Scripts\Activate.ps1
   ```

3. Install Python dependencies:

   ```powershell
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. Install the CARLA Python wheel that matches the server release and your Python version. For the Windows CPython 3.10 CARLA 0.9.15 package, adjust the path to your extracted CARLA directory:

   ```powershell
   pip install "C:\CARLA_0.9.15\PythonAPI\carla\dist\carla-0.9.15-cp310-cp310-win_amd64.whl"
   ```

5. Install a compatible CUDA-enabled PyTorch build from the official selector: https://pytorch.org/get-started/locally/
6. Check the local environment:

   ```powershell
   python scripts/00_check_environment.py
   ```

7. Start the CARLA server, then verify the client connection:

   ```powershell
   python scripts/01_check_carla.py
   ```

8. Collect a short initial dataset:

   ```powershell
   python scripts/02_collect_scenarios.py --scenario mixed_traffic --episodes 3 --seconds 60
   ```

The CARLA server and Python client versions must match. Confirm the exact wheel path and compatibility with your local installation before installing.

## Collected data

Each episode is written under:

```text
data/raw/<scenario>/<episode_id>/
├── ego.csv
├── actors.csv
├── metadata.json
└── rgb/
```

- `ego.csv`: ego pose, velocity, acceleration, and applied control.
- `actors.csv`: nearby vehicle and pedestrian states.
- `rgb/`: front-camera images sampled according to `camera_stride`.
- `metadata.json`: seed, map, versions, and collection configuration.

Raw and processed datasets are excluded from Git by default. Do not commit large datasets, CARLA binaries, credentials, or model weights.

## Trajectory forecasting pipeline (new starter modules)

After collecting **at least three distinct episodes**, prepare a trajectory dataset:

```powershell
python scripts/03_prepare_trajectory_dataset.py --obs-steps 8 --pred-steps 12 --sample-stride 5
```

This creates actor-centric, heading-aligned XY windows under `data/processed/`. Episode-level splitting is used to reduce leakage between overlapping windows. With fewer than three episodes, the script deliberately keeps all samples in the training split; collect more episodes before attempting held-out evaluation.

Evaluate the constant-velocity baseline:

```powershell
python scripts/04_evaluate_constant_velocity.py --split test --dt 0.25
```

Train the optional LSTM baseline (requires a separately installed PyTorch build):

```powershell
python scripts/05_train_lstm.py --epochs 40
```

The LSTM checkpoint and metrics are starter outputs, not evidence of a successful experiment until the scripts have been run on collected data. Review dataset size, split composition, and metrics before drawing conclusions.

Run unit tests for coordinate transforms and metric calculations:

```powershell
pytest -q
```

Train the Transformer alternative:
 
```powershell
python scripts/06_train_transformer.py --epochs 60 --d-model 64 --heads 4 --layers 2
```

The prediction configuration is in `configs/prediction.yaml`. The planner configuration is in `configs/planner.yaml`. The risk-scoring module in `src/pwmd/planning/risk.py` generates constant-steering bicycle-model paths and scores them against constant-velocity actor extrapolations. It is a transparent kinematic baseline; it does not yet consume learned multi-agent predictions, optimize a full MPC objective, enforce road boundaries or traffic rules, model uncertainty, or control a CARLA ego vehicle.

The current forecasting scripts use actor position tracks from CARLA logs. They do not yet include camera-based detection, multi-agent interaction modelling, calibrated predictive uncertainty, or an introspective safety fallback.

## Planned evaluation

| Area | Metrics |
|---|---|
| Trajectory prediction | ADE, FDE, miss rate |
| Safety | Collision rate, near-collision rate, minimum time-to-collision |
| Driving performance | Route completion, travel time |
| Comfort | Acceleration and jerk |
| Introspection | Risk recall, false-alarm rate, uncertainty calibration |
| Runtime | Inference and planning latency |

Safety comparisons should account for route progress so that a system that simply stops is not automatically considered better. Metrics and scenarios will be finalized before reporting experimental results.

## Research integrity

This repository distinguishes implemented functionality from planned work. Experimental results will be reported only after the corresponding experiments have been run. Any reconstruction of earlier research planning will be described as planning, not as earlier software implementation.

## License

No open-source license has been added yet. Until a license is selected, standard copyright restrictions apply.
