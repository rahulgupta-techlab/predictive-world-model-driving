import importlib.util
from pathlib import Path

from pwmd.planning.safety import PlannerDecision

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("carla_experiment", ROOT / "scripts/08_run_carla_experiment.py")
experiment = importlib.util.module_from_spec(spec)
spec.loader.exec_module(experiment)


class FakeCarla:
    class VehicleControl:
        def __init__(self, throttle=0.0, brake=0.0, steer=0.0):
            self.throttle, self.brake, self.steer = throttle, brake, steer


def test_safe_predictive_decision_accelerates_toward_cruise_speed():
    decision = PlannerDecision("predictive_candidate", 0.0, 0.0, False, "safe", 2.0, None)
    control, _ = experiment._control_for_decision(FakeCarla, decision, 0.0, 0.0, 6.0)
    assert control.throttle > 0.0
    assert control.brake == 0.0


def test_fallback_decision_applies_full_brake():
    decision = PlannerDecision("fallback_stop", 0.0, 0.0, True, "risk", -0.1, 0.2)
    control, steer = experiment._control_for_decision(FakeCarla, decision, 4.0, 0.2, 6.0)
    assert control.throttle == 0.0 and control.brake == 1.0
    assert steer == 0.0
