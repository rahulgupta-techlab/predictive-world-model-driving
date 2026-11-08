import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_module(name, relative_path):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    # Package-relative imports are exercised through the package path below.
    import sys
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_safety_gate_uses_stop_fallback_for_blocked_path():
    import sys
    import types

    package = types.ModuleType("pwmd")
    package.__path__ = []
    planning_package = types.ModuleType("pwmd.planning")
    planning_package.__path__ = [str(ROOT / "src/pwmd/planning")]
    sys.modules["pwmd"] = package
    sys.modules["pwmd.planning"] = planning_package

    import importlib.util
    risk_spec = importlib.util.spec_from_file_location(
        "pwmd.planning.risk", ROOT / "src/pwmd/planning/risk.py"
    )
    risk = importlib.util.module_from_spec(risk_spec)
    sys.modules["pwmd.planning.risk"] = risk
    risk_spec.loader.exec_module(risk)

    safety_spec = importlib.util.spec_from_file_location(
        "pwmd.planning.safety", ROOT / "src/pwmd/planning/safety.py"
    )
    safety = importlib.util.module_from_spec(safety_spec)
    sys.modules["pwmd.planning.safety"] = safety
    safety_spec.loader.exec_module(safety)

    times, paths = risk.generate_candidate_paths(
        5.0, horizon_s=2.0, dt=0.1, steering_angles_deg=(0,)
    )
    actor = risk.ActorState("blocker", (2.5, 0.0), (0.0, 0.0), radius_m=0.8)
    decision = safety.decide_action(paths, times, [actor], current_speed_mps=5.0)
    assert decision.mode == "fallback_stop"
    assert decision.emergency_brake
    assert decision.target_speed_mps == 0.0


def test_safety_gate_keeps_clear_candidate():
    import sys
    import types

    package = types.ModuleType("pwmd")
    package.__path__ = []
    planning_package = types.ModuleType("pwmd.planning")
    planning_package.__path__ = [str(ROOT / "src/pwmd/planning")]
    sys.modules["pwmd"] = package
    sys.modules["pwmd.planning"] = planning_package

    import importlib.util
    risk_spec = importlib.util.spec_from_file_location(
        "pwmd.planning.risk", ROOT / "src/pwmd/planning/risk.py"
    )
    risk = importlib.util.module_from_spec(risk_spec)
    sys.modules["pwmd.planning.risk"] = risk
    risk_spec.loader.exec_module(risk)

    safety_spec = importlib.util.spec_from_file_location(
        "pwmd.planning.safety", ROOT / "src/pwmd/planning/safety.py"
    )
    safety = importlib.util.module_from_spec(safety_spec)
    sys.modules["pwmd.planning.safety"] = safety
    safety_spec.loader.exec_module(safety)

    times, paths = risk.generate_candidate_paths(4.0, horizon_s=2.0)
    decision = safety.decide_action(paths, times, [], current_speed_mps=4.0)
    assert decision.mode == "predictive_candidate"
    assert not decision.emergency_brake
