from __future__ import annotations
from run_case import (Any, Path, _case_name, _duration, _load_vehicle_specs, _mech_cfg, _rgb, _scene_oblique, _step_size, _times, _vehicle_profile_from_case, _vehicle_track, np)

def _simulate_surrogate_vehicle_lane_change(case: dict[str, Any]) -> dict[str, Any]:
    mech = _mech_cfg(case)
    spec = _load_vehicle_specs(_vehicle_profile_from_case(case))
    vehicle = mech.get("vehicle") if isinstance(mech.get("vehicle"), dict) else {}
    pos0 = np.asarray(vehicle.get("position", [-28.0, 0.0, 0.6]), dtype=float)
    speed = float(np.linalg.norm(np.asarray(vehicle.get("velocity", [14.0, 0.0, 0.0]), dtype=float)))
    dt = _step_size(case)
    times = _times(_duration(case), dt)
    lane_width = float(mech.get("lane_width", 3.5))
    duration = float(times[-1]) if len(times) else 0.0
    tau = np.clip(times / max(duration, 1.0e-6), 0.0, 1.0)
    s_curve = 0.5 * (1.0 - np.cos(np.pi * np.clip((tau - 0.15) / 0.55, 0.0, 1.0)))
    recovery = 0.5 * (1.0 - np.cos(np.pi * np.clip((tau - 0.55) / 0.35, 0.0, 1.0)))
    y = pos0[1] + lane_width * s_curve - 0.15 * lane_width * recovery
    x = pos0[0] + speed * times
    yaw = np.degrees(np.arctan2(np.gradient(y, times, edge_order=1), np.gradient(x, times, edge_order=1)))
    positions = np.column_stack([x, y, np.full(len(times), pos0[2], dtype=float)])
    return {
        "times": times.tolist(),
        "objects": [
            _vehicle_track(name="vehicle", spec=spec, color=_rgb(0.18, 0.49, 0.82), positions=positions, yaw_deg=[float(v) for v in yaw]),
        ],
        "scene": _scene_oblique(_case_name(case, Path(".")), ground_value=0.0),
    }
