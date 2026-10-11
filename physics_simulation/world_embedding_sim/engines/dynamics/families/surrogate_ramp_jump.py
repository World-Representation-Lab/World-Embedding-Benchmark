from __future__ import annotations
from run_case import (Any, Path, _case_name, _duration, _gravity, _load_vehicle_specs, _mech_cfg, _rgb, _step_size, _times, _vehicle_profile_from_case, _vehicle_track, math, np)

def _simulate_surrogate_ramp_jump(case: dict[str, Any]) -> dict[str, Any]:
    mech = _mech_cfg(case)
    spec = _load_vehicle_specs(_vehicle_profile_from_case(case))
    vehicle = mech.get("vehicle") if isinstance(mech.get("vehicle"), dict) else {}
    ramp = mech.get("ramp") if isinstance(mech.get("ramp"), dict) else {}
    pos0 = np.asarray(vehicle.get("position", [-34.0, 0.0, 0.6]), dtype=float)
    speed = float(np.linalg.norm(np.asarray(vehicle.get("velocity", [18.0, 0.0, 0.0]), dtype=float)))
    angle = math.radians(float(ramp.get("angle_deg", 12.0)))
    ramp_start = float(ramp.get("start_x", -6.0))
    ramp_length = float(ramp.get("length", 5.0))
    ramp_height = float(ramp.get("height", 1.2))
    dt = _step_size(case)
    times = _times(_duration(case), dt)
    g = abs(float(_gravity(case)[2]))
    takeoff_speed = speed * 0.94
    takeoff_x = ramp_start + ramp_length
    takeoff_z = pos0[2] + ramp_height
    t_takeoff = max(0.1, (takeoff_x - pos0[0]) / max(speed, 1.0e-6))
    positions = np.zeros((len(times), 3), dtype=float)
    yaw = [0.0 for _ in times]
    for i, t in enumerate(times):
        if t <= t_takeoff:
            s = np.clip((pos0[0] + speed * t - ramp_start) / max(ramp_length, 1.0e-6), 0.0, 1.0)
            x = pos0[0] + speed * t
            z = pos0[2] + s * ramp_height
            positions[i] = [x, pos0[1], z]
        else:
            tau = t - t_takeoff
            x = takeoff_x + takeoff_speed * math.cos(angle) * tau
            z = takeoff_z + takeoff_speed * math.sin(angle) * tau - 0.5 * g * tau * tau
            if z < pos0[2]:
                z = pos0[2]
            positions[i] = [x, pos0[1], z]
    return {
        "times": times.tolist(),
        "objects": [_vehicle_track(name="vehicle", spec=spec, color=_rgb(0.20, 0.50, 0.82), positions=positions, yaw_deg=yaw)],
        "scene": {"view_plane": "xz", "ground_value": 0.0, "title": _case_name(case, Path("."))},
    }
