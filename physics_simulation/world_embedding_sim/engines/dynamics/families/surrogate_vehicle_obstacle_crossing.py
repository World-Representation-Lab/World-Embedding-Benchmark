from __future__ import annotations
from run_case import (Any, Path, _box_track, _case_name, _duration, _load_vehicle_specs, _mech_cfg, _rgb, _scene_oblique, _step_size, _times, _vehicle_profile_from_case, _vehicle_track, math, np)

def _simulate_surrogate_vehicle_obstacle_crossing(case: dict[str, Any]) -> dict[str, Any]:
    mech = _mech_cfg(case)
    spec = _load_vehicle_specs(_vehicle_profile_from_case(case))
    vehicle = mech.get("vehicle") if isinstance(mech.get("vehicle"), dict) else {}
    obstacles = mech.get("obstacles") if isinstance(mech.get("obstacles"), list) else []
    pos0 = np.asarray(vehicle.get("position", [-30.0, 0.0, 0.6]), dtype=float)
    speed = float(np.linalg.norm(np.asarray(vehicle.get("velocity", [10.0, 0.0, 0.0]), dtype=float)))
    dt = _step_size(case)
    times = _times(_duration(case), dt)
    positions = np.zeros((len(times), 3), dtype=float)
    yaw = np.zeros((len(times),), dtype=float)
    for i, t in enumerate(times):
        x = pos0[0] + speed * t
        z = pos0[2]
        for obs in obstacles:
            if not isinstance(obs, dict):
                continue
            size = obs.get("size", [0.8, 3.5, 0.12])
            center = obs.get("position", [0.0, 0.0, 0.06])
            half_x = 0.5 * float(size[0])
            local = (x - float(center[0])) / max(half_x, 1.0e-6)
            if abs(local) <= 1.0:
                z += float(center[2]) * (0.5 + 0.5 * math.cos(math.pi * local))
        positions[i] = [x, pos0[1], z]
    objects = [_vehicle_track(name="vehicle", spec=spec, color=_rgb(0.18, 0.49, 0.82), positions=positions, yaw_deg=[float(v) for v in yaw])]
    for idx, obs in enumerate(obstacles):
        if not isinstance(obs, dict):
            continue
        center = np.repeat(np.asarray(obs.get("position", [0.0, 0.0, 0.06]), dtype=float).reshape(1, 3), len(times), axis=0)
        objects.append(_box_track(name=f"obstacle_{idx:02d}", size=obs.get("size", [0.8, 3.5, 0.12]), mass=0.0, color=_rgb(0.44, 0.46, 0.50), positions=center))
    return {"times": times.tolist(), "objects": objects, "scene": _scene_oblique(_case_name(case, Path(".")), ground_value=0.0)}
