from __future__ import annotations
from run_case import (Any, Path, _capsule_track, _case_name, _circle_track, _duration, _gravity, _mech_cfg, _rgb, _step_size, _times, math, np)

def _simulate_surrogate_pendulum(case: dict[str, Any]) -> dict[str, Any]:
    mech = _mech_cfg(case)
    g = abs(float(_gravity(case)[2]))
    L = float(mech.get("length", 1.0))
    r = float(mech.get("bob_radius", 0.08))
    mass = float(mech.get("mass", 1.0))
    pivot = np.asarray(mech.get("pivot", [0.0, 0.0, 0.0]), dtype=float)
    theta = math.radians(float(mech.get("initial_angle_deg", 25.0)))
    omega = math.radians(float(mech.get("initial_angular_velocity_deg_s", 0.0)))
    damping = float(mech.get("damping", 0.005))
    dt = _step_size(case)
    times = _times(_duration(case), dt)

    bob_positions = np.zeros((len(times), 3), dtype=float)
    rod_a = np.repeat(pivot.reshape(1, 3), len(times), axis=0)
    rod_b = np.zeros((len(times), 3), dtype=float)

    for i, _ in enumerate(times):
        bob = pivot + np.asarray([L * math.sin(theta), 0.0, -L * math.cos(theta)], dtype=float)
        bob_positions[i] = bob
        rod_b[i] = bob
        if i == len(times) - 1:
            break
        alpha = -(g / max(L, 1.0e-6)) * math.sin(theta) - damping * omega
        omega += dt * alpha
        theta += dt * omega

    return {
        "times": times.tolist(),
        "objects": [
            _capsule_track(name="rod", color=_rgb(0.55, 0.57, 0.63), points_a=rod_a, points_b=rod_b, radius=max(0.018, 0.24 * r)),
            _circle_track(name="bob", radius=r, mass=mass, color=_rgb(0.92, 0.36, 0.27), positions=bob_positions),
        ],
        "scene": {
            "view_plane": "xz",
            "ground_value": None,
            "title": _case_name(case, Path(".")),
        },
    }
