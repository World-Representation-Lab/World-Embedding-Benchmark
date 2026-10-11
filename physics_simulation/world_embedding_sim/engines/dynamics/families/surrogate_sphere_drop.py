from __future__ import annotations
from run_case import (Any, Path, _case_name, _circle_track, _duration, _gravity, _mech_cfg, _rgb, _step_size, _times, np)

def _simulate_surrogate_sphere_drop(case: dict[str, Any]) -> dict[str, Any]:
    mech = _mech_cfg(case)
    sphere = mech.get("sphere") if isinstance(mech.get("sphere"), dict) else {}
    radius = float(sphere.get("radius", 0.22))
    mass = float(sphere.get("mass", 1.0))
    pos = np.asarray(sphere.get("position", [0.0, 0.0, radius + 2.0]), dtype=float)
    vel = np.asarray(sphere.get("velocity", [0.0, 0.0, 0.0]), dtype=float)
    restitution = float(mech.get("restitution", 0.65))
    friction = float(mech.get("friction", 0.12))
    dt = _step_size(case)
    times = _times(_duration(case), dt)
    g = _gravity(case)
    positions = np.zeros((len(times), 3), dtype=float)
    spin_y: list[float] = []
    omega = 0.0

    for i, _ in enumerate(times):
        positions[i] = pos
        spin_y.append(float(omega))
        if i == len(times) - 1:
            break
        vel = vel + g * dt
        next_pos = pos + vel * dt
        if next_pos[2] < radius:
            penetration = radius - float(next_pos[2])
            next_pos[2] = radius + restitution * penetration
            vel[2] = abs(float(vel[2])) * restitution
            tangential_damp = max(0.7, 1.0 - 0.22 * friction)
            vel[0] *= tangential_damp
            vel[1] *= tangential_damp
            omega = float(vel[0] / max(radius, 1.0e-6))
        pos = next_pos

    return {
        "times": times.tolist(),
        "objects": [
            _circle_track(
                name="ball",
                radius=radius,
                mass=mass,
                color=_rgb(0.20, 0.71, 0.86),
                positions=positions,
                scalars={"omega_y": spin_y},
            )
        ],
        "scene": {"view_plane": "xz", "ground_value": 0.0, "title": _case_name(case, Path("."))},
    }
