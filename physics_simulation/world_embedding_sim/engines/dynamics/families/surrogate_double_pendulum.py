from __future__ import annotations
from run_case import (Any, Path, _capsule_track, _case_name, _circle_track, _duration, _gravity, _mech_cfg, _rgb, _step_size, _times, math, np)

def _simulate_surrogate_double_pendulum(case: dict[str, Any]) -> dict[str, Any]:
    mech = _mech_cfg(case)
    g = abs(float(_gravity(case)[2]))
    L1 = float(mech.get("length_1", 1.0))
    L2 = float(mech.get("length_2", 0.9))
    m1 = float(mech.get("mass_1", 1.0))
    m2 = float(mech.get("mass_2", 1.0))
    r = float(mech.get("bob_radius", 0.07))
    pivot = np.asarray(mech.get("pivot", [0.0, 0.0, 0.0]), dtype=float)
    t1 = math.radians(float(mech.get("initial_angle_1_deg", 35.0)))
    t2 = math.radians(float(mech.get("initial_angle_2_deg", 20.0)))
    w1 = math.radians(float(mech.get("initial_omega_1_deg_s", 0.0)))
    w2 = math.radians(float(mech.get("initial_omega_2_deg_s", 0.0)))
    damping = float(mech.get("damping", 0.003))
    dt = _step_size(case)
    times = _times(_duration(case), dt)

    bob1 = np.zeros((len(times), 3), dtype=float)
    bob2 = np.zeros((len(times), 3), dtype=float)
    rod1_a = np.repeat(pivot.reshape(1, 3), len(times), axis=0)
    rod1_b = np.zeros((len(times), 3), dtype=float)
    rod2_a = np.zeros((len(times), 3), dtype=float)
    rod2_b = np.zeros((len(times), 3), dtype=float)

    for i, _ in enumerate(times):
        p1 = pivot + np.asarray([L1 * math.sin(t1), 0.0, -L1 * math.cos(t1)], dtype=float)
        p2 = p1 + np.asarray([L2 * math.sin(t2), 0.0, -L2 * math.cos(t2)], dtype=float)
        bob1[i] = p1
        bob2[i] = p2
        rod1_b[i] = p1
        rod2_a[i] = p1
        rod2_b[i] = p2
        if i == len(times) - 1:
            break

        delta = t1 - t2
        den = max(1.0e-6, 2 * m1 + m2 - m2 * math.cos(2 * t1 - 2 * t2))
        a1 = (
            -g * (2 * m1 + m2) * math.sin(t1)
            - m2 * g * math.sin(t1 - 2 * t2)
            - 2 * math.sin(delta) * m2 * (w2 * w2 * L2 + w1 * w1 * L1 * math.cos(delta))
        ) / (L1 * den)
        a2 = (
            2
            * math.sin(delta)
            * (
                w1 * w1 * L1 * (m1 + m2)
                + g * (m1 + m2) * math.cos(t1)
                + w2 * w2 * L2 * m2 * math.cos(delta)
            )
        ) / (L2 * den)
        a1 -= damping * w1
        a2 -= damping * w2
        w1 += dt * a1
        w2 += dt * a2
        t1 += dt * w1
        t2 += dt * w2

    return {
        "times": times.tolist(),
        "objects": [
            _capsule_track(name="rod_1", color=_rgb(0.56, 0.58, 0.64), points_a=rod1_a, points_b=rod1_b, radius=max(0.016, 0.22 * r)),
            _capsule_track(name="rod_2", color=_rgb(0.46, 0.48, 0.58), points_a=rod2_a, points_b=rod2_b, radius=max(0.016, 0.22 * r)),
            _circle_track(name="bob_1", radius=r, mass=m1, color=_rgb(0.16, 0.53, 0.78), positions=bob1),
            _circle_track(name="bob_2", radius=r, mass=m2, color=_rgb(0.93, 0.45, 0.18), positions=bob2),
        ],
        "scene": {
            "view_plane": "xz",
            "ground_value": None,
            "title": _case_name(case, Path(".")),
        },
    }
