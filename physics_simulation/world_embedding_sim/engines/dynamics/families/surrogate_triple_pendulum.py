from __future__ import annotations
from run_case import (Any, Path, _capsule_track, _case_name, _circle_track, _duration, _gravity, _mech_cfg, _rgb, _step_size, _times, math, np)

def _simulate_surrogate_triple_pendulum(case: dict[str, Any]) -> dict[str, Any]:
    mech = _mech_cfg(case)
    g = abs(float(_gravity(case)[2]))
    L1 = float(mech.get("length_1", 0.95))
    L2 = float(mech.get("length_2", 0.78))
    L3 = float(mech.get("length_3", 0.62))
    m1 = float(mech.get("mass_1", 1.0))
    m2 = float(mech.get("mass_2", 0.9))
    m3 = float(mech.get("mass_3", 0.8))
    r = float(mech.get("bob_radius", 0.06))
    pivot = np.asarray(mech.get("pivot", [0.0, 0.0, 0.0]), dtype=float)
    t1 = math.radians(float(mech.get("initial_angle_1_deg", 36.0)))
    t2 = math.radians(float(mech.get("initial_angle_2_deg", 24.0)))
    t3 = math.radians(float(mech.get("initial_angle_3_deg", 12.0)))
    w1 = math.radians(float(mech.get("initial_omega_1_deg_s", 0.0)))
    w2 = math.radians(float(mech.get("initial_omega_2_deg_s", 0.0)))
    w3 = math.radians(float(mech.get("initial_omega_3_deg_s", 0.0)))
    damping = float(mech.get("damping", 0.003))
    dt = _step_size(case)
    times = _times(_duration(case), dt)

    bob1 = np.zeros((len(times), 3), dtype=float)
    bob2 = np.zeros((len(times), 3), dtype=float)
    bob3 = np.zeros((len(times), 3), dtype=float)
    rod1_a = np.repeat(pivot.reshape(1, 3), len(times), axis=0)
    rod1_b = np.zeros((len(times), 3), dtype=float)
    rod2_a = np.zeros((len(times), 3), dtype=float)
    rod2_b = np.zeros((len(times), 3), dtype=float)
    rod3_a = np.zeros((len(times), 3), dtype=float)
    rod3_b = np.zeros((len(times), 3), dtype=float)

    for i, _ in enumerate(times):
        p1 = pivot + np.asarray([L1 * math.sin(t1), 0.0, -L1 * math.cos(t1)], dtype=float)
        p2 = p1 + np.asarray([L2 * math.sin(t2), 0.0, -L2 * math.cos(t2)], dtype=float)
        p3 = p2 + np.asarray([L3 * math.sin(t3), 0.0, -L3 * math.cos(t3)], dtype=float)
        bob1[i] = p1
        bob2[i] = p2
        bob3[i] = p3
        rod1_b[i] = p1
        rod2_a[i] = p1
        rod2_b[i] = p2
        rod3_a[i] = p2
        rod3_b[i] = p3
        if i == len(times) - 1:
            break

        # Lightweight chained surrogate dynamics with gravity + nearest-link coupling.
        a1 = -(g / max(L1, 1.0e-6)) * math.sin(t1) - damping * w1 + 0.08 * math.sin(t2 - t1)
        a2 = -(g / max(L2, 1.0e-6)) * math.sin(t2) - damping * w2 + 0.11 * math.sin(t1 - t2) + 0.10 * math.sin(t3 - t2)
        a3 = -(g / max(L3, 1.0e-6)) * math.sin(t3) - damping * w3 + 0.12 * math.sin(t2 - t3)
        w1 += dt * a1
        w2 += dt * a2
        w3 += dt * a3
        t1 += dt * w1
        t2 += dt * w2
        t3 += dt * w3

    return {
        "times": times.tolist(),
        "objects": [
            _capsule_track(name="rod_1", color=_rgb(0.56, 0.58, 0.64), points_a=rod1_a, points_b=rod1_b, radius=max(0.015, 0.20 * r)),
            _capsule_track(name="rod_2", color=_rgb(0.46, 0.48, 0.58), points_a=rod2_a, points_b=rod2_b, radius=max(0.015, 0.20 * r)),
            _capsule_track(name="rod_3", color=_rgb(0.38, 0.42, 0.56), points_a=rod3_a, points_b=rod3_b, radius=max(0.015, 0.20 * r)),
            _circle_track(name="bob_1", radius=r, mass=m1, color=_rgb(0.16, 0.53, 0.78), positions=bob1),
            _circle_track(name="bob_2", radius=r, mass=m2, color=_rgb(0.93, 0.45, 0.18), positions=bob2),
            _circle_track(name="bob_3", radius=r, mass=m3, color=_rgb(0.82, 0.30, 0.52), positions=bob3),
        ],
        "scene": {
            "view_plane": "xz",
            "ground_value": None,
            "title": _case_name(case, Path(".")),
        },
    }
