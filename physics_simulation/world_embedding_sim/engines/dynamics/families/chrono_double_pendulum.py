from __future__ import annotations
from run_case import (Any, Path, _case_name, _chrono_density_for_sphere, _chrono_joint_frame_y, _chrono_material, _chrono_sampling_loop, _chrono_vec, _gravity, _mech_cfg, _rgb, math, np)

def _simulate_chrono_double_pendulum(case: dict[str, Any], chrono: Any) -> dict[str, Any]:
    system = chrono.ChSystemNSC()
    system.SetCollisionSystemType(chrono.ChCollisionSystem.Type_BULLET)
    system.SetGravitationalAcceleration(_chrono_vec(chrono, _gravity(case)))

    mech = _mech_cfg(case)
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
    mat = _chrono_material(chrono, friction=0.2, restitution=0.0)

    anchor = chrono.ChBody()
    anchor.SetFixed(True)
    anchor.EnableCollision(False)
    system.Add(anchor)

    bob1 = chrono.ChBodyEasySphere(r, _chrono_density_for_sphere(m1, r), True, True, mat)
    bob2 = chrono.ChBodyEasySphere(r, _chrono_density_for_sphere(m2, r), True, True, mat)

    p1 = pivot + np.asarray([L1 * math.sin(t1), 0.0, -L1 * math.cos(t1)], dtype=float)
    p2 = p1 + np.asarray([L2 * math.sin(t2), 0.0, -L2 * math.cos(t2)], dtype=float)
    v1 = np.asarray([L1 * math.cos(t1) * w1, 0.0, L1 * math.sin(t1) * w1], dtype=float)
    v2 = v1 + np.asarray([L2 * math.cos(t2) * w2, 0.0, L2 * math.sin(t2) * w2], dtype=float)
    bob1.SetPos(_chrono_vec(chrono, p1))
    bob2.SetPos(_chrono_vec(chrono, p2))
    bob1.SetPosDt(_chrono_vec(chrono, v1))
    bob2.SetPosDt(_chrono_vec(chrono, v2))
    bob1.EnableCollision(False)
    bob2.EnableCollision(False)
    system.Add(bob1)
    system.Add(bob2)

    link1 = chrono.ChLinkLockRevolute()
    link1.Initialize(anchor, bob1, _chrono_joint_frame_y(chrono, pivot))
    system.Add(link1)
    link2 = chrono.ChLinkLockRevolute()
    link2.Initialize(bob1, bob2, _chrono_joint_frame_y(chrono, p1))
    system.Add(link2)

    trackers = [
        {
            "name": "rod_1",
            "kind": "capsule",
            "size": [max(0.016, 0.22 * r)],
            "mass": 0.0,
            "color": _rgb(0.56, 0.58, 0.64),
            "static_fields": {"color_mode": "fixed"},
            "scalars": {},
            "sample": lambda: {
                "point_a": pivot.tolist(),
                "point_b": [bob1.GetPos().x, bob1.GetPos().y, bob1.GetPos().z],
                "position": [0.5 * (pivot[0] + bob1.GetPos().x), 0.5 * (pivot[1] + bob1.GetPos().y), 0.5 * (pivot[2] + bob1.GetPos().z)],
                "scalars": {},
            },
        },
        {
            "name": "rod_2",
            "kind": "capsule",
            "size": [max(0.016, 0.22 * r)],
            "mass": 0.0,
            "color": _rgb(0.46, 0.48, 0.58),
            "static_fields": {"color_mode": "fixed"},
            "scalars": {},
            "sample": lambda: {
                "point_a": [bob1.GetPos().x, bob1.GetPos().y, bob1.GetPos().z],
                "point_b": [bob2.GetPos().x, bob2.GetPos().y, bob2.GetPos().z],
                "position": [0.5 * (bob1.GetPos().x + bob2.GetPos().x), 0.5 * (bob1.GetPos().y + bob2.GetPos().y), 0.5 * (bob1.GetPos().z + bob2.GetPos().z)],
                "scalars": {},
            },
        },
        {
            "name": "bob_1",
            "kind": "circle",
            "size": [r],
            "mass": m1,
            "color": _rgb(0.16, 0.53, 0.78),
            "scalars": {},
            "sample": lambda: {"position": [bob1.GetPos().x, bob1.GetPos().y, bob1.GetPos().z], "scalars": {}},
        },
        {
            "name": "bob_2",
            "kind": "circle",
            "size": [r],
            "mass": m2,
            "color": _rgb(0.93, 0.45, 0.18),
            "scalars": {},
            "sample": lambda: {"position": [bob2.GetPos().x, bob2.GetPos().y, bob2.GetPos().z], "scalars": {}},
        },
    ]
    return _chrono_sampling_loop(case=case, system=system, trackers=trackers, scene={"view_plane": "xz", "ground_value": None, "title": _case_name(case, Path("."))})
