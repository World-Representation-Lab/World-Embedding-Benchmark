from __future__ import annotations
from run_case import (Any, Path, _case_name, _chrono_density_for_sphere, _chrono_joint_frame_y, _chrono_material, _chrono_sampling_loop, _chrono_vec, _gravity, _mech_cfg, _rgb, math, np)

def _simulate_chrono_pendulum(case: dict[str, Any], chrono: Any) -> dict[str, Any]:
    system = chrono.ChSystemNSC()
    system.SetCollisionSystemType(chrono.ChCollisionSystem.Type_BULLET)
    system.SetGravitationalAcceleration(_chrono_vec(chrono, _gravity(case)))

    mech = _mech_cfg(case)
    L = float(mech.get("length", 1.0))
    r = float(mech.get("bob_radius", 0.08))
    mass = float(mech.get("mass", 1.0))
    pivot = np.asarray(mech.get("pivot", [0.0, 0.0, 0.0]), dtype=float)
    theta = math.radians(float(mech.get("initial_angle_deg", 25.0)))
    omega = math.radians(float(mech.get("initial_angular_velocity_deg_s", 0.0)))
    mat = _chrono_material(chrono, friction=0.2, restitution=0.0)

    anchor = chrono.ChBody()
    anchor.SetFixed(True)
    anchor.EnableCollision(False)
    system.Add(anchor)

    bob = chrono.ChBodyEasySphere(r, _chrono_density_for_sphere(mass, r), True, True, mat)
    bob_pos = pivot + np.asarray([L * math.sin(theta), 0.0, -L * math.cos(theta)], dtype=float)
    bob_vel = np.asarray([L * math.cos(theta) * omega, 0.0, L * math.sin(theta) * omega], dtype=float)
    bob.SetPos(_chrono_vec(chrono, bob_pos))
    bob.SetPosDt(_chrono_vec(chrono, bob_vel))
    bob.EnableCollision(False)
    system.Add(bob)

    link = chrono.ChLinkLockRevolute()
    link.Initialize(anchor, bob, _chrono_joint_frame_y(chrono, pivot))
    system.Add(link)

    trackers = [
        {
            "name": "rod",
            "kind": "capsule",
            "size": [max(0.018, 0.24 * r)],
            "mass": 0.0,
            "color": _rgb(0.56, 0.58, 0.64),
            "static_fields": {"color_mode": "fixed"},
            "scalars": {},
            "sample": lambda: {
                "point_a": pivot.tolist(),
                "point_b": [bob.GetPos().x, bob.GetPos().y, bob.GetPos().z],
                "position": [0.5 * (pivot[0] + bob.GetPos().x), 0.5 * (pivot[1] + bob.GetPos().y), 0.5 * (pivot[2] + bob.GetPos().z)],
                "scalars": {},
            },
        },
        {
            "name": "bob",
            "kind": "circle",
            "size": [r],
            "mass": mass,
            "color": _rgb(0.92, 0.36, 0.27),
            "scalars": {},
            "sample": lambda: {"position": [bob.GetPos().x, bob.GetPos().y, bob.GetPos().z], "scalars": {}},
        },
    ]
    return _chrono_sampling_loop(case=case, system=system, trackers=trackers, scene={"view_plane": "xz", "ground_value": None, "title": _case_name(case, Path("."))})
