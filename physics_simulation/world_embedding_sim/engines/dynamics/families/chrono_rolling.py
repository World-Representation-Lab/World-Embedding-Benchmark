from __future__ import annotations
from run_case import (Any, Path, _case_name, _chrono_add_ground, _chrono_density_for_box, _chrono_material, _chrono_sampling_loop, _chrono_vec, _gravity, _mech_cfg, _rgb)

def _simulate_chrono_rolling(case: dict[str, Any], chrono: Any) -> dict[str, Any]:
    system = chrono.ChSystemNSC()
    system.SetCollisionSystemType(chrono.ChCollisionSystem.Type_BULLET)
    system.SetGravitationalAcceleration(_chrono_vec(chrono, _gravity(case)))
    _chrono_add_ground(system, chrono, case)

    mech = _mech_cfg(case)
    radius = float(mech.get("radius", 0.25))
    width = float(mech.get("width", 0.18))
    mass = float(mech.get("mass", 1.0))
    friction = float(mech.get("friction", 0.35))
    rolling_friction = float(mech.get("rolling_friction", 0.0))
    mat = _chrono_material(chrono, friction=friction, restitution=0.02, rolling_friction=rolling_friction)
    cyl = chrono.ChBodyEasyCylinder(
        chrono.ChAxis_Y,
        radius,
        width,
        _chrono_density_for_box(mass, [radius * 2.0, width, radius * 2.0]),
        True,
        True,
        mat,
    )
    cyl.SetPos(_chrono_vec(chrono, [0.0, 0.0, radius]))
    cyl.SetPosDt(_chrono_vec(chrono, [float(mech.get("initial_speed", 3.0)), 0.0, 0.0]))
    cyl.SetAngVelParent(_chrono_vec(chrono, [0.0, float(mech.get("initial_omega_rad_s", 0.0)), 0.0]))
    system.Add(cyl)

    trackers = [
        {
            "name": "roller",
            "kind": "circle",
            "size": [radius],
            "mass": mass,
            "color": _rgb(0.18, 0.61, 0.5),
            "scalars": {"omega_y": 0.0},
            "sample": lambda: {
                "position": [cyl.GetPos().x, cyl.GetPos().y, cyl.GetPos().z],
                "scalars": {"omega_y": getattr(cyl.GetAngVelParent(), "y", 0.0)},
            },
        }
    ]
    return _chrono_sampling_loop(case=case, system=system, trackers=trackers, scene={"view_plane": "xz", "ground_value": 0.0, "title": _case_name(case, Path(".")), "projected_depth": width})
