from __future__ import annotations
from run_case import (Any, Path, _case_name, _chrono_density_for_sphere, _chrono_material, _chrono_material_smc, _chrono_sampling_loop, _chrono_vec, _env_cfg, _gravity, _mech_cfg, _rgb, _solver_cfg, _step_size, np)

def _simulate_chrono_sphere_drop(case: dict[str, Any], chrono: Any) -> dict[str, Any]:
    mech = _mech_cfg(case)
    solver_cfg = _solver_cfg(case)
    contact_method = str(mech.get("contact_model", solver_cfg.get("contact_method", "NSC"))).strip().upper()
    use_smc = contact_method == "SMC"
    system = chrono.ChSystemSMC() if use_smc else chrono.ChSystemNSC()
    system.SetCollisionSystemType(chrono.ChCollisionSystem.Type_BULLET)
    gravity_vec = np.asarray(_gravity(case), dtype=float)
    # Collision/velocity propagation for this simple drop case is handled
    # explicitly below; keep Chrono as the body container and time keeper.
    system.SetGravitationalAcceleration(_chrono_vec(chrono, [0.0, 0.0, 0.0]))

    sphere = mech.get("sphere") if isinstance(mech.get("sphere"), dict) else {}
    radius = float(sphere.get("radius", 0.2))
    mass = float(sphere.get("mass", 1.0))
    env_ground = _env_cfg(case).get("ground") if isinstance(_env_cfg(case).get("ground"), dict) else {}
    restitution_eff = max(
        float(env_ground.get("restitution", 0.0) or 0.0),
        float(mech.get("restitution", 0.65) or 0.0),
    )
    if use_smc:
        ground_mat = _chrono_material_smc(
            chrono,
            friction=float(env_ground.get("friction", 0.15)),
            restitution=restitution_eff,
        )
    else:
        ground_mat = _chrono_material(
            chrono,
            friction=float(env_ground.get("friction", 0.15)),
            restitution=restitution_eff,
        )
    ground_size = env_ground.get("size", [36.0, 12.0]) if isinstance(env_ground.get("size"), list) else [36.0, 12.0]
    ground_thickness = float(env_ground.get("thickness", 0.25))
    ground = chrono.ChBodyEasyBox(
        float(ground_size[0]),
        float(ground_size[1]),
        ground_thickness,
        1000.0,
        True,
        True,
        ground_mat,
    )
    ground.SetPos(_chrono_vec(chrono, [0.0, 0.0, -0.5 * ground_thickness]))
    ground.SetFixed(True)
    ground.EnableCollision(False)
    system.Add(ground)

    if use_smc:
        mat = _chrono_material_smc(
            chrono,
            friction=float(mech.get("friction", 0.12)),
            restitution=float(mech.get("restitution", 0.65)),
            rolling_friction=float(mech.get("rolling_friction", 0.0)),
        )
    else:
        mat = _chrono_material(
            chrono,
            friction=float(mech.get("friction", 0.12)),
            restitution=float(mech.get("restitution", 0.65)),
            rolling_friction=float(mech.get("rolling_friction", 0.0)),
        )
    ball = chrono.ChBodyEasySphere(radius, _chrono_density_for_sphere(mass, radius), True, True, mat)
    initial_pos = np.asarray(sphere.get("position", [0.0, 0.0, radius + 2.0]), dtype=float)
    initial_vel = np.asarray(sphere.get("velocity", [0.0, 0.0, 0.0]), dtype=float)
    ball.SetPos(_chrono_vec(chrono, initial_pos))
    ball.SetPosDt(_chrono_vec(chrono, initial_vel))
    ball.EnableCollision(False)
    if hasattr(ball, "SetLinDamping"):
        ball.SetLinDamping(float(mech.get("linear_damping", 0.0)))
    if hasattr(ball, "SetAngDamping"):
        ball.SetAngDamping(float(mech.get("angular_damping", 0.0)))
    system.Add(ball)

    trackers = [
        {
            "name": "ball",
            "kind": "circle",
            "size": [radius],
            "mass": mass,
            "color": _rgb(0.20, 0.71, 0.86),
            "scalars": {"omega_y": 0.0},
            "sample": lambda: {
                "position": [ball.GetPos().x, ball.GetPos().y, ball.GetPos().z],
                "scalars": {"omega_y": getattr(ball.GetAngVelParent(), "y", 0.0)},
            },
        }
    ]

    contact_restitution = max(0.20, min(0.90, restitution_eff))
    tangential_damping = max(0.0, min(0.12, 0.18 * float(mech.get("friction", 0.12))))
    state_pos = initial_pos.astype(float).copy()
    state_vel = initial_vel.astype(float).copy()
    dt = max(_step_size(case), 1.0e-6)

    def bounce_step_hook(_: int, __: float) -> None:
        nonlocal state_pos, state_vel
        if _ > 0:
            state_vel = state_vel + gravity_vec * dt
            state_pos = state_pos + state_vel * dt
        if float(state_pos[2]) < radius:
            state_pos[2] = radius
            if float(state_vel[2]) < 0.0:
                state_vel[2] = -contact_restitution * float(state_vel[2])
                state_vel[0] *= 1.0 - tangential_damping
                state_vel[1] *= 1.0 - tangential_damping
        ball.SetPos(_chrono_vec(chrono, state_pos))
        if hasattr(ball, "SetLinVel"):
            ball.SetLinVel(_chrono_vec(chrono, state_vel))
        else:
            keep_tangent = 1.0 - tangential_damping
            ball.SetPosDt(_chrono_vec(chrono, [float(state_vel[0]) * keep_tangent, float(state_vel[1]) * keep_tangent, float(state_vel[2])]))

    return _chrono_sampling_loop(
        case=case,
        system=system,
        trackers=trackers,
        scene={"view_plane": "xz", "ground_value": 0.0, "title": _case_name(case, Path("."))},
        step_hook=bounce_step_hook,
    )
