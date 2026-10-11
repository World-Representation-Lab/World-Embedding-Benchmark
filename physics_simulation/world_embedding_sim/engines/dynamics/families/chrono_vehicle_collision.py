from __future__ import annotations
from run_case import (Any, Path, _BMW_E90_FALLBACK, _case_name, _chrono_material, _chrono_quat_z, _chrono_sampling_loop, _chrono_vec, _chrono_yaw_deg, _gravity, _j2_stress_from_eq_strain, _load_vehicle_specs, _mech_cfg, _rgb, _scene_oblique, _solver_cfg, _vehicle_data_root, _vehicle_mesh_paths, _vehicle_profile_from_case, math, np)

def _simulate_chrono_vehicle_collision(case: dict[str, Any], chrono: Any) -> dict[str, Any]:
    import pychrono.vehicle as veh

    profile = _vehicle_profile_from_case(case)
    root = _vehicle_data_root(veh, profile)
    spec = _load_vehicle_specs(profile, root)
    mech = _mech_cfg(case)
    vehicle_a = mech.get("vehicle_a") if isinstance(mech.get("vehicle_a"), dict) else {}
    vehicle_b = mech.get("vehicle_b") if isinstance(mech.get("vehicle_b"), dict) else {}
    friction = float(mech.get("friction", 0.6))
    restitution = float(mech.get("restitution", 0.08))
    thickness = float(mech.get("collision_thickness", 0.003))
    wheel_width_scale = max(0.8, min(2.6, float(mech.get("wheel_width_scale", 1.0))))
    j2_young_mpa = float(mech.get("j2_young_mpa", 210000.0))
    j2_yield_mpa = float(mech.get("j2_yield_mpa", 320.0))
    j2_hardening_mpa = float(mech.get("j2_hardening_mpa", 1600.0))
    j2_strain_scale = float(mech.get("j2_strain_scale", 0.20))

    system = chrono.ChSystemNSC()
    system.SetCollisionSystemType(chrono.ChCollisionSystem.Type_BULLET)
    system.SetGravitationalAcceleration(_chrono_vec(chrono, _gravity(case)))
    solver_type = str(_solver_cfg(case).get("solver_type", "BARZILAIBORWEIN")).strip().upper()
    solver_attr = getattr(chrono.ChSolver, f"Type_{solver_type}", None)
    if solver_attr is not None:
        system.SetSolverType(solver_attr)
    solver = system.GetSolver() if hasattr(system, "GetSolver") else None
    if solver is not None and hasattr(solver, "AsIterative"):
        iterative = solver.AsIterative()
        if iterative is not None and hasattr(iterative, "SetMaxIterations"):
            iterative.SetMaxIterations(int(_solver_cfg(case).get("max_iterations", 180)))
    if hasattr(system, "SetMaxPenetrationRecoverySpeed"):
        system.SetMaxPenetrationRecoverySpeed(float(_solver_cfg(case).get("max_penetration_recovery_speed", 6.0)))

    mesh_paths = _vehicle_mesh_paths(profile)
    collision_mesh_path = root / mesh_paths["chassis_col"]
    collision_mesh = chrono.ChTriangleMeshConnected()
    collision_mesh.LoadWavefrontMesh(str(collision_mesh_path))
    vis_mesh = chrono.ChTriangleMeshConnected()
    vis_mesh.LoadWavefrontMesh(str(root / mesh_paths["chassis_vis"]))
    material = _chrono_material(chrono, friction=friction, restitution=restitution)

    def add_vehicle(name: str, cfg: dict[str, Any], color: list[float]) -> Any:
        body = chrono.ChBody()
        body.SetMass(float(spec["mass"]))
        body.SetInertiaXX(_chrono_vec(chrono, spec["inertia"]))
        body.SetPos(_chrono_vec(chrono, cfg.get("position", [0.0, 0.0, 0.0])))
        body.SetRot(_chrono_quat_z(chrono, math.radians(float(cfg.get("initial_yaw_deg", 0.0)))))
        body.SetPosDt(_chrono_vec(chrono, cfg.get("velocity", [0.0, 0.0, 0.0])))
        ang_vel = np.asarray(cfg.get("angular_velocity", [0.0, 0.0, 0.0]), dtype=float)
        body.SetAngVelParent(_chrono_vec(chrono, ang_vel))
        cshape = chrono.ChCollisionShapeTriangleMesh(material, collision_mesh, False, False, thickness)
        body.AddCollisionShape(cshape)
        body.EnableCollision(True)
        vshape = chrono.ChVisualShapeTriangleMesh()
        vshape.SetMesh(vis_mesh)
        vshape.SetColor(chrono.ChColor(float(color[0]), float(color[1]), float(color[2])))
        body.AddVisualShape(vshape)
        system.Add(body)
        return body

    body_a = add_vehicle("vehicle_a", vehicle_a, _rgb(0.15, 0.46, 0.82))
    body_b = add_vehicle("vehicle_b", vehicle_b, _rgb(0.87, 0.36, 0.15))
    ref_a = np.asarray(vehicle_a.get("position", [0.0, 0.0, 0.0]), dtype=float)
    ref_b = np.asarray(vehicle_b.get("position", [0.0, 0.0, 0.0]), dtype=float)

    trackers = [
        {
            "name": "vehicle_a",
            "kind": "box",
            "size": list(spec["chassis_size"]),
            "mass": float(spec["mass"]),
            "color": _rgb(0.15, 0.46, 0.82),
            "scalars": {"yaw_deg": 0.0, "displacement_m": 0.0, "von_mises_mpa": 0.0, "PEEQ": 0.0},
            "color_metric": "von_mises_mpa",
            "render_shape": "vehicle_mesh",
            "static_fields": {
                "outline_xy": [list(p) for p in spec["outline_xy"]],
                "outline_xz": [list(p) for p in spec.get("outline_xz", _BMW_E90_FALLBACK["outline_xz"])],
                "wheel_offsets_xy": [list(p) for p in spec["wheel_offsets_xy"]],
                "wheel_size_xy": [float(spec["wheel_size_xy"][0]), float(spec["wheel_size_xy"][1])],
                "wheel_width_scale": wheel_width_scale,
                "vehicle_profile": str(spec.get("profile", "sedan")),
            },
            "sample": lambda body=body_a: {
                "position": [body.GetPos().x, body.GetPos().y, body.GetPos().z],
                "scalars": {
                    "yaw_deg": _chrono_yaw_deg(body),
                    "displacement_m": float(np.linalg.norm(np.asarray([body.GetPos().x, body.GetPos().y, body.GetPos().z], dtype=float) - ref_a)),
                    "von_mises_mpa": float(
                        _j2_stress_from_eq_strain(
                            j2_strain_scale
                            * (
                                np.linalg.norm(np.asarray([body.GetPos().x, body.GetPos().y, body.GetPos().z], dtype=float) - ref_a)
                                / max(float(spec["chassis_size"][0]), 1.0e-6)
                            ),
                            young_mpa=j2_young_mpa,
                            yield_mpa=j2_yield_mpa,
                            hardening_mpa=j2_hardening_mpa,
                        )[0]
                    ),
                    "PEEQ": float(
                        _j2_stress_from_eq_strain(
                            j2_strain_scale
                            * (
                                np.linalg.norm(np.asarray([body.GetPos().x, body.GetPos().y, body.GetPos().z], dtype=float) - ref_a)
                                / max(float(spec["chassis_size"][0]), 1.0e-6)
                            ),
                            young_mpa=j2_young_mpa,
                            yield_mpa=j2_yield_mpa,
                            hardening_mpa=j2_hardening_mpa,
                        )[1]
                    ),
                },
            },
        },
        {
            "name": "vehicle_b",
            "kind": "box",
            "size": list(spec["chassis_size"]),
            "mass": float(spec["mass"]),
            "color": _rgb(0.87, 0.36, 0.15),
            "scalars": {"yaw_deg": 0.0, "displacement_m": 0.0, "von_mises_mpa": 0.0, "PEEQ": 0.0},
            "color_metric": "von_mises_mpa",
            "render_shape": "vehicle_mesh",
            "static_fields": {
                "outline_xy": [list(p) for p in spec["outline_xy"]],
                "outline_xz": [list(p) for p in spec.get("outline_xz", _BMW_E90_FALLBACK["outline_xz"])],
                "wheel_offsets_xy": [list(p) for p in spec["wheel_offsets_xy"]],
                "wheel_size_xy": [float(spec["wheel_size_xy"][0]), float(spec["wheel_size_xy"][1])],
                "wheel_width_scale": wheel_width_scale,
                "vehicle_profile": str(spec.get("profile", "sedan")),
            },
            "sample": lambda body=body_b: {
                "position": [body.GetPos().x, body.GetPos().y, body.GetPos().z],
                "scalars": {
                    "yaw_deg": _chrono_yaw_deg(body),
                    "displacement_m": float(np.linalg.norm(np.asarray([body.GetPos().x, body.GetPos().y, body.GetPos().z], dtype=float) - ref_b)),
                    "von_mises_mpa": float(
                        _j2_stress_from_eq_strain(
                            j2_strain_scale
                            * (
                                np.linalg.norm(np.asarray([body.GetPos().x, body.GetPos().y, body.GetPos().z], dtype=float) - ref_b)
                                / max(float(spec["chassis_size"][0]), 1.0e-6)
                            ),
                            young_mpa=j2_young_mpa,
                            yield_mpa=j2_yield_mpa,
                            hardening_mpa=j2_hardening_mpa,
                        )[0]
                    ),
                    "PEEQ": float(
                        _j2_stress_from_eq_strain(
                            j2_strain_scale
                            * (
                                np.linalg.norm(np.asarray([body.GetPos().x, body.GetPos().y, body.GetPos().z], dtype=float) - ref_b)
                                / max(float(spec["chassis_size"][0]), 1.0e-6)
                            ),
                            young_mpa=j2_young_mpa,
                            yield_mpa=j2_yield_mpa,
                            hardening_mpa=j2_hardening_mpa,
                        )[1]
                    ),
                },
            },
        },
    ]
    scene = _scene_oblique(_case_name(case, Path(".")), ground_value=0.0)
    return _chrono_sampling_loop(case=case, system=system, trackers=trackers, scene=scene)
