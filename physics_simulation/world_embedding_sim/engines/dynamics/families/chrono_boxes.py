from __future__ import annotations
from run_case import (Any, Callable, Optional, Path, _case_name, _chrono_add_ground, _chrono_density_for_box, _chrono_material, _chrono_quat_z, _chrono_quaternion, _chrono_sampling_loop, _chrono_vec, _chrono_yaw_deg, _duration, _family, _gravity, _mech_cfg, _rgb, _scene_oblique, _solver_cfg, _step_size, math, re)

def _simulate_chrono_boxes(case: dict[str, Any], chrono: Any) -> dict[str, Any]:
    family = _family(case)
    if family == "stackStability" and _step_size(case) > 0.001:
        case = dict(case)
        case["solver"] = dict(_solver_cfg(case))
        case["solver"]["step_size"] = 0.001
    if family == "stackStability" and _duration(case) < 9.0:
        case = dict(case)
        case["solver"] = dict(_solver_cfg(case))
        case["solver"]["duration"] = 9.0
    system = chrono.ChSystemNSC()
    system.SetCollisionSystemType(chrono.ChCollisionSystem.Type_BULLET)
    system.SetGravitationalAcceleration(_chrono_vec(chrono, _gravity(case)))
    if hasattr(system, "SetSleepingAllowed"):
        system.SetSleepingAllowed(True)
    solver_type = str(_solver_cfg(case).get("solver_type", "BARZILAIBORWEIN")).strip().upper()
    solver_attr = getattr(chrono.ChSolver, f"Type_{solver_type}", None)
    if solver_attr is not None:
        system.SetSolverType(solver_attr)
    _chrono_add_ground(system, chrono, case)

    trackers: list[dict[str, Any]] = []
    mech = _mech_cfg(case)

    def add_box(
        *,
        name: str,
        size: list[float],
        mass: float,
        pos: list[float],
        vel: list[float],
        color: list[float],
        friction: float,
        restitution: float,
        scalars: Optional[dict[str, list[float]]] = None,
        render_shape: Optional[str] = None,
        initial_yaw_deg: float = 0.0,
        linear_damping: float = 0.0,
        angular_damping: float = 0.0,
        sleep_min_linear_speed: Optional[float] = None,
        sleep_min_angular_speed: Optional[float] = None,
        sleep_time: Optional[float] = None,
        visual_size: Optional[list[float]] = None,
    ) -> Any:
        mat = _chrono_material(chrono, friction=friction, restitution=restitution)
        body = chrono.ChBodyEasyBox(
            float(size[0]),
            float(size[1]),
            float(size[2]),
            _chrono_density_for_box(mass, size),
            True,
            True,
            mat,
        )
        body.SetPos(_chrono_vec(chrono, pos))
        if abs(float(initial_yaw_deg)) > 1.0e-9:
            body.SetRot(_chrono_quat_z(chrono, math.radians(float(initial_yaw_deg))))
        body.SetPosDt(_chrono_vec(chrono, vel))
        if hasattr(body, "SetLinDamping"):
            body.SetLinDamping(float(max(0.0, linear_damping)))
        if hasattr(body, "SetAngDamping"):
            body.SetAngDamping(float(max(0.0, angular_damping)))
        if hasattr(body, "SetSleepingAllowed"):
            body.SetSleepingAllowed(True)
        if sleep_min_linear_speed is not None and hasattr(body, "SetSleepMinLinVel"):
            body.SetSleepMinLinVel(float(max(0.0, sleep_min_linear_speed)))
        if sleep_min_angular_speed is not None and hasattr(body, "SetSleepMinAngVel"):
            body.SetSleepMinAngVel(float(max(0.0, sleep_min_angular_speed)))
        if sleep_time is not None and hasattr(body, "SetSleepTime"):
            body.SetSleepTime(float(max(0.0, sleep_time)))
        system.Add(body)
        trackers.append(
            {
                "name": name,
                "kind": "box",
                "size": [
                    float((visual_size or size)[0]),
                    float((visual_size or size)[1]),
                    float((visual_size or size)[2]),
                ],
                "mass": float(mass),
                "color": color,
                "scalars": scalars or {},
                "render_shape": render_shape,
                "track_quaternion": True,
                "sample": lambda body=body: {
                    "position": [body.GetPos().x, body.GetPos().y, body.GetPos().z],
                    "quaternion": _chrono_quaternion(body),
                    "scalars": {
                        "yaw_deg": _chrono_yaw_deg(body),
                    } if scalars and "yaw_deg" in scalars else {},
                },
            }
        )
        return body

    def rebind_collision(body: Any) -> None:
        collision_system = system.GetCollisionSystem()
        try:
            collision_system.UnbindItem(body)
        except Exception:
            pass
        collision_system.BindItem(body)
        if hasattr(body, "SyncCollisionModels"):
            body.SyncCollisionModels()
        system.Update(True)

    if family == "rigidCollision":
        body_a = mech.get("body_a") if isinstance(mech.get("body_a"), dict) else {}
        body_b = mech.get("body_b") if isinstance(mech.get("body_b"), dict) else {}
        size_a = list(body_a.get("size", [0.7, 0.5, 0.4]))
        size_b = list(body_b.get("size", [0.9, 0.6, 0.5]))
        pos_a = list(body_a.get("position", [-2.0, 0.0, 0.2]))
        pos_b = list(body_b.get("position", [2.0, 0.0, 0.25]))
        if len(pos_a) >= 2 and len(pos_b) >= 2 and len(size_a) >= 2 and len(size_b) >= 2:
            center_y = 0.5 * (float(pos_a[1]) + float(pos_b[1]))
            half_y_sum = 0.5 * (float(size_a[1]) + float(size_b[1]))
            offset_limit = 0.58 * half_y_sum
            offset_y = max(-offset_limit, min(offset_limit, float(pos_b[1]) - float(pos_a[1])))
            min_offset = 0.18 * min(float(size_a[1]), float(size_b[1]))
            if abs(offset_y) < min_offset:
                offset_y = math.copysign(min_offset, offset_y if abs(offset_y) > 1.0e-9 else 1.0)
            pos_a[1] = center_y - 0.5 * offset_y
            pos_b[1] = center_y + 0.5 * offset_y
        fr = min(0.080, max(0.0, float(mech.get("friction", 0.02))))
        # Use the configured restitution. Previous releases forced this to
        # zero, which made most "rigid collision" cases look adhesive.
        re = max(0.0, min(0.95, float(mech.get("restitution", 0.55))))
        rigid_bodies: list[Any] = []
        rigid_bodies.append(
            add_box(
            name="body_a",
            size=size_a,
            mass=float(body_a.get("mass", 5.0)),
            pos=pos_a,
            vel=body_a.get("velocity", [3.0, 0.0, 0.0]),
            color=_rgb(0.19, 0.52, 0.83),
            friction=fr,
            restitution=re,
            scalars={"yaw_deg": []},
            initial_yaw_deg=float(body_a.get("initial_yaw_deg", 0.0)),
            linear_damping=float(body_a.get("linear_damping", mech.get("linear_damping", 0.0))),
            angular_damping=float(body_a.get("angular_damping", mech.get("angular_damping", 0.0))),
            )
        )
        rigid_bodies.append(
            add_box(
            name="body_b",
            size=size_b,
            mass=float(body_b.get("mass", 7.0)),
            pos=pos_b,
            vel=body_b.get("velocity", [-1.0, 0.0, 0.0]),
            color=_rgb(0.89, 0.39, 0.17),
            friction=fr,
            restitution=re,
            scalars={"yaw_deg": []},
            initial_yaw_deg=float(body_b.get("initial_yaw_deg", 0.0)),
            linear_damping=float(body_b.get("linear_damping", mech.get("linear_damping", 0.0))),
            angular_damping=float(body_b.get("angular_damping", mech.get("angular_damping", 0.0))),
            )
        )
        step_hook: Optional[Callable[[int, float], None]] = None
        if bool(mech.get("enforce_planar_motion", True)):
            def rigid_step_hook(_: int, __: float) -> None:
                for body in rigid_bodies:
                    pos = body.GetPos()
                    vel = body.GetPosDt()
                    body.SetPos(_chrono_vec(chrono, [pos.x, pos.y, 0.0]))
                    body.SetPosDt(_chrono_vec(chrono, [vel.x, vel.y, 0.0]))
                    if hasattr(body, "SetAngVelParent"):
                        ang = body.GetAngVelParent()
                        body.SetAngVelParent(_chrono_vec(chrono, [0.0, 0.0, ang.z]))

            step_hook = rigid_step_hook
        scene = {"view_plane": "xy", "ground_value": None, "title": _case_name(case, Path("."))}
        return _chrono_sampling_loop(case=case, system=system, trackers=trackers, scene=scene, step_hook=step_hook)

    if family == "stackStability":
        levels = max(2, int(mech.get("levels", 4)))
        size = mech.get("box_size", [0.35, 0.35, 0.28])
        box_mass = float(mech.get("box_mass", 2.0))
        gap = max(float(mech.get("gap", 0.0)), 0.02)
        box_friction = float(mech.get("box_friction", mech.get("ground_friction", 0.5)))
        linear_damping = max(0.140, float(mech.get("linear_damping", 0.16)))
        angular_damping = max(0.200, float(mech.get("angular_damping", 0.24)))
        sleep_min_linear_speed = max(0.035, float(mech.get("sleep_min_linear_speed", 0.035)))
        sleep_min_angular_speed = max(0.110, float(mech.get("sleep_min_angular_speed", 0.110)))
        sleep_time = min(0.24, max(0.12, float(mech.get("sleep_time", 0.24))))
        stack_bodies: list[Any] = []
        for i in range(levels):
            stack_bodies.append(
                add_box(
                name=f"box_{i}",
                size=size,
                mass=box_mass,
                pos=[0.0, 0.0, 0.5 * float(size[2]) + i * (float(size[2]) + gap)],
                vel=[0.0, 0.0, 0.0],
                color=_rgb(0.22 + 0.08 * i, 0.44, max(0.18, 0.78 - 0.08 * i)),
                friction=box_friction,
                restitution=0.02,
                linear_damping=linear_damping,
                angular_damping=angular_damping,
                sleep_min_linear_speed=sleep_min_linear_speed,
                sleep_min_angular_speed=sleep_min_angular_speed,
                sleep_time=sleep_time,
                )
            )
        impactor = mech.get("impactor") if isinstance(mech.get("impactor"), dict) else {}
        impactor_body = None
        drive_release_x = 0.0
        impact_speed = float(impactor.get("speed", 1.5))
        impact_delay = float(impactor.get("start_delay", 0.20))
        settle_time = float(mech.get("settle_time", 0.18))
        if settle_time > 0.0:
            settle_steps = max(1, int(math.ceil(settle_time / max(_step_size(case), 1.0e-6))))
            for _ in range(settle_steps):
                system.DoStepDynamics(_step_size(case))
            zero = _chrono_vec(chrono, [0.0, 0.0, 0.0])
            for body in stack_bodies:
                body.SetPosDt(zero)
                if hasattr(body, "SetAngVelParent"):
                    body.SetAngVelParent(zero)
        impact_y = float(impactor.get("y_offset", 0.0))
        impact_z = 0.0
        if bool(impactor.get("enabled", True)):
            imp_size = impactor.get("size", [0.2, 0.2, 0.2])
            height_ratio = max(0.20, min(0.90, float(impactor.get("height_ratio", 0.42))))
            impact_z = min(
                0.5 * float(imp_size[2]) + height_ratio * levels * float(size[2]),
                0.5 * float(size[2]) + max(0, levels - 1) * (float(size[2]) + gap),
            )
            drive_release_x = -(0.5 * float(size[0]) + 0.5 * float(imp_size[0]))
            pretravel = max(
                0.30,
                min(0.75, max(0.0, impact_delay) * max(impact_speed, 0.0)),
                0.65 * max(float(size[0]), float(imp_size[0])),
            )
            impactor_body = add_box(
                name="impactor",
                size=imp_size,
                mass=float(impactor.get("mass", 1.0)),
                pos=[drive_release_x - pretravel, impact_y, impact_z],
                vel=[0.0, 0.0, 0.0],
                color=_rgb(0.9, 0.3, 0.25),
                friction=float(impactor.get("friction", max(0.06, 0.25 * float(mech.get("ground_friction", 0.5))))),
                restitution=0.02,
                linear_damping=0.5 * linear_damping,
                angular_damping=0.5 * angular_damping,
                sleep_min_linear_speed=sleep_min_linear_speed,
                sleep_min_angular_speed=sleep_min_angular_speed,
                sleep_time=sleep_time,
            )
            rebind_collision(impactor_body)
        launch_time = float(system.GetChTime()) + max(0.0, impact_delay)
        enforce_preimpact_cruise = bool(impactor.get("enforce_preimpact_cruise", True))
        freeze_stack_until_impact = bool(mech.get("freeze_stack_until_impact", True))
        impactor_launched = False
        impactor_released = False

        def stack_step_hook(_: int, sim_time: float) -> None:
            nonlocal impactor_launched, impactor_released
            if freeze_stack_until_impact and sim_time + 1.0e-12 < launch_time:
                zero = _chrono_vec(chrono, [0.0, 0.0, 0.0])
                for body in stack_bodies:
                    body.SetPosDt(zero)
                    if hasattr(body, "SetAngVelParent"):
                        body.SetAngVelParent(zero)
            if impactor_body is None:
                return
            vel = impactor_body.GetPosDt()
            if sim_time + 1.0e-12 < launch_time:
                impactor_body.SetPos(_chrono_vec(chrono, [impactor_body.GetPos().x, impact_y, impact_z]))
                impactor_body.SetPosDt(_chrono_vec(chrono, [0.0, 0.0, 0.0]))
                if hasattr(impactor_body, "SetAngVelParent"):
                    impactor_body.SetAngVelParent(_chrono_vec(chrono, [0.0, 0.0, 0.0]))
                return
            if not impactor_launched:
                if hasattr(impactor_body, "SetSleeping"):
                    impactor_body.SetSleeping(False)
                impactor_body.SetPosDt(_chrono_vec(chrono, [impact_speed, 0.0, 0.0]))
                impactor_launched = True
            vel = impactor_body.GetPosDt()
            if (
                impactor_released
                or not enforce_preimpact_cruise
                or impactor_body.GetPos().x >= drive_release_x
                or vel.x < 0.85 * impact_speed
            ):
                impactor_released = True
                return
            impactor_body.SetPos(_chrono_vec(chrono, [impactor_body.GetPos().x, impact_y, impact_z]))
            impactor_body.SetPosDt(_chrono_vec(chrono, [vel.x, 0.0, 0.0]))
            if hasattr(impactor_body, "SetAngVelParent"):
                impactor_body.SetAngVelParent(_chrono_vec(chrono, [0.0, 0.0, 0.0]))

        scene = _scene_oblique(_case_name(case, Path(".")), ground_value=0.0)
        scene["camera_yaw_deg"] = 34.0
        scene["camera_pitch_deg"] = 20.0
        scene["camera_zoom"] = 1.15
        return _chrono_sampling_loop(case=case, system=system, trackers=trackers, scene=scene, step_hook=stack_step_hook)

    if family == "blockTower3D":
        levels = max(3, int(mech.get("levels", 5)))
        layers = max(2, int(mech.get("layers", 3)))
        size = mech.get("block_size", [1.1, 0.28, 0.22])
        block_mass = float(mech.get("block_mass", 1.2))
        gap = max(float(mech.get("gap", 0.0)), 0.025)
        block_friction = min(0.38, float(mech.get("block_friction", mech.get("ground_friction", 0.5))))
        linear_damping = float(mech.get("linear_damping", 0.025))
        angular_damping = float(mech.get("angular_damping", 0.045))
        sx = float(size[0])
        sy = float(size[1])
        sz = float(size[2])
        tower_bodies: list[Any] = []
        for level in range(levels):
            orient_x = (level % 2) == 0
            block_size = [sx, sy, sz] if orient_x else [sy, sx, sz]
            spacing = sy * 1.12
            for layer in range(layers):
                x0 = 0.0 if orient_x else (layer - 0.5 * (layers - 1)) * spacing
                y0 = (layer - 0.5 * (layers - 1)) * spacing if orient_x else 0.0
                visual_block_size = [0.985 * block_size[0], 0.985 * block_size[1], 0.90 * block_size[2]]
                tower_bodies.append(add_box(
                    name=f"tower_block_{level}_{layer}",
                    size=block_size,
                    mass=block_mass,
                    pos=[x0, y0, 0.5 * sz + level * (sz + gap)],
                    vel=[0.0, 0.0, 0.0],
                    color=_rgb(0.26 + 0.04 * level, 0.42, max(0.18, 0.72 - 0.05 * level)),
                    friction=block_friction,
                    restitution=0.02,
                    linear_damping=linear_damping,
                    angular_damping=angular_damping,
                    sleep_min_linear_speed=float(mech.get("sleep_min_linear_speed", 0.0025)),
                    sleep_min_angular_speed=float(mech.get("sleep_min_angular_speed", 0.010)),
                    sleep_time=float(mech.get("sleep_time", 0.34)),
                    visual_size=visual_block_size,
                ))
        settle_time = float(mech.get("settle_time", 0.22))
        if settle_time > 0.0:
            settle_steps = max(1, int(math.ceil(settle_time / max(_step_size(case), 1.0e-6))))
            for _ in range(settle_steps):
                system.DoStepDynamics(_step_size(case))
            zero = _chrono_vec(chrono, [0.0, 0.0, 0.0])
            for body in tower_bodies:
                body.SetPosDt(zero)
                if hasattr(body, "SetAngVelParent"):
                    body.SetAngVelParent(zero)
        impactor = mech.get("impactor") if isinstance(mech.get("impactor"), dict) else {}
        impactor_body = None
        raw_impact_speed = float(impactor.get("speed", 5.0))
        impact_speed = min(7.4, max(5.2, raw_impact_speed))
        impact_delay = float(impactor.get("start_delay", 0.12))
        drive_release_x = 0.0
        raw_impact_y = float(impactor.get("y_offset", 0.0))
        impact_y = raw_impact_y
        impact_z = 0.0
        if bool(impactor.get("enabled", True)):
            imp_size = impactor.get("size", [0.36, 0.36, 0.28])
            height_ratio = max(0.52, min(0.84, float(impactor.get("height_ratio", 0.42))))
            raw_impact_mass = float(impactor.get("mass", 2.5))
            impact_mass = min(max(raw_impact_mass, 3.5 * block_mass), 6.0 * block_mass)
            target_level_raw = impactor.get("target_level")
            if target_level_raw is None:
                target_level = int(round(height_ratio * max(0, levels - 1)))
            else:
                target_level = int(target_level_raw)
            target_level = max(0, min(levels - 1, target_level))
            target_orient_x = (target_level % 2) == 0
            if target_orient_x:
                target_layer_raw = impactor.get("target_layer")
                if target_layer_raw is None:
                    target_layer = int(round(raw_impact_y / max(spacing, 1.0e-9) + 0.5 * (layers - 1)))
                else:
                    target_layer = int(target_layer_raw)
                target_layer = max(0, min(layers - 1, target_layer))
                base_y = (target_layer - 0.5 * (layers - 1)) * spacing
                target_width_y = sy
            else:
                base_y = 0.0
                target_width_y = sx
            if "y_jitter_fraction" in impactor:
                jitter_y = float(impactor.get("y_jitter_fraction", 0.0)) * 0.5 * target_width_y
            else:
                jitter_y = raw_impact_y - base_y
            max_jitter_y = max(0.0, 0.40 * target_width_y)
            impact_y = base_y + max(-max_jitter_y, min(max_jitter_y, jitter_y))
            target_center_z = 0.5 * sz + target_level * (sz + gap)
            jitter_z = float(impactor.get("z_jitter_fraction", 0.0)) * 0.5 * sz
            impact_z = max(
                0.5 * float(imp_size[2]),
                min(0.5 * sz + max(0, levels - 1) * (sz + gap), target_center_z + jitter_z),
            )
            drive_release_x = -(0.5 * sx + 0.5 * float(imp_size[0]))
            pretravel = max(0.55, min(1.15, impact_delay * max(impact_speed, 0.0)), 0.75 * max(sx, float(imp_size[0])))
            impactor_body = add_box(
                name="impactor",
                size=imp_size,
                mass=impact_mass,
                pos=[drive_release_x - pretravel, impact_y, impact_z],
                vel=[0.0, 0.0, 0.0],
                color=_rgb(0.9, 0.3, 0.22),
                friction=float(impactor.get("friction", max(0.05, 0.2 * float(mech.get("ground_friction", 0.5))))),
                restitution=0.02,
                linear_damping=0.5 * linear_damping,
                angular_damping=0.5 * angular_damping,
                sleep_min_linear_speed=float(mech.get("sleep_min_linear_speed", 0.0025)),
                sleep_min_angular_speed=float(mech.get("sleep_min_angular_speed", 0.010)),
                sleep_time=float(mech.get("sleep_time", 0.34)),
            )
            rebind_collision(impactor_body)
        launch_time = float(system.GetChTime()) + max(0.0, impact_delay)
        enforce_preimpact_cruise = bool(impactor.get("enforce_preimpact_cruise", True))
        impactor_launched = False
        impactor_released = False

        def tower_step_hook(_: int, sim_time: float) -> None:
            nonlocal impactor_launched, impactor_released
            if impactor_body is None:
                return
            if sim_time + 1.0e-12 < launch_time:
                impactor_body.SetPos(_chrono_vec(chrono, [impactor_body.GetPos().x, impact_y, impact_z]))
                impactor_body.SetPosDt(_chrono_vec(chrono, [0.0, 0.0, 0.0]))
                if hasattr(impactor_body, "SetAngVelParent"):
                    impactor_body.SetAngVelParent(_chrono_vec(chrono, [0.0, 0.0, 0.0]))
                return
            if not impactor_launched:
                if hasattr(impactor_body, "SetSleeping"):
                    impactor_body.SetSleeping(False)
                impactor_body.SetPosDt(_chrono_vec(chrono, [impact_speed, 0.0, 0.0]))
                impactor_launched = True
            vel = impactor_body.GetPosDt()
            if (
                impactor_released
                or not enforce_preimpact_cruise
                or impactor_body.GetPos().x >= drive_release_x
                or vel.x < 0.85 * impact_speed
            ):
                impactor_released = True
                return
            impactor_body.SetPos(_chrono_vec(chrono, [impactor_body.GetPos().x, impact_y, impact_z]))
            impactor_body.SetPosDt(_chrono_vec(chrono, [vel.x, 0.0, 0.0]))
            if hasattr(impactor_body, "SetAngVelParent"):
                impactor_body.SetAngVelParent(_chrono_vec(chrono, [0.0, 0.0, 0.0]))
        scene = _scene_oblique(_case_name(case, Path(".")), ground_value=0.0)
        scene["camera_yaw_deg"] = float(mech.get("camera_yaw_deg", 34.0))
        scene["camera_pitch_deg"] = float(mech.get("camera_pitch_deg", 24.0))
        scene["camera_zoom"] = float(mech.get("camera_zoom", 1.24))
        return _chrono_sampling_loop(case=case, system=system, trackers=trackers, scene=scene, step_hook=tower_step_hook)

    raise ValueError(f"Unsupported Chrono box family: {family}")
