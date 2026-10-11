from __future__ import annotations
from run_case import (Any, Path, _case_name, _chrono_density_for_sphere, _chrono_material_smc, _chrono_vec, _chrono_vec_np, _duration, _env_cfg, _gravity, _mech_cfg, _rgb, _scene_bounds, _scene_oblique, _solver_cfg, math, np)

def _simulate_chrono_ball_chute_pile(case: dict[str, Any], chrono: Any) -> dict[str, Any]:
    import pychrono.fea as fea

    mech = _mech_cfg(case)
    env = _env_cfg(case)
    solver_cfg = _solver_cfg(case)
    system = chrono.ChSystemSMC()
    system.SetCollisionSystemType(chrono.ChCollisionSystem.Type_BULLET)
    system.SetGravitationalAcceleration(_chrono_vec(chrono, _gravity(case)))
    if hasattr(system, "SetTimestepperType"):
        system.SetTimestepperType(chrono.ChTimestepper.Type_EULER_IMPLICIT_LINEARIZED)

    solver_type = str(solver_cfg.get("solver_type", "BARZILAIBORWEIN")).strip().upper()
    solver_attr = getattr(chrono.ChSolver, f"Type_{solver_type}", None)
    if solver_attr is not None and hasattr(system, "SetSolverType"):
        system.SetSolverType(solver_attr)

    friction = min(0.55, max(0.35, float(mech.get("friction", 0.44))))
    restitution = min(0.20, max(0.05, float(mech.get("restitution", 0.12))))
    rolling_friction = min(0.015, max(0.0015, float(mech.get("rolling_friction", 0.004))))
    contact_young = min(5.0e5, max(1.0e5, float(mech.get("contact_young_modulus_pa", 3.0e5))))
    duration = _duration(case)
    dt = min(5.0e-4, max(1.0e-4, float(mech.get("internal_step_size_s", 0.0002))))
    export_interval = max(0.002, float(mech.get("export_interval_s", 0.005)))
    rng = np.random.default_rng(int(mech.get("random_seed", int(case.get("seed", 0)))))
    linear_damping = min(0.05, max(0.002, float(mech.get("linear_damping", 0.01))))
    angular_damping = min(0.08, max(0.003, float(mech.get("angular_damping", 0.015))))

    n_balls = max(1, int(mech.get("ball_count", 120)))
    radius_mean = max(0.01, float(mech.get("ball_radius", 0.072)))
    radius_jitter = max(0.0, float(mech.get("radius_jitter", 0.10)))
    ball_density = float(mech.get("ball_density_kg_m3", 2450.0))
    spawn_interval = max(0.010, float(mech.get("spawn_interval_s", 0.055)))
    spawn_offset = max(0.0, float(mech.get("spawn_along_chute_offset_m", 0.10)))
    spawn_height_above_chute = max(0.02, float(mech.get("spawn_height_above_chute_m", 0.10)))

    chute_angle_deg = float(mech.get("chute_angle_deg", 45.0))
    chute_angle = math.radians(chute_angle_deg)
    chute_top = np.asarray(mech.get("chute_top_point", [-1.05, 0.0, 1.55]), dtype=float)
    chute_length = max(0.2, float(mech.get("chute_length", 1.45)))
    chute_width = max(0.2, float(mech.get("chute_width", 0.72)))
    chute_thickness = max(0.02, float(mech.get("chute_thickness", 0.06)))
    chute_wall_thickness = max(0.01, float(mech.get("chute_wall_thickness", 0.03)))
    chute_wall_height = max(0.05, float(mech.get("chute_wall_height", 0.22)))
    flight_gap = max(0.05, float(mech.get("flight_gap", 0.20)))

    box_length = max(0.4, float(mech.get("box_length", 1.65)))
    box_width = max(0.4, float(mech.get("box_width", 1.20)))
    box_wall_height = max(0.2, float(mech.get("box_wall_height", 0.80)))
    box_wall_thickness = max(0.02, float(mech.get("box_wall_thickness", 0.10)))
    box_open_front = bool(mech.get("box_open_front", True))

    ground = env.get("ground") if isinstance(env.get("ground"), dict) else {}
    ground_enabled = bool(ground.get("enabled", True))
    ground_size = np.asarray(ground.get("size", [4.8, 2.6]), dtype=float)
    if ground_size.size < 2:
        ground_size = np.asarray([4.8, 2.6], dtype=float)
    ground_thickness = max(0.02, float(ground.get("thickness", 0.18)))
    ground_z = 0.0

    tangent = np.asarray([math.cos(chute_angle), 0.0, -math.sin(chute_angle)], dtype=float)
    tangent /= max(float(np.linalg.norm(tangent)), 1.0e-9)
    binormal = np.asarray([0.0, 1.0, 0.0], dtype=float)
    normal = np.asarray([math.sin(chute_angle), 0.0, math.cos(chute_angle)], dtype=float)
    normal /= max(float(np.linalg.norm(normal)), 1.0e-9)
    min_exit_clearance = max(0.08, 1.5 * radius_mean)
    chute_exit = chute_top + chute_length * tangent
    required_exit_z = box_wall_height + min_exit_clearance
    if float(chute_exit[2]) < required_exit_z:
        chute_top = chute_top.copy()
        chute_top[2] += required_exit_z - float(chute_exit[2])
        chute_exit = chute_top + chute_length * tangent
    box_front_x = float(chute_exit[0] + flight_gap)
    box_center = np.asarray([box_front_x + 0.5 * box_length, 0.0, 0.5 * box_wall_height], dtype=float)

    if hasattr(chrono, "QuatFromAngleY"):
        chute_rot = chrono.QuatFromAngleY(float(chute_angle))
    else:
        half = 0.5 * float(chute_angle)
        chute_rot = chrono.ChQuaterniond(math.cos(half), 0.0, math.sin(half), 0.0)

    def _smc_material(local_friction: float, local_restitution: float, local_rolling: float = 0.0) -> Any:
        mat = _chrono_material_smc(
            chrono,
            friction=local_friction,
            restitution=local_restitution,
            rolling_friction=local_rolling,
        )
        if hasattr(mat, "SetYoungModulus"):
            mat.SetYoungModulus(float(contact_young))
        return mat

    world_mat = _smc_material(
        float(ground.get("friction", friction)),
        float(ground.get("restitution", 0.05)),
        rolling_friction,
    )
    chute_restitution = min(0.22, max(0.06, float(mech.get("chute_rebound_restitution", 0.14))))
    chute_mat = _smc_material(max(0.40, friction), chute_restitution, max(rolling_friction, 0.003))
    ball_mat = _smc_material(friction, restitution, rolling_friction)

    static_objects: list[dict[str, Any]] = []
    contact_boxes: list[dict[str, Any]] = []

    def _make_box_object(
        *,
        name: str,
        size_xyz: np.ndarray,
        pos_xyz: np.ndarray,
        rot: Any,
        yaw_deg: float,
        mat: Any,
        color: tuple[float, float, float],
        opacity: float = 1.0,
        category: str = "static",
        soft_edges: bool = False,
    ) -> Any:
        body = chrono.ChBodyEasyBox(
            float(size_xyz[0]),
            float(size_xyz[1]),
            float(size_xyz[2]),
            1000.0,
            True,
            True,
            mat,
        )
        body.SetPos(_chrono_vec(chrono, pos_xyz))
        body.SetRot(rot)
        body.SetFixed(True)
        if hasattr(body, "GetVisualShape"):
            try:
                vis = body.GetVisualShape(0)
                if vis is not None and hasattr(vis, "SetColor"):
                    vis.SetColor(chrono.ChColor(float(color[0]), float(color[1]), float(color[2])))
                if vis is not None and hasattr(vis, "SetOpacity"):
                    vis.SetOpacity(float(opacity))
            except Exception:
                pass
        system.Add(body)
        static_objects.append(
            {
                "name": name,
                "kind": "box",
                "size": [float(size_xyz[0]), float(size_xyz[1]), float(size_xyz[2])],
                "mass": 0.0,
                "color": _rgb(color[0], color[1], color[2]),
                "positions": [],
                "quaternions": [],
                "color_mode": "fixed",
                "opacity": float(opacity),
                "soft_edges": bool(soft_edges),
                "_body": body,
            }
        )
        contact_boxes.append(
            {
                "name": name,
                "center": np.asarray(pos_xyz, dtype=float),
                "size": np.asarray(size_xyz, dtype=float),
                "yaw_rad": math.radians(float(yaw_deg)),
                "category": category,
            }
        )
        return body

    if ground_enabled:
        _make_box_object(
            name="ground",
            size_xyz=np.asarray([float(ground_size[0]), float(ground_size[1]), float(ground_thickness)], dtype=float),
            pos_xyz=np.asarray([0.0, 0.0, ground_z - 0.5 * ground_thickness], dtype=float),
            rot=chrono.QUNIT,
            yaw_deg=0.0,
            mat=world_mat,
            color=(0.28, 0.30, 0.33),
            opacity=0.25,
            category="ground",
            soft_edges=True,
        )

    chute_center = chute_top + 0.5 * chute_length * tangent - 0.5 * chute_thickness * normal
    _make_box_object(
        name="chute",
        size_xyz=np.asarray([chute_length, chute_width, chute_thickness], dtype=float),
        pos_xyz=chute_center,
        rot=chute_rot,
        yaw_deg=chute_angle_deg,
        mat=chute_mat,
        color=(0.70, 0.73, 0.78),
        opacity=1.0,
        category="chute",
        soft_edges=True,
    )
    for sign, name in [(1.0, "chute_wall_left"), (-1.0, "chute_wall_right")]:
        _make_box_object(
            name=name,
            size_xyz=np.asarray([chute_length, chute_wall_thickness, chute_wall_height], dtype=float),
            pos_xyz=chute_top
            + 0.5 * chute_length * tangent
            + sign * (0.5 * chute_width + 0.5 * chute_wall_thickness) * binormal
            + 0.5 * chute_wall_height * normal,
            rot=chute_rot,
            yaw_deg=chute_angle_deg,
            mat=chute_mat,
            color=(0.62, 0.66, 0.72),
            opacity=1.0,
            category="chute",
            soft_edges=True,
        )

    collector_color = (0.70, 0.86, 0.95)
    collector_opacity = 0.25
    _make_box_object(
        name="collector_wall_left",
        size_xyz=np.asarray([box_length, box_wall_thickness, box_wall_height], dtype=float),
        pos_xyz=np.asarray([box_center[0], 0.5 * (box_width + box_wall_thickness), 0.5 * box_wall_height], dtype=float),
        rot=chrono.QUNIT,
        yaw_deg=0.0,
        mat=world_mat,
        color=collector_color,
        opacity=collector_opacity,
        category="collector",
        soft_edges=True,
    )
    _make_box_object(
        name="collector_wall_right",
        size_xyz=np.asarray([box_length, box_wall_thickness, box_wall_height], dtype=float),
        pos_xyz=np.asarray([box_center[0], -0.5 * (box_width + box_wall_thickness), 0.5 * box_wall_height], dtype=float),
        rot=chrono.QUNIT,
        yaw_deg=0.0,
        mat=world_mat,
        color=collector_color,
        opacity=collector_opacity,
        category="collector",
        soft_edges=True,
    )
    _make_box_object(
        name="collector_wall_back",
        size_xyz=np.asarray([box_wall_thickness, box_width + 2.0 * box_wall_thickness, box_wall_height], dtype=float),
        pos_xyz=np.asarray([box_center[0] + 0.5 * (box_length + box_wall_thickness), 0.0, 0.5 * box_wall_height], dtype=float),
        rot=chrono.QUNIT,
        yaw_deg=0.0,
        mat=world_mat,
        color=collector_color,
        opacity=collector_opacity,
        category="collector",
        soft_edges=True,
    )
    if not box_open_front:
        _make_box_object(
            name="collector_wall_front",
            size_xyz=np.asarray([box_wall_thickness, box_width + 2.0 * box_wall_thickness, box_wall_height], dtype=float),
            pos_xyz=np.asarray([box_center[0] - 0.5 * (box_length + box_wall_thickness), 0.0, 0.5 * box_wall_height], dtype=float),
            rot=chrono.QUNIT,
            yaw_deg=0.0,
            mat=world_mat,
            color=collector_color,
            opacity=collector_opacity,
            category="collector",
            soft_edges=True,
        )

    probe_points: list[list[float]] = []
    try:
        probe_mesh = fea.ChMesh()
        probe_mesh.SetAutomaticGravity(False)
        probe_nx = max(20, int(mech.get("field_nx", 34)))
        probe_ny = max(3, int(mech.get("field_ny", 7)))
        probe_nz = max(10, int(mech.get("field_nz", 18)))
        x_min = float(chute_top[0] - 0.15)
        x_max = float(box_center[0] + 0.5 * box_length + 0.15)
        y_span = 0.5 * max(chute_width, box_width) + 0.18
        z_min = -0.02
        z_max = float(chute_top[2] + 0.18)
        for x in np.linspace(x_min, x_max, probe_nx):
            for y in np.linspace(-y_span, y_span, probe_ny):
                for z in np.linspace(z_min, z_max, probe_nz):
                    node = fea.ChNodeFEAxyz(chrono.ChVector3d(float(x), float(y), float(z)))
                    node.SetFixed(True)
                    probe_mesh.AddNode(node)
                    probe_points.append([float(x), float(y), float(z)])
        system.Add(probe_mesh)
    except Exception:
        probe_points = []

    ball_specs: list[dict[str, Any]] = []
    ball_objects: list[dict[str, Any]] = []
    hidden_anchor = np.asarray([box_center[0] + 8.0, 0.0, -5.0], dtype=float)
    release_origin = chute_top + spawn_offset * tangent
    for i in range(n_balls):
        radius = max(0.55 * radius_mean, radius_mean * (1.0 + radius_jitter * float(rng.uniform(-1.0, 1.0))))
        mass = float(ball_density * (4.0 / 3.0) * math.pi * radius**3)
        density = _chrono_density_for_sphere(mass, radius)
        body = chrono.ChBodyEasySphere(float(radius), density, True, True, ball_mat)
        if hasattr(body, "SetMass"):
            body.SetMass(float(mass))
        inertia = (2.0 / 5.0) * float(mass) * float(radius) * float(radius)
        if hasattr(body, "SetInertiaXX"):
            body.SetInertiaXX(_chrono_vec(chrono, [inertia, inertia, inertia]))
        if hasattr(body, "SetDamping"):
            body.SetDamping(float(linear_damping))
        if hasattr(body, "SetRotationalDamping"):
            body.SetRotationalDamping(float(angular_damping))
        body.SetPos(_chrono_vec(chrono, hidden_anchor + np.asarray([0.14 * i, 0.0, 0.0], dtype=float)))
        body.SetPosDt(_chrono_vec(chrono, [0.0, 0.0, 0.0]))
        body.SetFixed(True)
        if hasattr(body, "GetVisualShape"):
            try:
                vis = body.GetVisualShape(0)
                if vis is not None and hasattr(vis, "SetColor"):
                    hue = 0.26 + 0.18 * float(i / max(1, n_balls - 1))
                    vis.SetColor(chrono.ChColor(float(hue), 0.68, 0.90))
            except Exception:
                pass
        system.Add(body)

        lateral_room = max(0.0, 0.5 * chute_width - radius - 0.5 * chute_wall_thickness - 0.0015)
        lateral = float(rng.uniform(-0.25 * lateral_room, 0.25 * lateral_room)) if lateral_room > 1.0e-6 else 0.0
        release_pos = release_origin + lateral * binormal + (radius + 0.004 + spawn_height_above_chute) * normal
        ball_specs.append(
            {
                "body": body,
                "radius": float(radius),
                "mass": mass,
                "spawn_time": float(i * spawn_interval),
                "released": False,
                "release_pos": release_pos,
                "name": f"ball_{i:03d}",
            }
        )
        ball_objects.append(
            {
                "name": f"ball_{i:03d}",
                "kind": "circle",
                "size": [float(radius)],
                "mass": mass,
                "color": _rgb(0.26 + 0.18 * float(i / max(1, n_balls - 1)), 0.68, 0.90),
                "positions": [],
                "scalars": {
                    "speed_mag": [],
                    "omega_mag": [],
                    "radius": [],
                    "contact_force_mag": [],
                    "in_flight": [],
                    "visible": [],
                    "phase_id": [],
                    "visual_scalar": [],
                },
                "color_metric": "speed_mag",
            }
        )

    def _sphere_box_signed_distance(point: np.ndarray, box: dict[str, Any]) -> float:
        rel = np.asarray(point, dtype=float) - np.asarray(box["center"], dtype=float)
        ang = -float(box["yaw_rad"])
        c = math.cos(ang)
        s = math.sin(ang)
        local = np.asarray([c * rel[0] - s * rel[2], rel[1], s * rel[0] + c * rel[2]], dtype=float)
        q = np.abs(local) - 0.5 * np.asarray(box["size"], dtype=float)
        outside = np.linalg.norm(np.maximum(q, 0.0))
        inside = min(float(np.max(q)), 0.0)
        return float(outside + inside)

    step_count = max(1, int(math.ceil(duration / max(dt, 1.0e-9))))
    next_output_t = 0.0
    times: list[float] = []
    contact_pairs: list[float] = []
    stage_chute_count: list[float] = []
    stage_flight_count: list[float] = []
    stage_box_count: list[float] = []
    stage_rest_count: list[float] = []
    release_clearance = max(5.5 * radius_mean, 0.24)
    log_interval_steps = max(1, int(round(0.05 / dt)))
    peak_contacts = 0
    peak_ke = 0.0

    def _record_frame(current_t: float) -> None:
        times.append(float(current_t))
        for sobj in static_objects:
            body = sobj["_body"]
            sobj["positions"].append([float(body.GetPos().x), float(body.GetPos().y), float(body.GetPos().z)])
            rot = body.GetRot()
            sobj["quaternions"].append([float(rot.e0), float(rot.e1), float(rot.e2), float(rot.e3)])

        released_ids = [i for i, spec in enumerate(ball_specs) if spec["released"]]
        positions = {i: _chrono_vec_np(ball_specs[i]["body"].GetPos()) for i in released_ids}
        velocities = {i: _chrono_vec_np(ball_specs[i]["body"].GetPosDt()) for i in released_ids}
        omegas = {
            i: (_chrono_vec_np(ball_specs[i]["body"].GetAngVelParent()) if hasattr(ball_specs[i]["body"], "GetAngVelParent") else np.zeros(3, dtype=float))
            for i in released_ids
        }
        pair_contacts = 0
        contact_force = {i: 0.0 for i in released_ids}
        phase_map = {i: 0 for i in range(n_balls)}

        for i in released_ids:
            spec = ball_specs[i]
            pos = positions[i]
            radius = float(spec["radius"])
            speed = float(np.linalg.norm(velocities[i]))
            chute_hit = False
            collector_hit = False
            static_contacts = 0
            for box in contact_boxes:
                signed_distance = _sphere_box_signed_distance(pos, box)
                if signed_distance <= radius + 0.003:
                    penetration = max(0.0, radius - signed_distance)
                    contact_force[i] += 8.0e3 * penetration + 80.0
                    static_contacts += 1
                    if box["category"] == "chute":
                        chute_hit = True
                    if box["category"] in {"collector", "ground"}:
                        collector_hit = True
            for j in released_ids:
                if j <= i:
                    continue
                diff = positions[i] - positions[j]
                limit = float(ball_specs[i]["radius"] + ball_specs[j]["radius"] + 0.004)
                dist = float(np.linalg.norm(diff))
                if dist <= limit:
                    pair_contacts += 1
                    penetration = max(0.0, limit - dist)
                    mag = 6.0e3 * penetration + 45.0
                    contact_force[i] += mag
                    contact_force[j] = contact_force.get(j, 0.0) + mag

            if speed < 0.16 and collector_hit and contact_force[i] > 1.0:
                phase = 4
            elif collector_hit:
                phase = 3
            elif chute_hit and pos[0] <= chute_exit[0] + 0.10:
                phase = 1
            elif static_contacts == 0 and contact_force[i] < 1.0:
                phase = 2
            else:
                phase = 3
            phase_map[i] = phase

        active_chute = 0
        active_flight = 0
        active_box = 0
        active_rest = 0
        for i, out in enumerate(ball_objects):
            spec = ball_specs[i]
            if not spec["released"]:
                out["positions"].append([float(hidden_anchor[0]), float(hidden_anchor[1]), float(hidden_anchor[2])])
                out["scalars"]["speed_mag"].append(0.0)
                out["scalars"]["omega_mag"].append(0.0)
                out["scalars"]["radius"].append(float(spec["radius"]))
                out["scalars"]["contact_force_mag"].append(0.0)
                out["scalars"]["in_flight"].append(0.0)
                out["scalars"]["visible"].append(0.0)
                out["scalars"]["phase_id"].append(0.0)
                out["scalars"]["visual_scalar"].append(0.0)
                continue

            pos = positions[i]
            vel = velocities[i]
            omg = omegas[i]
            speed = float(np.linalg.norm(vel))
            omega_mag = float(np.linalg.norm(omg))
            phase = int(phase_map.get(i, 0))
            if phase == 1:
                active_chute += 1
            elif phase == 2:
                active_flight += 1
            elif phase == 3:
                active_box += 1
            elif phase == 4:
                active_rest += 1
            cforce = float(contact_force.get(i, 0.0))
            out["positions"].append([float(pos[0]), float(pos[1]), float(pos[2])])
            out["scalars"]["speed_mag"].append(speed)
            out["scalars"]["omega_mag"].append(omega_mag)
            out["scalars"]["radius"].append(float(spec["radius"]))
            out["scalars"]["contact_force_mag"].append(cforce)
            out["scalars"]["in_flight"].append(1.0 if phase == 2 else 0.0)
            out["scalars"]["visible"].append(1.0)
            out["scalars"]["phase_id"].append(float(phase))
            out["scalars"]["visual_scalar"].append(speed)

        contact_pairs.append(float(pair_contacts))
        stage_chute_count.append(float(active_chute))
        stage_flight_count.append(float(active_flight))
        stage_box_count.append(float(active_box))
        stage_rest_count.append(float(active_rest))

    _record_frame(0.0)
    for step_idx in range(step_count):
        now = float(system.GetChTime())
        for spec in ball_specs:
            if spec["released"] or now + 1.0e-12 < float(spec["spawn_time"]):
                continue
            lane_clear = True
            cand_pos = np.asarray(spec["release_pos"], dtype=float)
            for other in ball_specs:
                if not other["released"]:
                    continue
                opos = _chrono_vec_np(other["body"].GetPos())
                rel = opos - cand_pos
                along = float(np.dot(rel, tangent))
                lateral_sep = abs(float(np.dot(rel, binormal)))
                normal_sep = abs(float(np.dot(rel, normal)))
                near_release_gate = (-0.18 <= along <= release_clearance)
                same_lane = lateral_sep <= max(0.18, 2.5 * float(spec["radius"] + other["radius"]))
                same_height_band = normal_sep <= max(0.16, 2.0 * float(spec["radius"] + other["radius"]))
                if near_release_gate and same_lane and same_height_band:
                    lane_clear = False
                    break
            if not lane_clear:
                break
            body = spec["body"]
            body.SetPos(_chrono_vec(chrono, spec["release_pos"]))
            body.SetPosDt(_chrono_vec(chrono, [0.0, 0.0, 0.0]))
            body.SetRot(chrono.QUNIT)
            body.SetFixed(False)
            spec["released"] = True
            break
        system.DoStepDynamics(dt)
        t_after = float(system.GetChTime())
        contacts_now = int(system.GetNumContacts()) if hasattr(system, "GetNumContacts") else 0
        peak_contacts = max(peak_contacts, contacts_now)
        total_ke = 0.0
        released_count = 0
        for spec in ball_specs:
            if not spec["released"]:
                continue
            released_count += 1
            body = spec["body"]
            vel = _chrono_vec_np(body.GetPosDt())
            omega = _chrono_vec_np(body.GetAngVelParent()) if hasattr(body, "GetAngVelParent") else np.zeros(3, dtype=float)
            mass = float(spec["mass"])
            radius = float(spec["radius"])
            inertia = (2.0 / 5.0) * mass * radius * radius
            total_ke += 0.5 * mass * float(np.dot(vel, vel)) + 0.5 * inertia * float(np.dot(omega, omega))
        peak_ke = max(peak_ke, total_ke)
        if step_idx % log_interval_steps == 0:
            first_active = next((spec for spec in ball_specs if spec["released"]), None)
            z_first = float(first_active["body"].GetPos().z) if first_active is not None else float("nan")
            print(
                f"[ballChutePile3D][diag] step={step_idx} t={t_after:.4f} dt={dt:.1e} "
                f"contacts={contacts_now} released={released_count} KE={total_ke:.6f} z0={z_first:.4f}"
            )
        if t_after + 1.0e-12 >= next_output_t + export_interval:
            _record_frame(t_after)
            next_output_t += export_interval

    if not times or abs(times[-1] - float(system.GetChTime())) > 1.0e-9:
        _record_frame(float(system.GetChTime()))

    for sobj in static_objects:
        sobj.pop("_body", None)

    objects = static_objects + ball_objects
    scene = _scene_oblique(_case_name(case, Path(".")), ground_value=float(ground_z))
    scene["camera_yaw_deg"] = float(mech.get("camera_yaw_deg", 36.0))
    scene["camera_pitch_deg"] = float(mech.get("camera_pitch_deg", 25.0))
    scene["bounds_min_world_z"] = -0.02
    scene.update(_scene_bounds(objects, scene["view_plane"], scene))
    zoom = float(mech.get("camera_zoom", 1.85))
    if zoom > 1.0 and isinstance(scene.get("xlim"), list) and isinstance(scene.get("ylim"), list):
        cx = 0.5 * (float(scene["xlim"][0]) + float(scene["xlim"][1]))
        cy = 0.5 * (float(scene["ylim"][0]) + float(scene["ylim"][1]))
        hx = 0.5 * (float(scene["xlim"][1]) - float(scene["xlim"][0])) / zoom
        hy = 0.5 * (float(scene["ylim"][1]) - float(scene["ylim"][0])) / zoom
        scene["xlim"] = [cx - hx, cx + hx]
        scene["ylim"] = [cy - hy, cy + hy]
    return {
        "times": times,
        "objects": objects,
        "scene": scene,
        "global_series": {
            "contact_pairs": contact_pairs,
            "stage_chute_count": stage_chute_count,
            "stage_flight_count": stage_flight_count,
            "stage_box_count": stage_box_count,
            "stage_rest_count": stage_rest_count,
        },
        "field_probes": {
            "points": probe_points,
            "meta": {
                "type": "fea_probe_grid",
                "description": "Fixed ChNodeFEAxyz probe grid used for velocity-cloud export.",
            },
        },
        "diagnostics": {
            "dt": float(dt),
            "friction": float(friction),
            "restitution": float(restitution),
            "rolling_friction": float(rolling_friction),
            "contact_young_modulus_pa": float(contact_young),
            "linear_damping": float(linear_damping),
            "angular_damping": float(angular_damping),
            "peak_contacts": int(peak_contacts),
            "peak_total_kinetic_energy_j": float(peak_ke),
        },
    }
