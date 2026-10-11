from __future__ import annotations
from run_case import (Any, Path, _BMW_E90_FALLBACK, _case_name, _chrono_add_bmw_vehicle, _chrono_init_bmw_vehicle_scene, _chrono_quat_z, _chrono_vec, _chrono_vehicle_pose, _chrono_vehicle_visual_offset, _chrono_yaw_deg, _driver_sequence_entries, _j2_stress_from_eq_strain, _mech_cfg, _rgb, _sample_count, _scene_bounds, _scene_oblique, _signed_gap_poly_poly, _vehicle_profile_from_case, _vehicle_world_outline_xy, math, np)

def _simulate_chrono_vehicle_collision_full(case: dict[str, Any], chrono: Any) -> dict[str, Any]:
    mech = _mech_cfg(case)
    profile = _vehicle_profile_from_case(case)
    veh, system, terrain, spec, step, duration, tire_enum, collision_enum = _chrono_init_bmw_vehicle_scene(case, chrono)
    terrain.Initialize()
    vehicle_cfg_a = mech.get("vehicle_a") if isinstance(mech.get("vehicle_a"), dict) else {}
    vehicle_cfg_b = mech.get("vehicle_b") if isinstance(mech.get("vehicle_b"), dict) else {}
    wrapper_a = _chrono_add_bmw_vehicle(veh, chrono, system, step, tire_enum, collision_enum, vehicle_cfg_a)
    wrapper_b = _chrono_add_bmw_vehicle(veh, chrono, system, step, tire_enum, collision_enum, vehicle_cfg_b)

    zero_inputs = _driver_sequence_entries(veh, [], duration)
    driver_a = veh.ChDataDriver(wrapper_a.GetVehicle(), zero_inputs)
    driver_b = veh.ChDataDriver(wrapper_b.GetVehicle(), zero_inputs)
    driver_a.Initialize()
    driver_b.Initialize()

    vehicle_a_ref = wrapper_a.GetVehicle()
    vehicle_b_ref = wrapper_b.GetVehicle()
    body_a = vehicle_a_ref.GetChassisBody()
    body_b = vehicle_b_ref.GetChassisBody()
    visual_offset_a = _chrono_vehicle_visual_offset(vehicle_a_ref)
    visual_offset_b = _chrono_vehicle_visual_offset(vehicle_b_ref)

    settle_time = max(0.0, float(mech.get("pre_collision_settle_time", 0.45)))
    enforce_planar = bool(mech.get("enforce_planar_chassis_motion", True))
    z_target_a = float(vehicle_cfg_a.get("position", [0.0, 0.0, 0.6])[2])
    z_target_b = float(vehicle_cfg_b.get("position", [0.0, 0.0, 0.6])[2])

    def _apply_planar_stabilizer() -> None:
        if not enforce_planar:
            return
        for body, z_target in ((body_a, z_target_a), (body_b, z_target_b)):
            pos = body.GetPos()
            vel = body.GetPosDt()
            body.SetPos(_chrono_vec(chrono, [pos.x, pos.y, z_target]))
            body.SetPosDt(_chrono_vec(chrono, [vel.x, vel.y, 0.0]))
            body.SetRot(_chrono_quat_z(chrono, math.radians(_chrono_yaw_deg(body))))
            if hasattr(body, "SetAngVelParent"):
                ang = body.GetAngVelParent()
                body.SetAngVelParent(_chrono_vec(chrono, [0.0, 0.0, ang.z]))
    if settle_time > 0.0:
        zero_vec = _chrono_vec(chrono, [0.0, 0.0, 0.0])
        body_a.SetPosDt(zero_vec)
        body_b.SetPosDt(zero_vec)
        if hasattr(body_a, "SetAngVelParent"):
            body_a.SetAngVelParent(zero_vec)
        if hasattr(body_b, "SetAngVelParent"):
            body_b.SetAngVelParent(zero_vec)
        settle_steps = max(1, int(math.ceil(settle_time / max(step, 1.0e-6))))
        for _ in range(settle_steps):
            sim_time = float(system.GetChTime())
            inputs_a = driver_a.GetInputs()
            inputs_b = driver_b.GetInputs()
            driver_a.Synchronize(sim_time)
            driver_b.Synchronize(sim_time)
            wrapper_a.Synchronize(sim_time, inputs_a, terrain)
            wrapper_b.Synchronize(sim_time, inputs_b, terrain)
            terrain.Synchronize(sim_time)
            driver_a.Advance(step)
            driver_b.Advance(step)
            wrapper_a.Advance(step)
            wrapper_b.Advance(step)
            terrain.Advance(step)
            system.DoStepDynamics(step)
            _apply_planar_stabilizer()

    body_a.SetPosDt(_chrono_vec(chrono, vehicle_cfg_a.get("velocity", [0.0, 0.0, 0.0])))
    body_b.SetPosDt(_chrono_vec(chrono, vehicle_cfg_b.get("velocity", [0.0, 0.0, 0.0])))
    if hasattr(body_a, "SetAngVelParent"):
        body_a.SetAngVelParent(_chrono_vec(chrono, [0.0, 0.0, 0.0]))
    if hasattr(body_b, "SetAngVelParent"):
        body_b.SetAngVelParent(_chrono_vec(chrono, [0.0, 0.0, 0.0]))
    _apply_planar_stabilizer()
    t0 = float(system.GetChTime())
    target_vel_a = np.asarray(vehicle_cfg_a.get("velocity", [0.0, 0.0, 0.0]), dtype=float)
    target_vel_b = np.asarray(vehicle_cfg_b.get("velocity", [0.0, 0.0, 0.0]), dtype=float)
    speed_hold_enabled = bool(mech.get("pre_collision_speed_hold", True))
    hold_ratio = max(0.0, min(1.0, float(mech.get("pre_collision_hold_ratio", 0.95))))
    hold_time = float(mech.get("pre_collision_hold_time", 0.0))
    if speed_hold_enabled and hold_time <= 0.0:
        p0_a = np.asarray(vehicle_cfg_a.get("position", [0.0, 0.0, 0.0]), dtype=float)
        p0_b = np.asarray(vehicle_cfg_b.get("position", [0.0, 0.0, 0.0]), dtype=float)
        rel_p = p0_b - p0_a
        rel_v = target_vel_b - target_vel_a
        rv2 = float(np.dot(rel_v, rel_v))
        if rv2 > 1.0e-9:
            t_est = max(0.0, -float(np.dot(rel_p, rel_v)) / rv2)
            hold_time = hold_ratio * t_est
        else:
            hold_time = min(duration, 0.6 * duration)
    hold_time = max(0.0, min(duration, hold_time))
    hold_until_gap = float(mech.get("pre_collision_hold_until_gap", 0.10))
    hold_disarmed = False

    total_steps = max(2, int(math.ceil(duration / max(step, 1.0e-6))))
    stride = max(1, total_steps // max(2, _sample_count(case) - 1))
    contact_outline_xy_src = [list(p) for p in spec.get("contact_outline_xy", spec["outline_xy"])]
    contact_outline_xz_src = [list(p) for p in spec.get("contact_outline_xz", spec.get("outline_xz", _BMW_E90_FALLBACK["outline_xz"]))]
    contact_shrink_x = max(0.5, min(1.0, float(mech.get("contact_outline_shrink_x", 0.90))))
    contact_shrink_y = max(0.5, min(1.0, float(mech.get("contact_outline_shrink_y", 0.88))))
    contact_shrink_z = max(0.5, min(1.0, float(mech.get("contact_outline_shrink_z", 0.92))))
    contact_outline_xy = [[float(p[0]) * contact_shrink_x, float(p[1]) * contact_shrink_y] for p in contact_outline_xy_src]
    contact_outline_xz = [[float(p[0]) * contact_shrink_x, float(p[1]) * contact_shrink_z] for p in contact_outline_xz_src]
    wheel_width_scale = max(0.8, min(2.6, float(mech.get("wheel_width_scale", 1.0))))
    dx_a, dy_a, dz_a = float(visual_offset_a[0]), float(visual_offset_a[1]), float(visual_offset_a[2])
    dx_b, dy_b, dz_b = float(visual_offset_b[0]), float(visual_offset_b[1]), float(visual_offset_b[2])
    local_contact_a = np.asarray([[float(p[0] + dx_a), float(p[1] + dy_a)] for p in contact_outline_xy], dtype=float)
    local_contact_b = np.asarray([[float(p[0] + dx_b), float(p[1] + dy_b)] for p in contact_outline_xy], dtype=float)

    objects = [
        {
            "name": "vehicle_a",
            "kind": "box",
            "size": list(spec["chassis_size"]),
            "mass": float(spec["mass"]),
            "color": _rgb(0.15, 0.46, 0.82),
            "positions": [],
            "scalars": {"yaw_deg": [], "crush_ratio": [], "displacement_m": [], "von_mises_mpa": [], "PEEQ": [], "stress_hotspot_x_norm": []},
            "color_metric": "displacement_m",
            "render_shape": "vehicle_mesh",
            "outline_xy": [[float(p[0] + dx_a), float(p[1] + dy_a)] for p in spec["outline_xy"]],
            "outline_xz": [[float(p[0] + dx_a), float(p[1] + dz_a)] for p in spec.get("outline_xz", _BMW_E90_FALLBACK["outline_xz"])],
            "contact_outline_xy": [list(p) for p in contact_outline_xy],
            "contact_outline_xz": [list(p) for p in contact_outline_xz],
            "wheel_offsets_xy": [[float(p[0] + dx_a), float(p[1] + dy_a)] for p in spec["wheel_offsets_xy"]],
            "wheel_size_xy": [float(spec["wheel_size_xy"][0]), float(spec["wheel_size_xy"][1])],
            "wheel_width_scale": wheel_width_scale,
            "stress_hotspot_width": float(mech.get("stress_hotspot_width", 0.10)),
            "vehicle_profile": profile,
        },
        {
            "name": "vehicle_b",
            "kind": "box",
            "size": list(spec["chassis_size"]),
            "mass": float(spec["mass"]),
            "color": _rgb(0.87, 0.36, 0.15),
            "positions": [],
            "scalars": {"yaw_deg": [], "crush_ratio": [], "displacement_m": [], "von_mises_mpa": [], "PEEQ": [], "stress_hotspot_x_norm": []},
            "color_metric": "displacement_m",
            "render_shape": "vehicle_mesh",
            "outline_xy": [[float(p[0] + dx_b), float(p[1] + dy_b)] for p in spec["outline_xy"]],
            "outline_xz": [[float(p[0] + dx_b), float(p[1] + dz_b)] for p in spec.get("outline_xz", _BMW_E90_FALLBACK["outline_xz"])],
            "contact_outline_xy": [list(p) for p in contact_outline_xy],
            "contact_outline_xz": [list(p) for p in contact_outline_xz],
            "wheel_offsets_xy": [[float(p[0] + dx_b), float(p[1] + dy_b)] for p in spec["wheel_offsets_xy"]],
            "wheel_size_xy": [float(spec["wheel_size_xy"][0]), float(spec["wheel_size_xy"][1])],
            "wheel_width_scale": wheel_width_scale,
            "stress_hotspot_width": float(mech.get("stress_hotspot_width", 0.10)),
            "vehicle_profile": profile,
        },
    ]
    times: list[float] = []
    damage_a = 0.0
    damage_b = 0.0
    max_crush = max(0.0, min(0.65, float(mech.get("max_crush_ratio", 0.22))))
    crush_gain = max(0.0, float(mech.get("crush_gain", 0.18)))
    crush_trigger_gap = float(mech.get("crush_trigger_gap", 0.06))
    j2_young_mpa = float(mech.get("j2_young_mpa", 210000.0))
    j2_yield_mpa = float(mech.get("j2_yield_mpa", 320.0))
    j2_hardening_mpa = float(mech.get("j2_hardening_mpa", 1600.0))
    j2_strain_scale = float(mech.get("j2_strain_scale", 0.22))
    plastic_disp_scale = max(0.05, float(mech.get("plastic_disp_scale", 0.30)))
    max_visual_penetration = max(0.0, float(mech.get("max_visual_penetration", 0.08)))
    for step_idx in range(total_steps + 1):
        _apply_planar_stabilizer()
        elapsed = float(system.GetChTime() - t0)
        pa = np.asarray([body_a.GetPos().x, body_a.GetPos().y], dtype=float)
        pb = np.asarray([body_b.GetPos().x, body_b.GetPos().y], dtype=float)
        yaw_a_body = _chrono_yaw_deg(body_a)
        yaw_b_body = _chrono_yaw_deg(body_b)
        poly_a_pre = _vehicle_world_outline_xy(local_contact_a, pa, yaw_a_body)
        poly_b_pre = _vehicle_world_outline_xy(local_contact_b, pb, yaw_b_body)
        gap_pre = float(_signed_gap_poly_poly(poly_a_pre, poly_b_pre))
        if gap_pre <= hold_until_gap:
            hold_disarmed = True
        keep_speed = speed_hold_enabled and (not hold_disarmed) and (elapsed <= hold_time or gap_pre > hold_until_gap)
        if keep_speed:
            vel_a = body_a.GetPosDt()
            vel_b = body_b.GetPosDt()
            body_a.SetPosDt(_chrono_vec(chrono, [float(target_vel_a[0]), float(target_vel_a[1]), vel_a.z]))
            body_b.SetPosDt(_chrono_vec(chrono, [float(target_vel_b[0]), float(target_vel_b[1]), vel_b.z]))
            if hasattr(body_a, "SetAngVelParent"):
                body_a.SetAngVelParent(_chrono_vec(chrono, [0.0, 0.0, 0.0]))
            if hasattr(body_b, "SetAngVelParent"):
                body_b.SetAngVelParent(_chrono_vec(chrono, [0.0, 0.0, 0.0]))
        pa = np.asarray([body_a.GetPos().x, body_a.GetPos().y], dtype=float)
        pb = np.asarray([body_b.GetPos().x, body_b.GetPos().y], dtype=float)
        yaw_a_body = _chrono_yaw_deg(body_a)
        yaw_b_body = _chrono_yaw_deg(body_b)
        poly_a = _vehicle_world_outline_xy(local_contact_a, pa, yaw_a_body)
        poly_b = _vehicle_world_outline_xy(local_contact_b, pb, yaw_b_body)
        va = np.asarray([body_a.GetPosDt().x, body_a.GetPosDt().y], dtype=float)
        vb = np.asarray([body_b.GetPosDt().x, body_b.GetPosDt().y], dtype=float)
        rel_p = pb - pa
        rel_v = vb - va
        dist = float(np.linalg.norm(rel_p))
        n = rel_p / max(dist, 1.0e-9)
        closing = max(0.0, -float(np.dot(rel_v, n)))
        gap = float(_signed_gap_poly_poly(poly_a, poly_b))
        penetration = max(0.0, -gap)
        if gap < crush_trigger_gap and (closing > 0.0 or penetration > 1.0e-6) and crush_gain > 0.0:
            inc = crush_gain * (closing + 6.0 * penetration) * step * (1.0 + 0.35 * penetration)
            damage_a = min(max_crush, damage_a + 0.75 * inc)
            damage_b = min(max_crush, damage_b + 1.00 * inc)
        if step_idx == 0 or step_idx == total_steps or step_idx % stride == 0:
            times.append(float(system.GetChTime() - t0))
            pos_a, yaw_a = _chrono_vehicle_pose(vehicle_a_ref)
            pos_b, yaw_b = _chrono_vehicle_pose(vehicle_b_ref)
            pos_a_arr = np.asarray(pos_a, dtype=float)
            pos_b_arr = np.asarray(pos_b, dtype=float)
            # Visualization-only anti-interpenetration correction (does not feed back to dynamics).
            for _ in range(8):
                poly_a_sample = _vehicle_world_outline_xy(local_contact_a, pos_a_arr[:2], yaw_a)
                poly_b_sample = _vehicle_world_outline_xy(local_contact_b, pos_b_arr[:2], yaw_b)
                gap_sample = float(_signed_gap_poly_poly(poly_a_sample, poly_b_sample))
                if gap_sample >= -max_visual_penetration:
                    break
                rel_xy = pos_b_arr[:2] - pos_a_arr[:2]
                rel_norm = float(np.linalg.norm(rel_xy))
                if rel_norm <= 1.0e-9:
                    break
                n_xy = rel_xy / rel_norm
                corr = 0.62 * ((-gap_sample) - max_visual_penetration) * n_xy
                pos_a_arr[0] -= float(corr[0])
                pos_a_arr[1] -= float(corr[1])
                pos_b_arr[0] += float(corr[0])
                pos_b_arr[1] += float(corr[1])
            objects[0]["positions"].append([float(pos_a_arr[0]), float(pos_a_arr[1]), float(pos_a_arr[2])])
            objects[0]["scalars"]["yaw_deg"].append(yaw_a)
            objects[0]["scalars"]["crush_ratio"].append(float(damage_a))
            objects[0]["scalars"]["displacement_m"].append(float(damage_a * float(spec["chassis_size"][0]) * plastic_disp_scale))
            sigma_a, ep_a = _j2_stress_from_eq_strain(
                j2_strain_scale * float(damage_a),
                young_mpa=j2_young_mpa,
                yield_mpa=j2_yield_mpa,
                hardening_mpa=j2_hardening_mpa,
            )
            objects[0]["scalars"]["von_mises_mpa"].append(float(sigma_a))
            objects[0]["scalars"]["PEEQ"].append(float(ep_a))
            len_ref = max(1.0e-6, float(spec["chassis_size"][0]))
            rel_ab = pos_b_arr[:2] - pos_a_arr[:2]
            ca = math.cos(math.radians(float(yaw_a)))
            sa = math.sin(math.radians(float(yaw_a)))
            x_local_a = ca * float(rel_ab[0]) + sa * float(rel_ab[1])
            objects[0]["scalars"]["stress_hotspot_x_norm"].append(float(max(0.05, min(0.95, 0.5 + x_local_a / len_ref))))
            objects[1]["positions"].append([float(pos_b_arr[0]), float(pos_b_arr[1]), float(pos_b_arr[2])])
            objects[1]["scalars"]["yaw_deg"].append(yaw_b)
            objects[1]["scalars"]["crush_ratio"].append(float(damage_b))
            objects[1]["scalars"]["displacement_m"].append(float(damage_b * float(spec["chassis_size"][0]) * plastic_disp_scale))
            sigma_b, ep_b = _j2_stress_from_eq_strain(
                j2_strain_scale * float(damage_b),
                young_mpa=j2_young_mpa,
                yield_mpa=j2_yield_mpa,
                hardening_mpa=j2_hardening_mpa,
            )
            objects[1]["scalars"]["von_mises_mpa"].append(float(sigma_b))
            objects[1]["scalars"]["PEEQ"].append(float(ep_b))
            cb = math.cos(math.radians(float(yaw_b)))
            sb = math.sin(math.radians(float(yaw_b)))
            x_local_b = cb * float(-rel_ab[0]) + sb * float(-rel_ab[1])
            objects[1]["scalars"]["stress_hotspot_x_norm"].append(float(max(0.05, min(0.95, 0.5 + x_local_b / len_ref))))
        if step_idx == total_steps:
            break
        sim_time = float(system.GetChTime())
        inputs_a = driver_a.GetInputs()
        inputs_b = driver_b.GetInputs()
        driver_a.Synchronize(sim_time)
        driver_b.Synchronize(sim_time)
        wrapper_a.Synchronize(sim_time, inputs_a, terrain)
        wrapper_b.Synchronize(sim_time, inputs_b, terrain)
        terrain.Synchronize(sim_time)
        driver_a.Advance(step)
        driver_b.Advance(step)
        wrapper_a.Advance(step)
        wrapper_b.Advance(step)
        terrain.Advance(step)
        system.DoStepDynamics(step)
        _apply_planar_stabilizer()

    scene = _scene_oblique(_case_name(case, Path(".")), ground_value=0.0)
    scene.update(_scene_bounds(objects, scene["view_plane"], scene))
    return {"times": times, "objects": objects, "scene": scene}
