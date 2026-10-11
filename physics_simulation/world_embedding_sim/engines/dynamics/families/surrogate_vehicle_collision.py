from __future__ import annotations
from run_case import (Any, Optional, Path, _case_name, _cross2, _duration, _j2_stress_from_eq_strain, _load_vehicle_specs, _mech_cfg, _rgb, _scene_oblique, _step_size, _times, _vehicle_profile_from_case, _vehicle_track, math, np)

def _simulate_surrogate_vehicle_collision(case: dict[str, Any]) -> dict[str, Any]:
    mech = _mech_cfg(case)
    spec = _load_vehicle_specs(_vehicle_profile_from_case(case))
    vehicle_a = mech.get("vehicle_a") if isinstance(mech.get("vehicle_a"), dict) else {}
    vehicle_b = mech.get("vehicle_b") if isinstance(mech.get("vehicle_b"), dict) else {}
    pos_a0 = np.asarray(vehicle_a.get("position", [-16.0, 0.0, 0.0]), dtype=float)
    pos_b0 = np.asarray(vehicle_b.get("position", [16.0, 0.0, 0.0]), dtype=float)
    vel_a0 = np.asarray(vehicle_a.get("velocity", [10.0, 0.0, 0.0]), dtype=float)
    vel_b0 = np.asarray(vehicle_b.get("velocity", [-10.0, 0.0, 0.0]), dtype=float)
    yaw_a0 = float(vehicle_a.get("initial_yaw_deg", 0.0))
    yaw_b0 = float(vehicle_b.get("initial_yaw_deg", 180.0))
    m1 = float(spec["mass"])
    m2 = float(spec["mass"])
    e = float(mech.get("restitution", 0.08))
    j2_young_mpa = float(mech.get("j2_young_mpa", 210000.0))
    j2_yield_mpa = float(mech.get("j2_yield_mpa", 320.0))
    j2_hardening_mpa = float(mech.get("j2_hardening_mpa", 1600.0))
    j2_strain_scale = float(mech.get("j2_strain_scale", 0.22))
    plastic_disp_scale = max(0.05, float(mech.get("plastic_disp_scale", 0.30)))
    dt = _step_size(case)
    times = _times(_duration(case), dt)

    radius_a = 0.5 * float(np.linalg.norm(np.asarray(spec["chassis_size"][:2], dtype=float)))
    radius_b = radius_a
    rel_p = pos_b0[:2] - pos_a0[:2]
    rel_v = vel_b0[:2] - vel_a0[:2]
    a = float(np.dot(rel_v, rel_v))
    b = float(2.0 * np.dot(rel_p, rel_v))
    c = float(np.dot(rel_p, rel_p) - (radius_a + radius_b) ** 2)
    t_hit: Optional[float] = None
    if a > 1.0e-9:
        disc = b * b - 4.0 * a * c
        if disc >= 0.0:
            sqrt_disc = math.sqrt(disc)
            roots = sorted(t for t in [(-b - sqrt_disc) / (2.0 * a), (-b + sqrt_disc) / (2.0 * a)] if t >= 0.0)
            if roots:
                t_hit = float(roots[0])

    va_post = vel_a0[:2].copy()
    vb_post = vel_b0[:2].copy()
    yaw_rate_a = 0.0
    yaw_rate_b = 0.0
    if t_hit is not None and t_hit <= float(times[-1]):
        pa_hit = pos_a0[:2] + vel_a0[:2] * t_hit
        pb_hit = pos_b0[:2] + vel_b0[:2] * t_hit
        normal = pb_hit - pa_hit
        nmag = float(np.linalg.norm(normal))
        if nmag < 1.0e-6:
            normal = np.asarray([1.0, 0.0], dtype=float)
        else:
            normal /= nmag
        tangent = np.asarray([-normal[1], normal[0]], dtype=float)
        va_n = float(np.dot(vel_a0[:2], normal))
        vb_n = float(np.dot(vel_b0[:2], normal))
        va_t = float(np.dot(vel_a0[:2], tangent))
        vb_t = float(np.dot(vel_b0[:2], tangent))
        va_n_post = ((m1 - e * m2) / (m1 + m2)) * va_n + (((1.0 + e) * m2) / (m1 + m2)) * vb_n
        vb_n_post = (((1.0 + e) * m1) / (m1 + m2)) * va_n + ((m2 - e * m1) / (m1 + m2)) * vb_n
        tangential_scale = max(0.75, 1.0 - 0.08 * float(mech.get("friction", 0.6)))
        va_post = va_n_post * normal + tangential_scale * va_t * tangent
        vb_post = vb_n_post * normal + tangential_scale * vb_t * tangent
        spin = 0.3 * _cross2(normal, vel_b0[:2] - vel_a0[:2])
        yaw_rate_a = -math.degrees(spin / max(radius_a, 1.0e-6))
        yaw_rate_b = math.degrees(spin / max(radius_b, 1.0e-6))

    pos_a = np.zeros((len(times), 3), dtype=float)
    pos_b = np.zeros((len(times), 3), dtype=float)
    yaw_a: list[float] = []
    yaw_b: list[float] = []
    vm_a: list[float] = []
    vm_b: list[float] = []
    peeq_a: list[float] = []
    peeq_b: list[float] = []
    hotspot_a: list[float] = []
    hotspot_b: list[float] = []
    contact_a = pos_a0[:2] + vel_a0[:2] * (t_hit or 0.0)
    contact_b = pos_b0[:2] + vel_b0[:2] * (t_hit or 0.0)
    for i, t in enumerate(times):
        if t_hit is None or t <= t_hit:
            pa_xy = pos_a0[:2] + vel_a0[:2] * t
            pb_xy = pos_b0[:2] + vel_b0[:2] * t
            yaw_a.append(float(yaw_a0))
            yaw_b.append(float(yaw_b0))
        else:
            tau = t - t_hit
            pa_xy = contact_a + va_post * tau
            pb_xy = contact_b + vb_post * tau
            yaw_a.append(float(yaw_a0 + yaw_rate_a * tau))
            yaw_b.append(float(yaw_b0 + yaw_rate_b * tau))
        pos_a[i] = [float(pa_xy[0]), float(pa_xy[1]), float(pos_a0[2])]
        pos_b[i] = [float(pb_xy[0]), float(pb_xy[1]), float(pos_b0[2])]
        dist_xy = float(np.linalg.norm(pb_xy - pa_xy))
        overlap = max(0.0, (radius_a + radius_b) - dist_xy)
        crush_ratio = overlap / max(radius_a + radius_b, 1.0e-9)
        sigma_a, ep_a = _j2_stress_from_eq_strain(
            j2_strain_scale * crush_ratio,
            young_mpa=j2_young_mpa,
            yield_mpa=j2_yield_mpa,
            hardening_mpa=j2_hardening_mpa,
        )
        sigma_b, ep_b = _j2_stress_from_eq_strain(
            j2_strain_scale * crush_ratio,
            young_mpa=j2_young_mpa,
            yield_mpa=j2_yield_mpa,
            hardening_mpa=j2_hardening_mpa,
        )
        vm_a.append(float(sigma_a))
        vm_b.append(float(sigma_b))
        peeq_a.append(float(ep_a))
        peeq_b.append(float(ep_b))
        # Approximate local impact hotspot along body longitudinal axis [0,1].
        len_ref = max(1.0e-6, float(spec["chassis_size"][0]))
        rel_ab = np.asarray(pb_xy, dtype=float) - np.asarray(pa_xy, dtype=float)
        ca = math.cos(math.radians(float(yaw_a[-1])))
        sa = math.sin(math.radians(float(yaw_a[-1])))
        cb = math.cos(math.radians(float(yaw_b[-1])))
        sb = math.sin(math.radians(float(yaw_b[-1])))
        x_local_a = ca * rel_ab[0] + sa * rel_ab[1]
        x_local_b = cb * (-rel_ab[0]) + sb * (-rel_ab[1])
        hotspot_a.append(float(max(0.05, min(0.95, 0.5 + x_local_a / len_ref))))
        hotspot_b.append(float(max(0.05, min(0.95, 0.5 + x_local_b / len_ref))))
    disp_a = [float(v) for v in np.linalg.norm(pos_a - pos_a[0], axis=1)]
    disp_b = [float(v) for v in np.linalg.norm(pos_b - pos_b[0], axis=1)]

    return {
        "times": times.tolist(),
        "objects": [
            _vehicle_track(
                name="vehicle_a",
                spec=spec,
                color=_rgb(0.15, 0.46, 0.82),
                positions=pos_a,
                yaw_deg=yaw_a,
                scalars_extra={
                    "displacement_m": disp_a,
                    "von_mises_mpa": vm_a,
                    "PEEQ": peeq_a,
                    "stress_hotspot_x_norm": hotspot_a,
                },
                color_metric="von_mises_mpa",
            ),
            _vehicle_track(
                name="vehicle_b",
                spec=spec,
                color=_rgb(0.87, 0.36, 0.15),
                positions=pos_b,
                yaw_deg=yaw_b,
                scalars_extra={
                    "displacement_m": disp_b,
                    "von_mises_mpa": vm_b,
                    "PEEQ": peeq_b,
                    "stress_hotspot_x_norm": hotspot_b,
                },
                color_metric="von_mises_mpa",
            ),
        ],
        "scene": _scene_oblique(_case_name(case, Path(".")), ground_value=0.0),
    }
