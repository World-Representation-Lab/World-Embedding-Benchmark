from __future__ import annotations
from run_case import (Any, Path, _add_csv_export, _box_track, _case_name, _circle_track, _duration, _gravity, _mech_cfg, _rgb, _sample_count, _scene_oblique, _step_size, math, np)

def _simulate_surrogate_drop_weight_impact(case: dict[str, Any]) -> dict[str, Any]:
    mech = _mech_cfg(case)
    g = abs(float(_gravity(case)[2]))
    m = float(mech.get("sphere_mass_kg", 0.5))
    r = float(mech.get("sphere_radius_m", 0.02))
    h0 = float(mech.get("drop_height_m", 0.05))
    kn = float(mech.get("smc_kn", 1.0e6))
    cn = float(mech.get("smc_cn", 180.0))
    plate_k = float(mech.get("plate_k", 6.0e4))
    plate_c = float(mech.get("plate_c", 120.0))
    plate_m = float(mech.get("plate_m", 1.8))
    plate_z0 = float(mech.get("plate_base_z_m", 0.16))
    vz0 = float(mech.get("initial_velocity_z_m_s", -0.6))
    dt = _step_size(case)
    duration = max(_duration(case), 4.5)
    times = np.linspace(0.0, duration, max(480, _sample_count(case)))

    # z axis points upward; initialize sphere center at height h0 above the undeformed plate top.
    z = plate_z0 + h0 + r
    vz = vz0
    # q: plate center deflection (downward positive), qdot: downward velocity.
    q = 0.0
    qdot = 0.0
    sphere_pos = np.zeros((len(times), 3), dtype=float)
    force_hist: list[list[float]] = []
    center_q: list[float] = []

    sim_t = 0.0
    fc = 0.0
    for i, t in enumerate(times):
        while sim_t + 1.0e-12 < float(t):
            h_step = min(dt, float(t) - sim_t)
            z_plate = plate_z0 - q
            v_plate = -qdot
            pen = max(0.0, (z_plate + r) - z)
            rel_v = v_plate - vz
            # SMC normal force in compression only (no tensile attraction).
            fc = max(0.0, kn * pen + cn * rel_v) if pen > 0.0 else 0.0
            az = -g + fc / max(m, 1.0e-9)
            qdd = (fc - plate_c * qdot - plate_k * q) / max(plate_m, 1.0e-9)
            vz += az * h_step
            z += vz * h_step
            qdot += qdd * h_step
            q += qdot * h_step
            sim_t += h_step
        sphere_pos[i] = [0.0, 0.0, z]
        force_hist.append([float(t), float(fc)])
        center_q.append(float(q))

    nx = max(6, int(mech.get("plate_nodes_x", 12)))
    ny = max(6, int(mech.get("plate_nodes_y", 12))
    )
    plate_t = float(mech.get("plate_thickness_m", 0.01))
    size = float(mech.get("plate_size_m", 0.3))
    xs = np.linspace(-0.5 * size, 0.5 * size, nx)
    ys = np.linspace(-0.5 * size, 0.5 * size, ny)
    dx = float(xs[1] - xs[0]) if len(xs) >= 2 else size
    dy = float(ys[1] - ys[0]) if len(ys) >= 2 else size
    objects: list[dict[str, Any]] = []
    center_q_arr = np.asarray(center_q, dtype=float)
    # This demo is an elastic plate impact model, not a plasticity model.
    # Render the signed spring-damper plate response directly so post-rebound
    # motion is vibration, not an artificial permanent dent.
    render_q_arr = center_q_arr * max(1.0, float(mech.get("deformation_visual_gain", 1.90)))
    for ix in range(len(xs) - 1):
        for iy in range(len(ys) - 1):
            xv = 0.5 * float(xs[ix] + xs[ix + 1])
            yv = 0.5 * float(ys[iy] + ys[iy + 1])
            rr = math.sqrt(xv * xv + yv * yv) / max(0.5 * size, 1.0e-9)
            shape = max(0.0, (1.0 - rr * rr)) ** 2
            z_node = plate_z0 - render_q_arr * shape
            pos = np.column_stack([np.full_like(times, xv), np.full_like(times, yv), z_node])
            disp_series = np.abs(z_node - plate_z0)
            objects.append(
                _box_track(
                    name=f"plate_elem_{ix:02d}_{iy:02d}",
                    size=[0.92 * dx, 0.92 * dy, max(0.002, 0.8 * plate_t)],
                    mass=0.02,
                    color=_rgb(0.34, 0.62, 0.86),
                    positions=pos,
                    scalars={"displacement_m": [float(v) for v in disp_series]},
                    extras={"color_metric": "displacement_m", "scalar_smoothing": True, "scalar_smooth_group": "drop_plate"},
                )
            )
    objects.append(_circle_track(name="impactor", radius=r, mass=m, color=_rgb(0.92, 0.44, 0.18), positions=sphere_pos))
    data = {"times": times.tolist(), "objects": objects, "scene": _scene_oblique(_case_name(case, Path(".")), ground_value=0.0)}
    data["visualization"] = {"plate_model": "elastic_spring_damper_no_plasticity", "rendered_deformation": "signed_elastic_displacement"}
    _add_csv_export(data, "contact_force.csv", ["time_s", "contact_force_n"], force_hist)
    return data
