from __future__ import annotations
from run_case import (Any, Path, _box_track, _case_name, _cross2, _duration, _mech_cfg, _rect_polygon, _rgb, _signed_gap_poly_poly, _step_size, _times, math, np)

def _simulate_surrogate_rigid_collision(case: dict[str, Any]) -> dict[str, Any]:
    mech = _mech_cfg(case)
    body_a = mech.get("body_a") if isinstance(mech.get("body_a"), dict) else {}
    body_b = mech.get("body_b") if isinstance(mech.get("body_b"), dict) else {}
    size_a = np.asarray(body_a.get("size", [0.7, 0.5, 0.4]), dtype=float)
    size_b = np.asarray(body_b.get("size", [0.9, 0.6, 0.5]), dtype=float)
    pos_a0 = np.asarray(body_a.get("position", [-2.0, 0.0, size_a[2] * 0.5]), dtype=float)
    pos_b0 = np.asarray(body_b.get("position", [2.0, 0.0, size_b[2] * 0.5]), dtype=float)
    vel_a0 = np.asarray(body_a.get("velocity", [3.0, 0.0, 0.0]), dtype=float)
    vel_b0 = np.asarray(body_b.get("velocity", [-1.0, 0.0, 0.0]), dtype=float)
    m1 = float(body_a.get("mass", 5.0))
    m2 = float(body_b.get("mass", 7.0))
    e = float(mech.get("restitution", 0.4))
    yaw_a0 = float(body_a.get("initial_yaw_deg", 0.0))
    yaw_b0 = float(body_b.get("initial_yaw_deg", 0.0))
    dt = _step_size(case)
    times = _times(_duration(case), dt)

    contact_clearance = float(mech.get("contact_clearance", 0.002))
    rel_vx = vel_a0[0] - vel_b0[0]

    def _signed_gap_at(t: float) -> float:
        pa = pos_a0 + vel_a0 * float(t)
        pb = pos_b0 + vel_b0 * float(t)
        poly_a = _rect_polygon(np.asarray([float(pa[0]), float(pa[1])], dtype=float), np.asarray([size_a[0], size_a[1]], dtype=float), yaw_a0)
        poly_b = _rect_polygon(np.asarray([float(pb[0]), float(pb[1])], dtype=float), np.asarray([size_b[0], size_b[1]], dtype=float), yaw_b0)
        return float(_signed_gap_poly_poly(poly_a, poly_b))

    t_candidate: float | None = None
    prev_t = 0.0
    prev_gap = _signed_gap_at(prev_t) - contact_clearance
    for t_probe in np.linspace(0.0, float(times[-1]), 640):
        gap = _signed_gap_at(float(t_probe)) - contact_clearance
        if prev_gap > 0.0 and gap <= 0.0:
            lo = prev_t
            hi = float(t_probe)
            for _ in range(32):
                mid = 0.5 * (lo + hi)
                if _signed_gap_at(mid) - contact_clearance > 0.0:
                    lo = mid
                else:
                    hi = mid
            t_candidate = 0.5 * (lo + hi)
            break
        if gap <= 0.0:
            t_candidate = float(t_probe)
            break
        prev_t = float(t_probe)
        prev_gap = gap
    half_y_sum = 0.5 * (float(size_a[1]) + float(size_b[1]))
    sep_y_at_hit = (
        (pos_b0[1] + vel_b0[1] * t_candidate) - (pos_a0[1] + vel_a0[1] * t_candidate)
        if t_candidate is not None
        else pos_b0[1] - pos_a0[1]
    )
    overlap_y_at_hit = abs(sep_y_at_hit) <= max(half_y_sum, 1.0e-9)
    t_hit = t_candidate if overlap_y_at_hit else None

    pos_a = np.zeros((len(times), 3), dtype=float)
    pos_b = np.zeros((len(times), 3), dtype=float)
    yaw_a = np.full(len(times), yaw_a0, dtype=float)
    yaw_b = np.full(len(times), yaw_b0, dtype=float)
    va_post = vel_a0.copy()
    vb_post = vel_b0.copy()
    omega_a = 0.0
    omega_b = 0.0

    def _unit_or(default: np.ndarray, vec: np.ndarray) -> np.ndarray:
        norm = float(np.linalg.norm(vec))
        return vec / norm if norm > 1.0e-10 else default

    def _support_vec(size_xy: np.ndarray, yaw_deg: float, direction: np.ndarray) -> np.ndarray:
        ang = math.radians(float(yaw_deg))
        ux = np.asarray([math.cos(ang), math.sin(ang)], dtype=float)
        uy = np.asarray([-math.sin(ang), math.cos(ang)], dtype=float)
        sx = 0.5 * float(size_xy[0])
        sy = 0.5 * float(size_xy[1])
        return ux * (sx if float(np.dot(ux, direction)) >= 0.0 else -sx) + uy * (sy if float(np.dot(uy, direction)) >= 0.0 else -sy)

    def _cross2(a: np.ndarray, b: np.ndarray) -> float:
        return float(a[0] * b[1] - a[1] * b[0])

    contact_a = pos_a0 + vel_a0 * (t_hit or 0.0)
    contact_b = pos_b0 + vel_b0 * (t_hit or 0.0)
    if t_hit is not None:
        pa_xy = contact_a[:2].astype(float)
        pb_xy = contact_b[:2].astype(float)
        va_xy = vel_a0[:2].astype(float)
        vb_xy = vel_b0[:2].astype(float)
        normal = _unit_or(np.asarray([1.0, 0.0], dtype=float), pb_xy - pa_xy)
        if float(np.dot(vb_xy - va_xy, normal)) > 0.0:
            normal = -normal
        tangent = np.asarray([-normal[1], normal[0]], dtype=float)

        # Approximate the active contact patch at the overlap midpoint along
        # the tangent. This is what turns an off-centre hit into angular
        # momentum instead of an unphysical one-dimensional velocity swap.
        sep_t = float(np.dot(pb_xy - pa_xy, tangent))
        half_t_a = 0.5 * max(float(size_a[0]), float(size_a[1]))
        half_t_b = 0.5 * max(float(size_b[0]), float(size_b[1]))
        eccentric_scale = float(mech.get("collision_eccentricity_scale", 0.18))
        r_a = _support_vec(size_a[:2], yaw_a0, normal) + tangent * float(np.clip(eccentric_scale * sep_t, -0.55 * half_t_a, 0.55 * half_t_a))
        r_b = _support_vec(size_b[:2], yaw_b0, -normal) - tangent * float(np.clip(eccentric_scale * sep_t, -0.55 * half_t_b, 0.55 * half_t_b))
        inertia_a = max(1.0e-9, (m1 / 12.0) * (float(size_a[0]) ** 2 + float(size_a[1]) ** 2))
        inertia_b = max(1.0e-9, (m2 / 12.0) * (float(size_b[0]) ** 2 + float(size_b[1]) ** 2))

        rel_v = vb_xy - va_xy
        rel_n = float(np.dot(rel_v, normal))
        denom_n = (1.0 / max(m1, 1.0e-9)) + (1.0 / max(m2, 1.0e-9)) + (_cross2(r_a, normal) ** 2) / inertia_a + (_cross2(r_b, normal) ** 2) / inertia_b
        impulse_n = max(0.0, -(1.0 + e) * rel_n / max(denom_n, 1.0e-9))
        va_xy = va_xy - (impulse_n / max(m1, 1.0e-9)) * normal
        vb_xy = vb_xy + (impulse_n / max(m2, 1.0e-9)) * normal
        omega_a = omega_a - impulse_n * _cross2(r_a, normal) / inertia_a
        omega_b = omega_b + impulse_n * _cross2(r_b, normal) / inertia_b

        mu = max(0.0, min(0.12, float(mech.get("friction", 0.045))))
        rel_v_t = (vb_xy + omega_b * np.asarray([-r_b[1], r_b[0]], dtype=float)) - (va_xy + omega_a * np.asarray([-r_a[1], r_a[0]], dtype=float))
        rel_t = float(np.dot(rel_v_t, tangent))
        denom_t = (1.0 / max(m1, 1.0e-9)) + (1.0 / max(m2, 1.0e-9)) + (_cross2(r_a, tangent) ** 2) / inertia_a + (_cross2(r_b, tangent) ** 2) / inertia_b
        impulse_t = 0.45 * float(np.clip(-rel_t / max(denom_t, 1.0e-9), -mu * impulse_n, mu * impulse_n))
        va_xy = va_xy - (impulse_t / max(m1, 1.0e-9)) * tangent
        vb_xy = vb_xy + (impulse_t / max(m2, 1.0e-9)) * tangent
        omega_a = omega_a - impulse_t * _cross2(r_a, tangent) / inertia_a
        omega_b = omega_b + impulse_t * _cross2(r_b, tangent) / inertia_b

        desired_sep = max(0.0, -e * rel_n)
        post_sep = float(np.dot(vb_xy - va_xy, normal))
        if post_sep < 0.70 * desired_sep:
            corr = (0.88 * desired_sep - post_sep) / max((1.0 / max(m1, 1.0e-9)) + (1.0 / max(m2, 1.0e-9)), 1.0e-9)
            va_xy = va_xy - (corr / max(m1, 1.0e-9)) * normal
            vb_xy = vb_xy + (corr / max(m2, 1.0e-9)) * normal

        if abs(float(rel_vx)) > 1.0e-9:
            va_1d_x = ((m1 - e * m2) / (m1 + m2)) * float(vel_a0[0]) + (((1.0 + e) * m2) / (m1 + m2)) * float(vel_b0[0])
            vb_1d_x = (((1.0 + e) * m1) / (m1 + m2)) * float(vel_a0[0]) + ((m2 - e * m1) / (m1 + m2)) * float(vel_b0[0])
            x_blend = float(mech.get("x_restitution_blend", 0.82))
            va_xy[0] = x_blend * va_1d_x + (1.0 - x_blend) * va_xy[0]
            vb_xy[0] = x_blend * vb_1d_x + (1.0 - x_blend) * vb_xy[0]

        max_omega = float(mech.get("max_post_spin_rad_s", 1.25 + 0.12 * abs(rel_n)))
        omega_a = float(np.clip(omega_a, -max_omega, max_omega))
        omega_b = float(np.clip(omega_b, -max_omega, max_omega))

        va_post[:2] = va_xy
        vb_post[:2] = vb_xy

        # Move the bodies a tiny distance apart along the collision normal so
        # the rendered contact frame does not show visible interpenetration.
        clearance_vec = normal * max(contact_clearance, 0.004)
        contact_a[:2] = contact_a[:2] - clearance_vec
        contact_b[:2] = contact_b[:2] + clearance_vec

    for i, t in enumerate(times):
        if t_hit is None or t < t_hit:
            pos_a[i] = pos_a0 + vel_a0 * t
            pos_b[i] = pos_b0 + vel_b0 * t
        else:
            tau = t - t_hit
            pos_a[i] = contact_a + va_post * tau
            pos_b[i] = contact_b + vb_post * tau
            yaw_a[i] = yaw_a0 + math.degrees(omega_a * tau)
            yaw_b[i] = yaw_b0 + math.degrees(omega_b * tau)

    view_center = 0.5 * (contact_a[:2] + contact_b[:2])
    body_span_x = float(size_a[0] + size_b[0])
    body_span_y = float(size_a[1] + size_b[1])
    view_half_y = max(0.38, 0.60 * body_span_y, 0.22 * body_span_x)
    view_half_x = max(0.78, 1.35 * view_half_y, 0.78 * body_span_x)

    return {
        "times": times.tolist(),
        "objects": [
            _box_track(name="body_a", size=size_a.tolist(), mass=m1, color=_rgb(0.19, 0.52, 0.83), positions=pos_a, scalars={"yaw_deg": yaw_a.tolist()}),
            _box_track(name="body_b", size=size_b.tolist(), mass=m2, color=_rgb(0.89, 0.39, 0.17), positions=pos_b, scalars={"yaw_deg": yaw_b.tolist()}),
        ],
        "scene": {
            "view_plane": "xy",
            "ground_value": None,
            "title": _case_name(case, Path(".")),
            "xlim": [float(view_center[0] - view_half_x), float(view_center[0] + view_half_x)],
            "ylim": [float(view_center[1] - view_half_y), float(view_center[1] + view_half_y)],
            "auto_bounds": False,
            "axis_background": "figure",
            "preserve_bounds_aspect": True,
            "disable_axes_clip": True,
        },
    }
