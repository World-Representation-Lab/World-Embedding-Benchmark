from __future__ import annotations
from run_case import (Any, Path, _case_name, _circle_track, _duration, _mech_cfg, _rgb, _scene_oblique, _step_size, _times, math, np)

def _simulate_surrogate_granular_collapse(case: dict[str, Any]) -> dict[str, Any]:
    mech = _mech_cfg(case)
    radius = float(mech.get("particle_radius", 0.15))
    mass = float(mech.get("particle_mass", 0.28))
    nx = max(2, int(mech.get("nx", 4)))
    ny = max(1, int(mech.get("ny", 3)))
    nz = max(2, int(mech.get("nz", 5)))
    spacing = float(mech.get("spacing", 0.015))
    release = float(mech.get("release_time", 0.25))
    dt = _step_size(case)
    times = _times(_duration(case), dt)
    objects: list[dict[str, Any]] = []
    base_x = -2.0 * radius * (nx - 0.5)
    base_y = -0.5 * (ny - 1) * (2.0 * radius + spacing)
    rng = np.random.default_rng(int(case.get("seed", 0)))
    for ix in range(nx):
        for iy in range(ny):
            for iz in range(nz):
                x0 = base_x + ix * (2.0 * radius + spacing)
                y0 = base_y + iy * (2.0 * radius + spacing)
                z0 = radius + iz * (2.0 * radius + spacing)
                spread = 0.35 + 0.12 * iz + 0.06 * ix
                vx = spread * (1.0 + 0.06 * float(rng.normal()))
                vy = 0.14 * (iy - 0.5 * (ny - 1))
                positions = np.zeros((len(times), 3), dtype=float)
                for i, t in enumerate(times):
                    if t <= release:
                        positions[i] = [x0, y0, z0]
                    else:
                        tau = t - release
                        x = x0 + vx * (1.0 - math.exp(-1.4 * tau)) * (1.0 + 0.22 * tau)
                        y = y0 + vy * (1.0 - math.exp(-1.2 * tau))
                        z = max(radius, z0 - 0.5 * 4.4 * tau * tau)
                        positions[i] = [x, y, z]
                color = _rgb(0.16 + 0.02 * iz, 0.54 + 0.03 * iy, 0.76 - 0.04 * ix)
                objects.append(_circle_track(name=f"grain_{ix}_{iy}_{iz}", radius=radius, mass=mass, color=color, positions=positions))
    return {"times": times.tolist(), "objects": objects, "scene": _scene_oblique(_case_name(case, Path(".")), ground_value=0.0)}
