import json
import math
import os
import shutil
import sys
import tempfile
import builtins as _py
import xml.etree.ElementTree as ET
from glob import glob
from pathlib import Path

from paraview.simple import *
import paraview

from hex_mesh_utils import rewrite_vtu_with_vtk_hex_cells, vtu_has_dolfinx_hex_order

# Force offscreen safety and avoid implicit camera reset.
paraview.simple._DisableFirstRenderCameraReset()


case_dir = sys.argv[1]
out_mp4 = sys.argv[2]
out_root, _ = os.path.splitext(out_mp4)
out_name = os.path.basename(out_mp4).lower()
is_vel_render = out_name.endswith("_vel.mp4")
os.makedirs(os.path.dirname(out_mp4), exist_ok=True)

fps = int(os.environ.get("PV_FPS", "30"))
width = max(320, int(os.environ.get("PV_WIDTH", "1920")))
height = max(240, int(os.environ.get("PV_HEIGHT", "1080")))

duration_mode = os.environ.get("PV_DURATION_MODE", "auto").strip().lower()
fixed_duration_sec = float(os.environ.get("PV_DURATION_SEC", "10.0"))
min_duration_sec = float(os.environ.get("PV_MIN_DURATION_SEC", "4.0"))
max_duration_sec = float(os.environ.get("PV_MAX_DURATION_SEC", "16.0"))
hold_last_sec = float(os.environ.get("PV_HOLD_LAST_SEC", "1.5"))
steps_per_video_second = float(os.environ.get("PV_STEPS_PER_VIDEO_SECOND", "15.0"))
sim_seconds_per_video_second = float(os.environ.get("PV_SIM_SECONDS_PER_VIDEO_SECOND", "2.0"))

color_range_mode = os.environ.get("PV_COLOR_RANGE_MODE", "video").strip().lower()
color_max_uref_mult = float(os.environ.get("PV_COLOR_MAX_UREF_MULT", "2.5"))
manual_color_min = float(os.environ.get("PV_COLOR_MIN", "0.0"))
manual_color_max = float(os.environ.get("PV_COLOR_MAX", "0.0"))
video_range_samples = max(3, int(os.environ.get("PV_VIDEO_RANGE_SAMPLES", "12")))
video_range_quantile = min(1.0, max(0.5, float(os.environ.get("PV_VIDEO_RANGE_QUANTILE", "0.95"))))
video_range_pad = max(1.0, float(os.environ.get("PV_VIDEO_RANGE_PAD", "1.08")))


def env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    s = raw.strip().lower()
    return s not in {"0", "false", "off", "no", ""}


def _parse_hex_rgb(value: str, default_hex: str = "#726F69") -> list[float]:
    raw = (value or "").strip()
    if not raw:
        raw = default_hex
    if raw.startswith("#"):
        raw = raw[1:]
    if len(raw) == 3:
        raw = "".join(ch * 2 for ch in raw)
    if len(raw) != 6:
        raw = default_hex.lstrip("#")
    try:
        r = int(raw[0:2], 16) / 255.0
        g = int(raw[2:4], 16) / 255.0
        b = int(raw[4:6], 16) / 255.0
        return [float(r), float(g), float(b)]
    except Exception:
        raw = default_hex.lstrip("#")
        return [int(raw[0:2], 16) / 255.0, int(raw[2:4], 16) / 255.0, int(raw[4:6], 16) / 255.0]


def set_prop(proxy, key: str, value) -> None:
    try:
        setattr(proxy, key, value)
    except Exception:
        pass


def _curved_surface_projection_payload(case_meta: dict) -> dict:
    geom = case_meta.get("geometry") if isinstance(case_meta.get("geometry"), dict) else {}
    solver = case_meta.get("solver") if isinstance(case_meta.get("solver"), dict) else {}
    mesh = solver.get("mesh") if isinstance(solver.get("mesh"), dict) else {}
    lx = float(geom.get("lx", 1.0))
    ly = float(geom.get("ly", 1.0))
    lz = float(geom.get("lz", geom.get("thickness", 1.0)))
    nx = max(4, int(mesh.get("nx", 24)))
    ny = max(4, int(mesh.get("ny", 24)))
    nz = max(4, int(mesh.get("nz", 24)))
    hx = lx / max(nx, 1)
    hy = ly / max(ny, 1)
    hz = lz / max(nz, 1)
    return {
        "shape_3d": str(geom.get("shape_3d", "")).strip().lower(),
        "lx": lx,
        "ly": ly,
        "lz": lz,
        "sphere_radius": float(geom.get("sphere_radius", 0.45 * min(lx, ly, lz))),
        "sphere_cx": float(geom.get("sphere_cx", 0.5 * lx)),
        "sphere_cy": float(geom.get("sphere_cy", 0.0)),
        "sphere_cz": float(geom.get("sphere_cz", 0.0)),
        "cylinder_radius": float(geom.get("cylinder_radius", 0.42 * min(ly, lz))),
        "cylinder_cx": float(geom.get("cylinder_cx", 0.5 * lx)),
        "cylinder_cy": float(geom.get("cylinder_cy", 0.0)),
        "cylinder_cz": float(geom.get("cylinder_cz", 0.0)),
        "hole_radius": float(geom.get("hole_radius", 0.2 * min(lx, ly))),
        "hole_cx": float(geom.get("hole_cx", 0.5 * lx)),
        "hole_cy": float(geom.get("hole_cy", 0.0)),
        "hx": hx,
        "hy": hy,
        "hz": hz,
    }


def _curved_surface_projection_script(case_meta: dict) -> str:
    payload = json.dumps(_curved_surface_projection_payload(case_meta), sort_keys=True)
    return f"""
import numpy as np
import builtins as _builtins

meta = {payload}

src = inputs[0]
src_vtk = src.VTKObject if hasattr(src, "VTKObject") else src
output.ShallowCopy(src_vtk)
pts_arr = output.GetPoints()
if pts_arr is None:
    globals()["max"] = _builtins.max
    globals()["min"] = _builtins.min
    return
pts = np.array(pts_arr, dtype=float, copy=True)
shape = str(meta.get("shape_3d", "")).strip().lower()

if pts.size:
    if shape == "sphere":
        cx = float(meta.get("sphere_cx", 0.5 * float(meta.get("lx", 1.0))))
        cy = float(meta.get("sphere_cy", 0.0))
        cz = float(meta.get("sphere_cz", 0.0))
        r = float(meta.get("sphere_radius", 0.45))
        dv = pts - np.array([cx, cy, cz], dtype=float)[None, :]
        rr = np.linalg.norm(dv, axis=1)
        ok = rr > 1.0e-12
        if np.any(ok):
            pts[ok] = np.array([cx, cy, cz], dtype=float)[None, :] + dv[ok] * (r / rr[ok])[:, None]

    elif shape == "cylinder":
        lx = float(meta.get("lx", 1.0))
        cx = float(meta.get("cylinder_cx", 0.5 * lx))
        cy = float(meta.get("cylinder_cy", 0.0))
        cz = float(meta.get("cylinder_cz", 0.0))
        r = float(meta.get("cylinder_radius", 0.42))
        band = 1.25 * _builtins.max(float(meta.get("hx", 0.02)), float(meta.get("hy", 0.02)), float(meta.get("hz", 0.02)))
        x_left = cx - 0.5 * lx
        x_right = cx + 0.5 * lx
        on_left = pts[:, 0] <= (x_left + band)
        on_right = pts[:, 0] >= (x_right - band)
        side = ~(on_left | on_right)
        yy = pts[:, 1] - cy
        zz = pts[:, 2] - cz
        rr = np.sqrt(yy * yy + zz * zz)
        side_ok = side & (rr > 1.0e-12)
        if np.any(side_ok):
            scl = r / rr[side_ok]
            pts[side_ok, 1] = cy + yy[side_ok] * scl
            pts[side_ok, 2] = cz + zz[side_ok] * scl
        rim = (np.abs(rr - r) <= band) & (rr > 1.0e-12)
        if np.any(rim):
            scl = r / rr[rim]
            pts[rim, 1] = cy + yy[rim] * scl
            pts[rim, 2] = cz + zz[rim] * scl
        if np.any(on_left):
            pts[on_left, 0] = x_left
        if np.any(on_right):
            pts[on_right, 0] = x_right

    elif shape == "plate_with_hole_3d":
        lx = float(meta.get("lx", 1.0))
        ly = float(meta.get("ly", 1.0))
        hx = float(meta.get("hx", 0.02))
        hy = float(meta.get("hy", 0.02))
        cx = float(meta.get("hole_cx", 0.5 * lx))
        cy = float(meta.get("hole_cy", 0.0))
        r = float(meta.get("hole_radius", 0.2))
        band = 1.8 * _builtins.max(hx, hy)
        xx = pts[:, 0] - cx
        yy = pts[:, 1] - cy
        rr = np.sqrt(xx * xx + yy * yy)
        on_outer = (
            (pts[:, 0] <= 0.75 * hx)
            | (pts[:, 0] >= lx - 0.75 * hx)
            | (np.abs(pts[:, 1]) >= 0.5 * ly - 0.75 * hy)
        )
        sel = (np.abs(rr - r) <= band) & (~on_outer) & (rr > 1.0e-12)
        if np.any(sel):
            scl = r / rr[sel]
            pts[sel, 0] = cx + xx[sel] * scl
            pts[sel, 1] = cy + yy[sel] * scl

output.Points = pts
globals()["max"] = _builtins.max
globals()["min"] = _builtins.min
"""


def has_png_frames(pattern: str) -> bool:
    return len(glob(pattern)) > 0


def load_case_meta(case_dir_path: str) -> dict:
    meta_path = os.path.join(case_dir_path, "case_meta.json")
    if not os.path.exists(meta_path):
        return {}
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if isinstance(payload, dict):
            return payload
    except Exception:
        pass
    return {}


def fmt_num(val) -> str:
    try:
        return f"{float(val):.4g}"
    except Exception:
        return "n/a"


def _pick_evenly_spaced(values: list[float], count: int) -> list[float]:
    n = len(values)
    if n <= count:
        return list(values)
    if count <= 1:
        return [values[n // 2]]
    out = []
    seen = set()
    for i in range(count):
        idx = int(round(i * (n - 1) / (count - 1)))
        if idx not in seen:
            seen.add(idx)
            out.append(values[idx])
    return out


def _quantile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    vs = sorted(values)
    if len(vs) == 1:
        return float(vs[0])
    pos = max(0.0, min(1.0, q)) * (len(vs) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(vs[lo])
    frac = pos - lo
    return float(vs[lo] * (1.0 - frac) + vs[hi] * frac)


def _array_info(info, name: str):
    arr = None
    try:
        p_info = info.GetPointDataInformation()
        if p_info is not None:
            arr = p_info.GetArrayInformation(name)
    except Exception:
        arr = None
    if arr is not None:
        return "POINTS", arr
    try:
        c_info = info.GetCellDataInformation()
        if c_info is not None:
            arr = c_info.GetArrayInformation(name)
    except Exception:
        arr = None
    if arr is not None:
        return "CELLS", arr
    return None, None


def _array_max(arr, component: int = 0) -> float:
    try:
        lo, hi = arr.GetComponentRange(component)
        hi_f = float(hi)
        if math.isfinite(hi_f):
            return hi_f
    except Exception:
        pass
    return float("nan")


def _sample_range(proxy, times: list[float], n_samples: int, q: float, arr_name: str, arr_is_vector: bool):
    sample_times = _pick_evenly_spaced(times, n_samples)
    mins = []
    maxs = []
    for t in sample_times:
        proxy.UpdatePipeline(float(t))
        info = proxy.GetDataInformation()
        _, arr = _array_info(info, arr_name)
        if arr is None:
            continue

        if arr_is_vector:
            try:
                lo, hi = arr.GetComponentRange(-1)
            except Exception:
                lo, hi = arr.GetComponentRange(0)
        else:
            lo, hi = arr.GetComponentRange(0)

        mins.append(float(lo))
        maxs.append(float(hi))

    if not maxs:
        return None
    return min(mins), _quantile(maxs, q)


def _sample_bounds(proxy, times: list[float], n_samples: int):
    sample_times = _pick_evenly_spaced(times, n_samples)
    mins = None
    maxs = None
    for t in sample_times:
        proxy.UpdatePipeline(float(t))
        info = proxy.GetDataInformation()
        if info is None:
            continue
        try:
            b = info.GetBounds()
            vals = [float(v) for v in b]
        except Exception:
            continue
        if len(vals) != 6 or not _py.all(math.isfinite(v) for v in vals):
            continue
        xmin, xmax, ymin, ymax, zmin, zmax = vals
        cur_min = [xmin, ymin, zmin]
        cur_max = [xmax, ymax, zmax]
        if mins is None:
            mins = list(cur_min)
            maxs = list(cur_max)
        else:
            mins = [min(mins[i], cur_min[i]) for i in range(3)]
            maxs = [max(maxs[i], cur_max[i]) for i in range(3)]
    if mins is None or maxs is None:
        return None
    return mins, maxs


def _find_data_source(case_dir_path: str) -> str:
    cand = os.path.join(case_dir_path, "results", "displacement.pvd")
    if os.path.exists(cand):
        return cand
    pvds = sorted(glob(os.path.join(case_dir_path, "results", "*.pvd")))
    if pvds:
        return pvds[0]
    raise FileNotFoundError(f"No .pvd found under {case_dir_path}/results")


_legacy_source_tmp = None


def _prepare_source_for_reader(source_file_path: str) -> str:
    global _legacy_source_tmp

    source = Path(source_file_path).resolve()
    suffix = source.suffix.lower()
    if suffix == ".vtu":
        if not vtu_has_dolfinx_hex_order(source):
            return str(source)
        _legacy_source_tmp = tempfile.TemporaryDirectory(prefix="solid_pv_hex_fix_")
        dst = Path(_legacy_source_tmp.name) / source.name
        rewrite_vtu_with_vtk_hex_cells(source, dst)
        print(f"[render_paraview] repaired legacy hex ordering in {source.name}")
        return str(dst)

    if suffix != ".pvd":
        return str(source)

    tree = ET.parse(source)
    rows = tree.getroot().findall(".//DataSet")
    datasets: list[tuple[str, Path, bool]] = []
    needs_fix = False
    for row in rows:
        fn = str(row.get("file", "")).strip()
        if not fn:
            continue
        src_frame = (source.parent / fn).resolve()
        fix_frame = src_frame.suffix.lower() == ".vtu" and vtu_has_dolfinx_hex_order(src_frame)
        datasets.append((fn, src_frame, fix_frame))
        needs_fix = needs_fix or fix_frame

    if not needs_fix:
        return str(source)

    _legacy_source_tmp = tempfile.TemporaryDirectory(prefix="solid_pv_hex_fix_")
    tmp_root = Path(_legacy_source_tmp.name)
    for rel_name, src_frame, fix_frame in datasets:
        dst_frame = tmp_root / rel_name
        dst_frame.parent.mkdir(parents=True, exist_ok=True)
        if fix_frame:
            rewrite_vtu_with_vtk_hex_cells(src_frame, dst_frame)
        else:
            shutil.copy2(src_frame, dst_frame)

    dst_pvd = tmp_root / source.name
    tree.write(dst_pvd, encoding="unicode")
    n_fixed = sum(1 for _, _, fix_frame in datasets if fix_frame)
    print(f"[render_paraview] repaired legacy hex ordering in {n_fixed} VTU frame(s)")
    return str(dst_pvd)


source_file = _find_data_source(case_dir)
source_file = _prepare_source_for_reader(source_file)
case_meta = load_case_meta(case_dir)
case_name_full = os.path.basename(case_dir)
case_name = case_name_full.lower()
case_dim = str(case_meta.get("dim", "2D")).strip().upper() if isinstance(case_meta, dict) else "2D"
geom_meta = case_meta.get("geometry") if isinstance(case_meta.get("geometry"), dict) else {}
shape_3d = str(geom_meta.get("shape_3d", "")).strip().lower()
is_plate_hole_3d = (shape_3d == "plate_with_hole_3d") or ("platewithhole3d" in case_name)
family_name = str(case_meta.get("family", "")).strip()
is_crack_3d = (
    ("crackpropagationczm3d" in case_name)
    or ("notchedrodtension3d" in case_name)
    or ("cohesiveinterfacetension3d" in case_name)
    or (family_name in {"crackPropagationCZM3D", "notchedRodTension3D", "cohesiveInterfaceTension3D"})
)
is_center_notch_2d = ("centernotchplate2d" in case_name) or (family_name == "centerNotchPlate2D")
is_legacy_crack_3d = ("crackpropagationczm3d" in case_name) or (family_name == "crackPropagationCZM3D")
is_notched_rod_3d = ("notchedrodtension3d" in case_name) or (family_name == "notchedRodTension3D")
is_cohesive_interface_3d = ("cohesiveinterfacetension3d" in case_name) or (family_name == "cohesiveInterfaceTension3D")

reader = PVDReader(FileName=source_file)

view = GetActiveViewOrCreate("RenderView")
view.ViewSize = [width, height]
bg_rgb = _parse_hex_rgb(os.environ.get("PV_BG_COLOR", "#726F69"), default_hex="#726F69")
set_prop(view, "UseColorPaletteForBackground", 0)
set_prop(view, "UseGradientBackground", 0)
set_prop(view, "Background", bg_rgb)
set_prop(view, "Background2", bg_rgb)
set_prop(view, "OrientationAxesVisibility", 0)
set_prop(view, "CenterAxesVisibility", 0)
set_prop(view, "UseFXAA", 1)
set_prop(view, "MultiSamples", int(os.environ.get("PV_MSAA_SAMPLES", "8")))

scene = GetAnimationScene()
scene.UpdateAnimationUsingDataTimeSteps()
time_keeper = GetTimeKeeper()
times = list(getattr(time_keeper, "TimestepValues", []) or getattr(reader, "TimestepValues", []) or [])
if not times:
    times = [0.0]

reader.UpdatePipeline(float(times[0]))
info0_reader = reader.GetDataInformation()
assoc_u, arr_u = _array_info(info0_reader, "u")
can_warp = False
if arr_u is not None and assoc_u == "POINTS":
    try:
        can_warp = int(arr_u.GetNumberOfComponents()) > 1
    except Exception:
        can_warp = True

deform_user = os.environ.get("PV_DEFORM_SCALE", "").strip()
deform_mult = max(0.1, float(os.environ.get("PV_DEFORM_SCALE_MULT", "1.5")))
if can_warp:
    b0 = reader.GetDataInformation().GetBounds()
    sx0 = max(1e-9, float(b0[1] - b0[0]))
    sy0 = max(1e-9, float(b0[3] - b0[2]))
    sz0 = max(1e-9, float(b0[5] - b0[4]))
    max_span0 = max(sx0, sy0, sz0)
    if deform_user:
        deform_scale = max(0.0, float(deform_user))
    else:
        sampled_u = _sample_range(reader, times, video_range_samples, video_range_quantile, "u", True)
        max_u = float(sampled_u[1]) if sampled_u is not None else 0.0
        if max_u > 1e-12:
            deform_scale = max(0.02, min(1.0, 0.18 * max_span0 / max_u))
        else:
            deform_scale = 1.0
        deform_scale = max(0.02, min(2.5, deform_scale * deform_mult))
        if max_u > 1e-12:
            default_vis_caps = {
                "bucklingSnapthrough": 0.18,
                "necking": 0.14,
                "yieldHardening": 0.14,
            }
            cap_raw = os.environ.get("PV_DEFORM_VIS_FRAC_CAP", "").strip()
            vis_cap = float(cap_raw) if cap_raw else default_vis_caps.get(family_name)
            if vis_cap is not None and vis_cap > 0.0:
                deform_scale = min(deform_scale, float(vis_cap) * max_span0 / max_u)
    render_source = WarpByVector(Input=reader)
    set_prop(render_source, "Vectors", ["POINTS", "u"])
    set_prop(render_source, "ScaleFactor", deform_scale)
else:
    deform_scale = 1.0
    render_source = reader

curved_shape_proxy_applied = False
curved_shape_proxy_enable = False
curved_shape_proxy_subdivide_level = 0
curved_shape_smooth_applied = False
curved_shape_subdivide_applied = False
curved_shape_iters = 0
curved_shape_subdivide_level = 0
curved_shape_set = {"sphere", "cylinder", "plate_with_hole_3d"}
curved_shape_proxy_enable = env_flag(
    "PV_CURVED_SHAPE_ANALYTIC_PROXY_ENABLE",
    (case_dim == "3D") and (shape_3d in curved_shape_set),
)
if curved_shape_proxy_enable and (case_dim == "3D") and (shape_3d in curved_shape_set):
    try:
        try:
            surface_source = ExtractSurface(Input=reader)
        except Exception:
            surface_source = GeometryFilter(Input=reader)
        proxy_input = Triangulate(Input=surface_source)
        curved_shape_proxy_subdivide_level = max(
            0,
            int(
                os.environ.get(
                    "PV_CURVED_SHAPE_PROXY_SUBDIV_LEVEL",
                    "2" if shape_3d in {"sphere", "plate_with_hole_3d"} else "1",
                )
            ),
        )
        if curved_shape_proxy_subdivide_level > 0:
            for ctor in (globals().get("LinearSubdivision"), globals().get("LoopSubdivision")):
                if ctor is None:
                    continue
                try:
                    subdiv = ctor(Input=proxy_input)
                    set_prop(subdiv, "NumberofSubdivisions", curved_shape_proxy_subdivide_level)
                    set_prop(subdiv, "NumberOfSubdivisions", curved_shape_proxy_subdivide_level)
                    set_prop(subdiv, "Level", curved_shape_proxy_subdivide_level)
                    proxy_input = subdiv
                    curved_shape_subdivide_applied = True
                    break
                except Exception:
                    continue
        curved_project = ProgrammableFilter(Input=proxy_input)
        set_prop(curved_project, "CopyArrays", 1)
        set_prop(curved_project, "Script", _curved_surface_projection_script(case_meta))
        projected_surface = GenerateSurfaceNormals(Input=curved_project)
        set_prop(projected_surface, "Consistency", 1)
        set_prop(projected_surface, "Splitting", 0)
        set_prop(projected_surface, "FeatureAngle", float(os.environ.get("PV_CURVED_SHAPE_NORMAL_FEATURE_ANGLE", "110.0")))
        if can_warp:
            warped_surface = WarpByVector(Input=projected_surface)
            set_prop(warped_surface, "Vectors", ["POINTS", "u"])
            set_prop(warped_surface, "ScaleFactor", deform_scale)
            render_source = warped_surface
        else:
            render_source = projected_surface
        curved_shape_proxy_applied = True
    except Exception:
        curved_shape_proxy_applied = False

curved_shape_enable = env_flag(
    "PV_CURVED_SHAPE_SMOOTH_ENABLE",
    (case_dim == "3D") and (shape_3d in {"sphere", "cylinder"}) and (not curved_shape_proxy_applied),
)
if curved_shape_enable and (case_dim == "3D") and (shape_3d in {"sphere", "cylinder"}):
    try:
        try:
            surface_source = ExtractSurface(Input=render_source)
        except Exception:
            surface_source = GeometryFilter(Input=render_source)

        smooth_input = surface_source
        curved_shape_subdivide_level = max(
            0,
            int(
                os.environ.get(
                    "PV_CURVED_SHAPE_SUBDIV_LEVEL",
                    "2" if shape_3d == "sphere" else "1",
                )
            ),
        )
        if curved_shape_subdivide_level > 0:
            tri_source = surface_source
            try:
                tri_source = Triangulate(Input=surface_source)
            except Exception:
                tri_source = surface_source
            for ctor in (globals().get("LinearSubdivision"), globals().get("LoopSubdivision")):
                if ctor is None:
                    continue
                try:
                    subdiv = ctor(Input=tri_source)
                    set_prop(subdiv, "NumberofSubdivisions", curved_shape_subdivide_level)
                    set_prop(subdiv, "NumberOfSubdivisions", curved_shape_subdivide_level)
                    set_prop(subdiv, "Level", curved_shape_subdivide_level)
                    smooth_input = subdiv
                    curved_shape_subdivide_applied = True
                    break
                except Exception:
                    continue

        curved_shape_iters = max(
            0,
            int(
                os.environ.get(
                    "PV_CURVED_SHAPE_SMOOTH_ITERS",
                    "72" if shape_3d == "sphere" else "44",
                )
            ),
        )
        if curved_shape_iters > 0:
            smooth = Smooth(Input=smooth_input)
            set_prop(smooth, "NumberOfIterations", curved_shape_iters)
            set_prop(smooth, "NumberofIterations", curved_shape_iters)
            set_prop(smooth, "RelaxationFactor", float(os.environ.get("PV_CURVED_SHAPE_SMOOTH_RELAX", "0.12")))
            set_prop(smooth, "FeatureEdgeSmoothing", 0)
            set_prop(smooth, "BoundarySmoothing", 1)
            normals = GenerateSurfaceNormals(Input=smooth)
            set_prop(normals, "Consistency", 1)
            set_prop(normals, "Splitting", 0)
            set_prop(normals, "FeatureAngle", float(os.environ.get("PV_CURVED_SHAPE_NORMAL_FEATURE_ANGLE", "110.0")))
            render_source = normals
            curved_shape_smooth_applied = True
        else:
            render_source = smooth_input
    except Exception:
        curved_shape_smooth_applied = False
        curved_shape_subdivide_applied = False

# Cohesive-separation rendering controls.
sep_field_name = os.environ.get("PV_COH_SEP_FIELD", "coh_sep_n").strip() or "coh_sep_n"
sep_thresh = max(1e-9, float(os.environ.get("PV_COH_SEP_THRESH", "0.0025")))
sep_use_for_hole = env_flag("PV_COH_SEP_USE_FOR_HOLE", True)
sep_overlay_default = is_legacy_crack_3d
sep_overlay_enable = env_flag("PV_COH_SEP_OVERLAY", sep_overlay_default)
sep_overlay_opacity = min(1.0, max(0.05, float(os.environ.get("PV_COH_SEP_OVERLAY_OPACITY", "0.92"))))
sep_overlay_color = _parse_hex_rgb(os.environ.get("PV_COH_SEP_COLOR", "#1B1B1B"), default_hex="#1B1B1B")
render_source_pre_crack = render_source

# Keep full mesh by default: do not delete cells for fracture visualization.
crack_hole_default = False
crack_hole_enable = env_flag("PV_CRACK_TRANSPARENT_HOLE", crack_hole_default)
default_damage_cut = (
    "0.90"
    if (is_center_notch_2d or is_notched_rod_3d or is_cohesive_interface_3d)
    else "0.93"
)
crack_damage_cut = min(1.0, max(0.0, float(os.environ.get("PV_CRACK_DAMAGE_CUT", default_damage_cut))))
crack_mask_cut = min(1.0, max(0.0, float(os.environ.get("PV_CRACK_MASK_CUT", "0.5"))))
crack_progressive_default = is_legacy_crack_3d
crack_progressive_cut = env_flag("PV_CRACK_PROGRESSIVE_CUT", crack_progressive_default)
crack_hole_applied = False
crack_filtered_proxy = None
crack_cut_field = "damage"
crack_cut_assoc = "POINTS"
crack_cut_uses_damage = True
prefer_damage_hole = is_center_notch_2d or is_notched_rod_3d or is_cohesive_interface_3d
if crack_hole_enable and (is_crack_3d or is_center_notch_2d):
    render_source.UpdatePipeline(float(times[0]))
    info_crack = render_source.GetDataInformation()
    assoc_sep, sep_arr = _array_info(info_crack, sep_field_name)
    assoc_damage, damage_arr = _array_info(info_crack, "damage")
    assoc_aux, aux_arr = _array_info(info_crack, "aux_field")
    assoc_cmask, cmask_arr = _array_info(info_crack, "crack_mask")
    if is_crack_3d and (not prefer_damage_hole) and sep_use_for_hole and sep_arr is not None:
        crack_filtered = Threshold(Input=render_source)
        set_prop(crack_filtered, "Scalars", [assoc_sep, sep_field_name])
        set_prop(crack_filtered, "ThresholdMethod", "Below Upper Threshold")
        set_prop(crack_filtered, "LowerThreshold", -1.0e30)
        set_prop(crack_filtered, "UpperThreshold", sep_thresh)
        set_prop(crack_filtered, "AllScalars", 0)
        render_source = crack_filtered
        crack_filtered_proxy = crack_filtered
        crack_hole_applied = True
        crack_cut_field = sep_field_name
        crack_cut_assoc = assoc_sep
        crack_cut_uses_damage = False
    elif damage_arr is not None:
        crack_filtered = Threshold(Input=render_source)
        set_prop(crack_filtered, "Scalars", [assoc_damage, "damage"])
        set_prop(crack_filtered, "ThresholdMethod", "Below Upper Threshold")
        set_prop(crack_filtered, "LowerThreshold", -1.0e30)
        set_prop(crack_filtered, "UpperThreshold", crack_damage_cut)
        set_prop(crack_filtered, "AllScalars", 0)
        render_source = crack_filtered
        crack_filtered_proxy = crack_filtered
        crack_hole_applied = True
        crack_cut_field = "damage"
        crack_cut_assoc = assoc_damage
        crack_cut_uses_damage = True
    elif aux_arr is not None:
        crack_filtered = Threshold(Input=render_source)
        set_prop(crack_filtered, "Scalars", [assoc_aux, "aux_field"])
        set_prop(crack_filtered, "ThresholdMethod", "Below Upper Threshold")
        set_prop(crack_filtered, "LowerThreshold", -1.0e30)
        set_prop(crack_filtered, "UpperThreshold", crack_damage_cut)
        set_prop(crack_filtered, "AllScalars", 0)
        render_source = crack_filtered
        crack_filtered_proxy = crack_filtered
        crack_hole_applied = True
        crack_cut_field = "aux_field"
        crack_cut_assoc = assoc_aux
        crack_cut_uses_damage = True
    elif cmask_arr is not None:
        crack_filtered = Threshold(Input=render_source)
        set_prop(crack_filtered, "Scalars", [assoc_cmask, "crack_mask"])
        set_prop(crack_filtered, "ThresholdMethod", "Below Upper Threshold")
        set_prop(crack_filtered, "LowerThreshold", -1.0e30)
        set_prop(crack_filtered, "UpperThreshold", crack_mask_cut)
        set_prop(crack_filtered, "AllScalars", 0)
        render_source = crack_filtered
        crack_filtered_proxy = crack_filtered
        crack_hole_applied = True
        crack_cut_field = "crack_mask"
        crack_cut_assoc = assoc_cmask
        crack_cut_uses_damage = False

split_opening_enable = False
split_opening_scale = 0.0
split_opening_applied = False
split_opening_field = ""
split_opening_plane_x = float("nan")

# Keep geometry faithful to solver output for modern 3D fracture cases.
# Surface smoothing remains only for the legacy crack-propagation visual proxy.
crack_smooth_default = is_legacy_crack_3d and (not crack_hole_enable) and (not split_opening_enable)
crack_smooth_enable = env_flag("PV_CRACK_SMOOTH_SURFACE", crack_smooth_default)
crack_smooth_applied = False
crack_subdivide_level = max(0, int(float(os.environ.get("PV_CRACK_SUBDIV_LEVEL", "1"))))
crack_subdivide_applied = False
if crack_smooth_enable and is_crack_3d:
    try:
        crack_iters = max(0, int(float(os.environ.get("PV_CRACK_SMOOTH_ITERS", "20"))))
        crack_relax = max(0.0, min(1.0, float(os.environ.get("PV_CRACK_SMOOTH_RELAX", "0.12"))))
        crack_feat = env_flag("PV_CRACK_SMOOTH_FEATURE_EDGES", False)
        surf = ExtractSurface(Input=render_source)
        tri = Triangulate(Input=surf)
        smooth_input = tri
        if crack_subdivide_level > 0:
            for ctor_name in ("LinearSubdivision", "LoopSubdivision", "ButterflySubdivision", "Subdivide"):
                ctor = globals().get(ctor_name)
                if ctor is None:
                    continue
                try:
                    subdiv = ctor(Input=tri)
                    set_prop(subdiv, "NumberofSubdivisions", crack_subdivide_level)
                    set_prop(subdiv, "NumberOfSubdivisions", crack_subdivide_level)
                    set_prop(subdiv, "Level", crack_subdivide_level)
                    smooth_input = subdiv
                    crack_subdivide_applied = True
                    break
                except Exception:
                    continue
        smooth = Smooth(Input=smooth_input)
        set_prop(smooth, "NumberOfIterations", crack_iters)
        set_prop(smooth, "NumberofIterations", crack_iters)
        set_prop(smooth, "RelaxationFactor", crack_relax)
        set_prop(smooth, "FeatureEdgeSmoothing", 1 if crack_feat else 0)
        set_prop(smooth, "BoundarySmoothing", 1)
        normals = GenerateSurfaceNormals(Input=smooth)
        set_prop(normals, "Consistency", 1)
        set_prop(normals, "Splitting", 0)
        set_prop(normals, "FeatureAngle", float(os.environ.get("PV_CRACK_NORMAL_FEATURE_ANGLE", "70.0")))
        render_source = normals
        crack_smooth_applied = True
    except Exception:
        crack_smooth_applied = False

display = Show(render_source, view)
set_prop(display, "Representation", "Surface")
set_prop(display, "RenderLinesAsTubes", 0)
set_prop(display, "LineWidth", 1.0)
surface_opacity = float(os.environ.get("PV_SURFACE_OPACITY", "1.0"))
surface_opacity = min(1.0, max(0.05, surface_opacity))
set_prop(display, "Opacity", surface_opacity)
view.Update()
Render(view)

# Overlay crack path/opening trajectory.
crack_overlay_on_vel_default = is_crack_3d
crack_path_overlay_default = (
    ((is_crack_3d and ((not is_vel_render) or crack_overlay_on_vel_default)) or (is_center_notch_2d and (not is_vel_render)))
    and (not crack_hole_enable)
    and (not split_opening_enable)
)
crack_path_overlay_enable = env_flag("PV_CRACK_PATH_OVERLAY", crack_path_overlay_default)
crack_path_applied = False
if crack_path_overlay_enable:
    render_source_pre_crack.UpdatePipeline(float(times[0]))
    info_cp = render_source_pre_crack.GetDataInformation()
    crack_field = "crack_mask"
    crack_thr = 0.5
    assoc_cp, crack_arr = _array_info(info_cp, crack_field)
    crack_proxy = None

    if is_crack_3d:
        # Recommended fracture visualization: contour on high damage core.
        damage_iso_default = "0.95" if (is_notched_rod_3d or is_cohesive_interface_3d) else "0.58"
        damage_iso = min(
            0.995,
            max(0.05, float(os.environ.get("PV_CRACK_DAMAGE_ISO", damage_iso_default))),
        )
        assoc_damage, arr_damage = _array_info(info_cp, "damage")
        assoc_aux, arr_aux = _array_info(info_cp, "aux_field")
        assoc_sep, arr_sep = _array_info(info_cp, sep_field_name)

        if arr_damage is not None:
            crack_field = "damage"
            crack_thr = damage_iso
            assoc_cp = assoc_damage
            crack_arr = arr_damage
            if assoc_cp == "POINTS":
                try:
                    crack_proxy = Contour(Input=render_source_pre_crack)
                    set_prop(crack_proxy, "ContourBy", [assoc_cp, crack_field])
                    set_prop(crack_proxy, "Isosurfaces", [crack_thr])
                    set_prop(crack_proxy, "PointMergeMethod", "Uniform Binning")
                except Exception:
                    crack_proxy = None
            else:
                crack_proxy = Threshold(Input=render_source_pre_crack)
                set_prop(crack_proxy, "Scalars", [assoc_cp, crack_field])
                set_prop(crack_proxy, "ThresholdMethod", "Above Lower Threshold")
                set_prop(crack_proxy, "LowerThreshold", crack_thr)
                set_prop(crack_proxy, "UpperThreshold", 1.0e30)
                set_prop(crack_proxy, "AllScalars", 0)

        if crack_proxy is None and arr_aux is not None:
            crack_field = "aux_field"
            crack_thr = damage_iso
            assoc_cp = assoc_aux
            crack_arr = arr_aux
            if assoc_cp == "POINTS":
                try:
                    crack_proxy = Contour(Input=render_source_pre_crack)
                    set_prop(crack_proxy, "ContourBy", [assoc_cp, crack_field])
                    set_prop(crack_proxy, "Isosurfaces", [crack_thr])
                    set_prop(crack_proxy, "PointMergeMethod", "Uniform Binning")
                except Exception:
                    crack_proxy = None
            else:
                crack_proxy = Threshold(Input=render_source_pre_crack)
                set_prop(crack_proxy, "Scalars", [assoc_cp, crack_field])
                set_prop(crack_proxy, "ThresholdMethod", "Above Lower Threshold")
                set_prop(crack_proxy, "LowerThreshold", crack_thr)
                set_prop(crack_proxy, "UpperThreshold", 1.0e30)
                set_prop(crack_proxy, "AllScalars", 0)

        # Cohesive fallback: opening threshold if damage-like field is unavailable.
        if crack_proxy is None and is_cohesive_interface_3d and arr_sep is not None:
            crack_field = sep_field_name
            crack_thr = max(1e-9, float(os.environ.get("PV_COH_CRACK_OPENING_THRESH", str(sep_thresh))))
            assoc_cp = assoc_sep
            crack_arr = arr_sep
            crack_proxy = Threshold(Input=render_source_pre_crack)
            set_prop(crack_proxy, "Scalars", [assoc_cp, crack_field])
            set_prop(crack_proxy, "ThresholdMethod", "Above Lower Threshold")
            set_prop(crack_proxy, "LowerThreshold", crack_thr)
            set_prop(crack_proxy, "UpperThreshold", 1.0e30)
            set_prop(crack_proxy, "AllScalars", 0)

        if crack_proxy is None:
            assoc_cp, crack_arr = _array_info(info_cp, "crack_mask")
            crack_field = "crack_mask"
            crack_thr = 0.5
            if crack_arr is not None and assoc_cp == "POINTS":
                try:
                    crack_proxy = Contour(Input=render_source_pre_crack)
                    set_prop(crack_proxy, "ContourBy", [assoc_cp, crack_field])
                    set_prop(crack_proxy, "Isosurfaces", [crack_thr])
                    set_prop(crack_proxy, "PointMergeMethod", "Uniform Binning")
                except Exception:
                    crack_proxy = None
    else:
        if crack_arr is not None and assoc_cp == "POINTS":
            try:
                crack_proxy = Contour(Input=render_source_pre_crack)
                set_prop(crack_proxy, "ContourBy", ["POINTS", crack_field])
                set_prop(crack_proxy, "Isosurfaces", [crack_thr])
                set_prop(crack_proxy, "PointMergeMethod", "Uniform Binning")
            except Exception:
                crack_proxy = None

    if crack_proxy is None and crack_arr is not None:
        crack_proxy = Threshold(Input=render_source_pre_crack)
        set_prop(crack_proxy, "Scalars", [assoc_cp, crack_field])
        set_prop(crack_proxy, "ThresholdMethod", "Above Lower Threshold")
        set_prop(crack_proxy, "LowerThreshold", crack_thr)
        set_prop(crack_proxy, "UpperThreshold", 1.0e30)
        set_prop(crack_proxy, "AllScalars", 0)

    if crack_proxy is not None and crack_arr is not None:
        crack_display = Show(crack_proxy, view)
        crack_color_default = "#1A1A1A" if is_crack_3d else "#111111"
        crack_opacity_default = "1.0" if is_crack_3d else "0.88"
        crack_color = _parse_hex_rgb(
            os.environ.get("PV_CRACK_PATH_COLOR", crack_color_default),
            default_hex=crack_color_default,
        )
        crack_opacity = min(
            1.0,
            max(
                0.05,
                float(os.environ.get("PV_CRACK_PATH_OPACITY", crack_opacity_default)),
            ),
        )
        crack_width = float(os.environ.get("PV_CRACK_PATH_WIDTH", "2.2" if is_center_notch_2d else "1.3"))
        set_prop(crack_display, "Representation", "Surface")
        set_prop(crack_display, "ColorArrayName", [None, ""])
        set_prop(crack_display, "MapScalars", 0)
        set_prop(crack_display, "Lighting", 0 if is_crack_3d else 1)
        set_prop(crack_display, "AmbientColor", crack_color)
        set_prop(crack_display, "DiffuseColor", crack_color)
        set_prop(crack_display, "Opacity", crack_opacity)
        set_prop(crack_display, "LineWidth", crack_width)
        crack_path_applied = True

# Dark overlay for opened cohesive region.
sep_overlay_applied = False
if sep_overlay_enable and is_crack_3d:
    render_source_pre_crack.UpdatePipeline(float(times[0]))
    info_sep = render_source_pre_crack.GetDataInformation()
    assoc_sep0, sep_arr0 = _array_info(info_sep, sep_field_name)
    if sep_arr0 is not None:
        sep_overlay_thr = max(1e-9, float(os.environ.get("PV_COH_SEP_OVERLAY_THRESH", str(sep_thresh))))
        sep_overlay = Threshold(Input=render_source_pre_crack)
        set_prop(sep_overlay, "Scalars", [assoc_sep0, sep_field_name])
        set_prop(sep_overlay, "ThresholdMethod", "Above Lower Threshold")
        set_prop(sep_overlay, "LowerThreshold", sep_overlay_thr)
        set_prop(sep_overlay, "UpperThreshold", 1.0e30)
        set_prop(sep_overlay, "AllScalars", 0)
        sep_overlay_display = Show(sep_overlay, view)
        set_prop(sep_overlay_display, "Representation", "Surface")
        set_prop(sep_overlay_display, "ColorArrayName", [None, ""])
        set_prop(sep_overlay_display, "MapScalars", 0)
        set_prop(sep_overlay_display, "Lighting", 1)
        set_prop(sep_overlay_display, "AmbientColor", sep_overlay_color)
        set_prop(sep_overlay_display, "DiffuseColor", sep_overlay_color)
        set_prop(sep_overlay_display, "Opacity", sep_overlay_opacity)
        sep_overlay_applied = True

# Prefer vector displacement array 'u', fallback to scalar 'u_mag', then 'sigma_vm'.
render_source.UpdatePipeline(float(times[0]))
info0 = render_source.GetDataInformation()
arr_assoc = "POINTS"
arr_name = "u"
arr_is_vector = True

requested_field = os.environ.get("PV_SCALAR_FIELD", "").strip()
requested_vec_mode = os.environ.get("PV_VECTOR_MODE", "magnitude").strip().lower()
auto_vel_field_selected = False
if requested_field:
    assoc_req, arr_req = _array_info(info0, requested_field)
    if arr_req is None:
        raise RuntimeError(
            f"Requested PV_SCALAR_FIELD='{requested_field}' is not available in source point/cell data"
        )
    arr_assoc = assoc_req
    arr_name = requested_field
    try:
        n_comp_req = int(arr_req.GetNumberOfComponents())
    except Exception:
        n_comp_req = 1
    arr_is_vector = bool(n_comp_req > 1 and requested_vec_mode in {"magnitude", "mag", "norm", "auto"})

if (not requested_field) and is_vel_render:
    assoc_vmag, arr_vmag = _array_info(info0, "v_mag")
    if arr_vmag is not None:
        arr_assoc = assoc_vmag
        arr_name = "v_mag"
        arr_is_vector = False
        auto_vel_field_selected = True
    else:
        assoc_umag, arr_umag = _array_info(info0, "u_mag")
        if arr_umag is not None:
            arr_assoc = assoc_umag
            arr_name = "u_mag"
            arr_is_vector = False
            auto_vel_field_selected = True

prefer_damage_field = env_flag("PV_PREFER_DAMAGE_FIELD", False)
assoc, arr = (None, None)
if (not requested_field) and (not auto_vel_field_selected) and prefer_damage_field:
    assoc, arr = _array_info(info0, "aux_field")
    if arr is not None:
        arr_assoc = assoc
        arr_name = "aux_field"
        arr_is_vector = False
if (not requested_field) and (not auto_vel_field_selected) and arr_name != "aux_field":
    if arr is None:
        assoc, arr = _array_info(info0, "u")
    if arr is not None:
        try:
            arr_is_vector = int(arr.GetNumberOfComponents()) > 1
        except Exception:
            arr_is_vector = True
        arr_assoc = assoc
        arr_name = "u"
    else:
        assoc, arr = _array_info(info0, "u_mag")
        if arr is not None:
            arr_assoc = assoc
            arr_name = "u_mag"
            arr_is_vector = False
        else:
            assoc, arr = _array_info(info0, "sigma_vm")
            if arr is None:
                raise RuntimeError("No renderable array found (expected one of: u, u_mag, sigma_vm)")
            arr_assoc = assoc
            arr_name = "sigma_vm"
            arr_is_vector = False

if arr_is_vector:
    ColorBy(display, (arr_assoc, arr_name, "Magnitude"))
else:
    ColorBy(display, (arr_assoc, arr_name))
set_prop(display, "EnableOpacityMapping", 0)

lut_name = arr_name
lut = GetColorTransferFunction(lut_name)
try:
    pwf = GetOpacityTransferFunction(lut_name)
except Exception:
    pwf = None

if color_range_mode not in {"auto", "meta", "manual", "video"}:
    color_range_mode = "video"

applied_range_mode = color_range_mode
range_min = None
range_max = None

if color_range_mode == "manual":
    if manual_color_max > manual_color_min:
        range_min = float(manual_color_min)
        range_max = float(manual_color_max)
    else:
        applied_range_mode = "auto"
elif color_range_mode == "meta":
    load_ref = ((case_meta.get("loading") or {}).get("load_factor") if isinstance(case_meta.get("loading"), dict) else None)
    if isinstance(load_ref, (int, float)) and float(load_ref) > 0:
        range_min = 0.0
        range_max = max(1e-8, float(load_ref) * max(1e-6, color_max_uref_mult))
    else:
        applied_range_mode = "auto"
elif color_range_mode == "video":
    sampled = _sample_range(render_source, times, video_range_samples, video_range_quantile, arr_name, arr_is_vector)
    if sampled is None:
        applied_range_mode = "auto"
    else:
        sampled_min, sampled_qmax = sampled
        range_min = 0.0 if sampled_min >= 0.0 else sampled_min
        range_max = max(1e-8, sampled_qmax * video_range_pad)
        if range_max <= range_min + 1e-8:
            range_max = range_min + 1e-3
if arr_name == "aux_field":
    applied_range_mode = "manual"
    range_min = 0.0
    range_max = 1.0

if is_crack_3d:
    crack_surface_opacity = float(os.environ.get("PV_CRACK_SURFACE_OPACITY", str(surface_opacity)))
    crack_surface_opacity = min(1.0, max(0.05, crack_surface_opacity))
    set_prop(display, "Opacity", crack_surface_opacity)
    surface_opacity = crack_surface_opacity

if applied_range_mode == "auto":
    display.RescaleTransferFunctionToDataRange(True, False)
else:
    lut.RescaleTransferFunction(range_min, range_max)
    if pwf is not None:
        pwf.RescaleTransferFunction(range_min, range_max)
try:
    lut.AutomaticRescaleRangeMode = "Never"
except Exception:
    pass

show_colorbar = env_flag("PV_SHOW_COLORBAR", True)
colorbar_title = os.environ.get("PV_COLORBAR_TITLE", "|u|").strip() or "|u|"
colorbar_unit = os.environ.get("PV_COLORBAR_COMPONENT_TITLE", "m").strip()
colorbar_label_fmt = os.environ.get("PV_COLORBAR_LABEL_FORMAT", "%-#6.3g").strip() or "%-#6.3g"
if is_plate_hole_3d and arr_name == "sigma_vm" and "PV_COLORBAR_TITLE" not in os.environ:
    colorbar_title = "sigma_vm + |u| context"
    colorbar_unit = "Pa"
if is_center_notch_2d and arr_name == "aux_field" and "PV_COLORBAR_TITLE" not in os.environ:
    colorbar_title = "damage / crack evolution"
    colorbar_unit = ""
if is_crack_3d and arr_name == "aux_field" and "PV_COLORBAR_TITLE" not in os.environ:
    colorbar_title = "damage-like field"
    colorbar_unit = ""
if arr_name == "damage" and "PV_COLORBAR_TITLE" not in os.environ:
    colorbar_title = "damage"
    colorbar_unit = ""
if arr_name == "crack_mask" and "PV_COLORBAR_TITLE" not in os.environ:
    colorbar_title = "crack mask"
    colorbar_unit = ""
if arr_name == "max_principal_stress" and "PV_COLORBAR_TITLE" not in os.environ:
    colorbar_title = "max principal stress"
    colorbar_unit = "Pa"
if arr_name == "max_principal_strain" and "PV_COLORBAR_TITLE" not in os.environ:
    colorbar_title = "max principal strain"
    colorbar_unit = ""
if arr_name == "v_mag" and "PV_COLORBAR_TITLE" not in os.environ:
    colorbar_title = "|du/dtau|"
    colorbar_unit = ""
if is_vel_render and arr_name in {"u", "u_mag"} and "PV_COLORBAR_TITLE" not in os.environ:
    colorbar_title = "quasi-static |u|"
    colorbar_unit = "m"

if show_colorbar:
    display.SetScalarBarVisibility(view, True)
    scalar_bar = GetScalarBar(lut, view)
    if colorbar_unit:
        colorbar_title = f"{colorbar_title} ({colorbar_unit})"
    set_prop(scalar_bar, "Title", colorbar_title)
    set_prop(scalar_bar, "ComponentTitle", "")
    set_prop(scalar_bar, "LabelFormat", colorbar_label_fmt)
    set_prop(scalar_bar, "RangeLabelFormat", colorbar_label_fmt)
    set_prop(scalar_bar, "WindowLocation", "Any Location")
    set_prop(scalar_bar, "TitleFontSize", int(os.environ.get("PV_COLORBAR_TITLE_FONT_SIZE", "18")))
    set_prop(scalar_bar, "LabelFontSize", int(os.environ.get("PV_COLORBAR_LABEL_FONT_SIZE", "16")))
    set_prop(scalar_bar, "ScalarBarThickness", int(os.environ.get("PV_COLORBAR_THICKNESS", "18")))

    cb_pos_x = float(os.environ.get("PV_COLORBAR_POS_X", "0.83"))
    cb_pos_y = float(os.environ.get("PV_COLORBAR_POS_Y", "0.11"))
    cb_len = float(os.environ.get("PV_COLORBAR_LEN", "0.74"))
    cb_pos_x = min(0.95, max(0.01, cb_pos_x))
    cb_pos_y = min(0.95, max(0.01, cb_pos_y))
    cb_len = min(0.95 - cb_pos_y, max(0.05, cb_len))
    set_prop(scalar_bar, "Position", [cb_pos_x, cb_pos_y])
    set_prop(scalar_bar, "ScalarBarLength", cb_len)

show_feature_edges = env_flag("PV_SHOW_FEATURE_EDGES", False)
if show_feature_edges:
    feature_src = FeatureEdges(Input=ExtractSurface(Input=render_source))
    set_prop(feature_src, "BoundaryEdges", 1)
    set_prop(feature_src, "FeatureEdges", 1)
    set_prop(feature_src, "NonManifoldEdges", 0)
    set_prop(feature_src, "ManifoldEdges", 0)
    feature_display = Show(feature_src, view)
    set_prop(feature_display, "Representation", "Wireframe")
    set_prop(feature_display, "ColorArrayName", [None, ""])
    set_prop(feature_display, "MapScalars", 0)
    set_prop(feature_display, "Lighting", 0)
    set_prop(feature_display, "AmbientColor", [0.08, 0.08, 0.08])
    set_prop(feature_display, "DiffuseColor", [0.08, 0.08, 0.08])
    set_prop(feature_display, "LineWidth", float(os.environ.get("PV_FEATURE_EDGE_WIDTH", "1.7")))
    # Dashed style support depends on ParaView/VTK backend; these are best-effort.
    set_prop(feature_display, "LineStyle", "Dashed")
    set_prop(feature_display, "LineStyle", "Dash")
    set_prop(feature_display, "LineStipplePattern", int(os.environ.get("PV_FEATURE_EDGE_STIPPLE", "61680")))
    set_prop(feature_display, "LineStippleRepeatFactor", int(os.environ.get("PV_FEATURE_EDGE_REPEAT", "1")))

show_meta_overlay = env_flag("PV_SHOW_META_OVERLAY", False)
show_time_overlay = env_flag("PV_SHOW_TIME_OVERLAY", False)

overlay_lines = []
if show_meta_overlay:
    overlay_lines.append(f"case: {case_name_full}")
    if case_meta.get("family"):
        overlay_lines.append(f"family: {case_meta['family']}")
    overlay_lines.append(f"dim: {case_dim}")
    if shape_3d:
        overlay_lines.append(f"shape_3d: {shape_3d}")
    mat = case_meta.get("material") if isinstance(case_meta.get("material"), dict) else {}
    loading = case_meta.get("loading") if isinstance(case_meta.get("loading"), dict) else {}
    nonlinear = case_meta.get("nonlinear") if isinstance(case_meta.get("nonlinear"), dict) else {}
    if mat.get("model"):
        overlay_lines.append(f"model: {mat.get('model')}")
    if nonlinear.get("model"):
        overlay_lines.append(f"nonlinear: {nonlinear.get('model')}")
    if loading.get("mode"):
        overlay_lines.append(f"load mode: {loading.get('mode')}")
    if loading.get("load_factor") is not None:
        overlay_lines.append(f"load factor: {fmt_num(loading.get('load_factor'))}")

    if applied_range_mode == "auto":
        overlay_lines.append(f"field: {arr_name}, range: auto")
    else:
        overlay_lines.append(f"field: {arr_name}, range: [{fmt_num(range_min)}, {fmt_num(range_max)}] {colorbar_unit}")

if overlay_lines:
    meta_text = Text()
    meta_text.Text = "\n".join(overlay_lines)
    meta_display = Show(meta_text, view)
    set_prop(meta_display, "WindowLocation", os.environ.get("PV_META_WINDOW_LOCATION", "UpperLeftCorner"))
    set_prop(meta_display, "FontSize", int(os.environ.get("PV_META_FONT_SIZE", "16")))
    set_prop(meta_display, "Color", [1.0, 1.0, 1.0])
    set_prop(meta_display, "Shadow", 1)

if show_time_overlay:
    time_anno = AnnotateTimeFilter(Input=reader)
    time_anno.Format = os.environ.get("PV_TIME_FORMAT", "t = %0.4g")
    time_display = Show(time_anno, view)
    set_prop(time_display, "WindowLocation", os.environ.get("PV_TIME_WINDOW_LOCATION", "LowerLeftCorner"))
    set_prop(time_display, "FontSize", int(os.environ.get("PV_TIME_FONT_SIZE", "15")))
    set_prop(time_display, "Color", [1.0, 1.0, 1.0])
    set_prop(time_display, "Shadow", 1)

camera_bounds_samples = max(3, int(os.environ.get("PV_CAMERA_BOUNDS_SAMPLES", str(video_range_samples))))
default_camera_pad = "1.24" if is_crack_3d else "1.12"
camera_bounds_pad = max(1.0, float(os.environ.get("PV_CAMERA_BOUNDS_PAD", default_camera_pad)))
# Keep camera on full global model bounds from physical deformed geometry only.
camera_source = render_source_pre_crack
b_sampled = _sample_bounds(camera_source, times, camera_bounds_samples)
if b_sampled is None:
    b = camera_source.GetDataInformation().GetBounds()
    xmin, xmax, ymin, ymax, zmin, zmax = [float(v) for v in b]
else:
    mins, maxs = b_sampled
    xmin, ymin, zmin = mins
    xmax, ymax, zmax = maxs
cx = 0.5 * (xmin + xmax)
cy = 0.5 * (ymin + ymax)
cz = 0.5 * (zmin + zmax)
sx = max(1e-6, (xmax - xmin) * camera_bounds_pad)
sy = max(1e-6, (ymax - ymin) * camera_bounds_pad)
sz = max(1e-6, (zmax - zmin) * camera_bounds_pad)

camera_mode = os.environ.get("PV_CAMERA_MODE", "auto").strip().lower()
z_ratio_thresh = float(os.environ.get("PV_3D_RATIO_THRESHOLD", "0.12"))
z_ratio = sz / max(sx, sy, 1e-6)
auto_is_3d = (case_dim == "3D") or ("3d" in case_name) or (z_ratio >= z_ratio_thresh)
if camera_mode == "3d":
    use_3d_camera = True
elif camera_mode == "2d":
    use_3d_camera = False
else:
    use_3d_camera = auto_is_3d

if use_3d_camera:
    ortho_camera = env_flag("PV_ORTHO_CAMERA", True)
    preset_by_shape = {
        "cube": (2.0, 38.0, 34.0, 34.0),
        "cuboid": (2.3, 42.0, 30.0, 30.0),
        "cylinder": (2.2, 50.0, 24.0, 30.0),
        "sphere": (1.9, 42.0, 42.0, 38.0),
        "plate_with_hole_3d": (2.25, 30.0, 34.0, 28.0),
        "notched_bar_3d": (2.35, 36.0, 22.0, 30.0),
    }
    p_dist, p_yaw, p_pitch, p_fov = preset_by_shape.get(shape_3d, (2.1, 42.0, 42.0, 32.0))
    if is_crack_3d:
        p_dist, p_yaw, p_pitch, p_fov = (2.25, 30.0, 34.0, 30.0)

    dist_factor = float(os.environ.get("PV_3D_DISTANCE_FACTOR", str(p_dist)))
    yaw_deg = float(os.environ.get("PV_3D_YAW_DEG", str(p_yaw)))
    pitch_deg = float(os.environ.get("PV_3D_PITCH_DEG", str(p_pitch)))
    fov_deg = float(os.environ.get("PV_3D_FOV_DEG", str(p_fov)))
    fit_margin = float(os.environ.get("PV_3D_FIT_MARGIN", "1.18"))
    if show_colorbar:
        fit_margin *= max(1.0, float(os.environ.get("PV_COLORBAR_SAFE_FIT_MULT", "1.30")))
    base_span = max(sx, sy, sz)
    user_distance = dist_factor * base_span

    half_diag = 0.5 * math.sqrt(sx * sx + sy * sy + sz * sz)
    half_fov = math.radians(max(5.0, min(120.0, fov_deg)) * 0.5)
    fit_distance = (fit_margin * half_diag) / max(math.sin(half_fov), 1e-6)
    radius = max(user_distance, fit_distance)

    yaw = math.radians(yaw_deg)
    pitch = math.radians(pitch_deg)
    px = cx + radius * math.cos(pitch) * math.cos(yaw)
    py = cy + radius * math.cos(pitch) * math.sin(yaw)
    pz = cz + radius * math.sin(pitch)

    view.CameraPosition = [px, py, pz]
    view.CameraFocalPoint = [cx, cy, cz]
    view.CameraViewUp = [0.0, 0.0, 1.0]
    view.CameraParallelProjection = 1 if ortho_camera else 0
    if ortho_camera:
        view.CameraParallelScale = max(1e-6, fit_margin * half_diag)
    view.CameraViewAngle = fov_deg
else:
    ortho_camera = True
    view.CameraPosition = [cx, cy, cz + 3.0 * max(sx, sy, sz)]
    view.CameraFocalPoint = [cx, cy, cz]
    view.CameraViewUp = [0.0, 1.0, 0.0]
    view.CameraParallelProjection = 1
    safe_fit_mult = max(1.0, float(os.environ.get("PV_COLORBAR_SAFE_FIT_MULT", "1.30"))) if show_colorbar else 1.0
    view.CameraParallelScale = 0.6 * max(sx, sy) * safe_fit_mult
try:
    view.ResetCamera(xmin, xmax, ymin, ymax, zmin, zmax)
except Exception:
    try:
        view.ResetCamera()
    except Exception:
        pass
if use_3d_camera:
    view.CameraPosition = [px, py, pz]
    view.CameraFocalPoint = [cx, cy, cz]
    view.CameraViewUp = [0.0, 0.0, 1.0]
    view.CameraParallelProjection = 1 if ortho_camera else 0
    if ortho_camera:
        view.CameraParallelScale = max(1e-6, fit_margin * half_diag)
    view.CameraViewAngle = fov_deg
else:
    view.CameraPosition = [cx, cy, cz + 3.0 * max(sx, sy, sz)]
    view.CameraFocalPoint = [cx, cy, cz]
    view.CameraViewUp = [0.0, 1.0, 0.0]
    view.CameraParallelProjection = 1
    safe_fit_mult = max(1.0, float(os.environ.get("PV_COLORBAR_SAFE_FIT_MULT", "1.30"))) if show_colorbar else 1.0
    view.CameraParallelScale = 0.6 * max(sx, sy) * safe_fit_mult
try:
    view.ResetCameraClippingRange()
except Exception:
    pass
Render(view)

png_glob = f"{out_root}_*.png"
for old in glob(png_glob):
    try:
        os.remove(old)
    except OSError:
        pass

t0 = float(times[0])
t1 = float(times[-1])
n_times = len(times)
span = max(0.0, t1 - t0)

if duration_mode == "fixed":
    duration_sec = fixed_duration_sec
else:
    duration_by_steps = n_times / max(steps_per_video_second, 1e-6)
    duration_by_span = span / max(sim_seconds_per_video_second, 1e-6)
    duration_sec = max(duration_by_steps, duration_by_span)
    if n_times <= 1:
        duration_sec = min_duration_sec

duration_sec = max(min_duration_sec, min(max_duration_sec, duration_sec))
base_frames = max(1, int(round(fps * duration_sec)))
hold_frames = max(0, int(round(max(0.0, hold_last_sec) * fps)))

if len(times) == 1:
    sample_times = [float(times[0])] * base_frames
else:
    sample_times = [t0 + (t1 - t0) * i / (base_frames - 1) for i in range(base_frames)]

if hold_frames > 0:
    sample_times.extend([float(t1)] * hold_frames)

target_frames = len(sample_times)

for i, t in enumerate(sample_times):
    if crack_filtered_proxy is not None and crack_progressive_cut and crack_cut_uses_damage:
        if t1 > t0:
            tau = min(1.0, max(0.0, (float(t) - t0) / max(1e-12, (t1 - t0))))
        else:
            tau = 1.0
        tau_s = tau * tau * (3.0 - 2.0 * tau)  # smoothstep
        cut_now = 1.0 - (1.0 - crack_damage_cut) * tau_s
        set_prop(crack_filtered_proxy, "UpperThreshold", cut_now)
    scene.AnimationTime = float(t)
    view.ViewTime = float(t)
    Render(view)
    SaveScreenshot(f"{out_root}_{i:04d}.png", view, ImageResolution=[width, height])

if has_png_frames(png_glob):
    if applied_range_mode == "auto":
        range_desc = "auto"
    else:
        range_desc = f"[{fmt_num(range_min)}, {fmt_num(range_max)}]"

    print(
        "[render_paraview] wrote "
        f"{target_frames} frames ({duration_sec:.1f}s + hold {hold_frames / max(fps, 1):.1f}s @ {fps}fps, {width}x{height}), "
        f"mode={duration_mode}, n_times={n_times}, span={span:.3g}, "
        f"camera={'3d' if use_3d_camera else '2d'} ortho={int(ortho_camera)} (z_ratio={z_ratio:.4f}), "
        f"array={arr_name}, range_mode={applied_range_mode}, range={range_desc}, colorbar={int(show_colorbar)}, "
        f"bg=#{''.join(f'{int(round(c * 255)):02X}' for c in bg_rgb)}, deform_scale={deform_scale:.4g}, "
        f"opacity={surface_opacity:.3g}, "
        f"crack_hole={int(crack_hole_enable)} applied={int(crack_hole_applied)} field={crack_cut_field} cut={crack_damage_cut:.3g} sep_cut={sep_thresh:.3g} progressive={int(crack_progressive_cut)}, "
        f"split_opening={int(split_opening_enable)} applied={int(split_opening_applied)} field={split_opening_field or '-'} scale={split_opening_scale:.3g} plane_x={fmt_num(split_opening_plane_x)}, "
        f"coh_overlay={int(sep_overlay_enable)} applied={int(sep_overlay_applied)}, "
        f"crack_path_overlay={int(crack_path_overlay_enable)} applied={int(crack_path_applied)}, "
        f"curved_shape_proxy={int(curved_shape_proxy_enable)} applied={int(curved_shape_proxy_applied)} "
        f"subdiv={curved_shape_proxy_subdivide_level}, "
        f"curved_shape_smooth={int(curved_shape_enable)} applied={int(curved_shape_smooth_applied)} "
        f"subdiv={curved_shape_subdivide_level} applied={int(curved_shape_subdivide_applied)} "
        f"iters={curved_shape_iters}, "
        f"crack_subdiv={crack_subdivide_level} applied={int(crack_subdivide_applied)}, "
        f"crack_smooth={int(crack_smooth_enable)} applied={int(crack_smooth_applied)}, "
        f"feature_edges={int(show_feature_edges)}, "
        f"meta_overlay={int(show_meta_overlay)}, time_overlay={int(show_time_overlay)}: {png_glob}"
    )
else:
    raise RuntimeError(f"failed to create PNG sequence for {case_dir}")
