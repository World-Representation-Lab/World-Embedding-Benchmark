import json
import math
import os
import sys
from glob import glob

from paraview.simple import *
import paraview

# Force OSMesa for offscreen rendering
paraview.simple._DisableFirstRenderCameraReset()

# Run with ParaView's Python:
#   pvpython scripts/render_paraview.py <case_dir> <out_mp4>
# Or (offscreen):
#   pvbatch scripts/render_paraview.py <case_dir> <out_mp4>


case_dir = sys.argv[1]
out_mp4 = sys.argv[2]
out_root, _ = os.path.splitext(out_mp4)
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


# Visualization reliability controls (for retrieval/identification)
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


def set_prop(proxy, key: str, value) -> None:
    try:
        setattr(proxy, key, value)
    except Exception:
        pass


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


def _sample_field_range(
    proxy,
    times: list[float],
    n_samples: int,
    q: float,
    *,
    field_name: str,
    vector_magnitude: bool,
) -> tuple[float, float] | None:
    sample_times = _pick_evenly_spaced(times, n_samples)
    mins = []
    maxs = []

    for t in sample_times:
        proxy.UpdatePipeline(float(t))
        info = proxy.GetDataInformation()
        arr_info = None
        try:
            p_info = info.GetPointDataInformation()
            if p_info is not None:
                arr_info = p_info.GetArrayInformation(field_name)
        except Exception:
            arr_info = None

        if arr_info is None:
            try:
                c_info = info.GetCellDataInformation()
                if c_info is not None:
                    arr_info = c_info.GetArrayInformation(field_name)
            except Exception:
                arr_info = None

        if arr_info is None:
            continue

        try:
            lo, hi = arr_info.GetComponentRange(-1 if vector_magnitude else 0)
        except Exception:
            lo, hi = arr_info.GetComponentRange(0)

        mins.append(float(lo))
        maxs.append(float(hi))

    if not maxs:
        return None

    return min(mins), _quantile(maxs, q)


foam_marker = os.path.join(case_dir, "case.foam")
if not os.path.exists(foam_marker):
    open(foam_marker, "w").close()

case_meta = load_case_meta(case_dir)
case_name_full = os.path.basename(case_dir)
case_name = case_name_full.lower()
solver_name = str(case_meta.get("solver", "") or "").strip()
scene_name = str(case_meta.get("scene", "") or "").strip().lower()
is_dye_case = solver_name == "scalarTransportFoam" or "dye" in scene_name or case_name.startswith("dye")

reader = OpenFOAMReader(FileName=foam_marker)
reader.MeshRegions = ["internalMesh"]
reader.CellArrays = ["U", "p", "T"]
reader.PointArrays = ["U", "p", "T"]

# Smooth visual appearance by interpolating cell-centered data to points.
c2p = CellDatatoPointData(Input=reader)

view = GetActiveViewOrCreate("RenderView")
view.ViewSize = [width, height]
display = Show(c2p, view)
view.Update()
Render(view)

scene = GetAnimationScene()
scene.UpdateAnimationUsingDataTimeSteps()
time_keeper = GetTimeKeeper()
times = list(getattr(time_keeper, "TimestepValues", []) or getattr(reader, "TimestepValues", []) or [])
if not times:
    times = [0.0]
skip_initial_time = env_flag("PV_SKIP_INITIAL_TIME", True)
if skip_initial_time and len(times) > 1 and abs(float(times[0])) <= 1e-12:
    # The OpenFOAM t=0 field is often an initialization state, not a meaningful
    # rendered simulation frame.  Including it causes a white/blank flash for
    # some RANS cases and weakens retrieval videos.
    times = [float(t) for t in times[1:]]

field_mode = os.environ.get("PV_FIELD", "auto").strip().lower()
if field_mode not in {"auto", "u", "t"}:
    field_mode = "auto"

if field_mode == "t" or (field_mode == "auto" and is_dye_case):
    active_field = "T"
    vector_magnitude = False
    ColorBy(display, ("POINTS", "T"))
    colorbar_title_default = "T"
    colorbar_unit_default = ""
else:
    active_field = "U"
    vector_magnitude = True
    ColorBy(display, ("POINTS", "U", "Magnitude"))
    colorbar_title_default = "|U|"
    colorbar_unit_default = "m/s"

u_lut = GetColorTransferFunction(active_field)
try:
    u_pwf = GetOpacityTransferFunction(active_field)
except Exception:
    u_pwf = None

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
    if active_field == "T":
        range_min = 0.0
        range_max = 1.0
    else:
        u_ref = case_meta.get("U_ref")
        if isinstance(u_ref, (int, float)) and float(u_ref) > 0:
            range_min = 0.0
            range_max = max(1e-8, float(u_ref) * max(1e-6, color_max_uref_mult))
        else:
            applied_range_mode = "auto"
elif color_range_mode == "video":
    sampled = _sample_field_range(
        c2p,
        times,
        video_range_samples,
        video_range_quantile,
        field_name=active_field,
        vector_magnitude=vector_magnitude,
    )
    if sampled is None:
        applied_range_mode = "auto"
    else:
        sampled_min, sampled_qmax = sampled
        range_min = 0.0 if sampled_min >= 0.0 else sampled_min
        range_max = max(1e-8, sampled_qmax * video_range_pad)
        if range_max <= range_min + 1e-8:
            range_max = range_min + 1e-3

if applied_range_mode == "auto":
    display.RescaleTransferFunctionToDataRange(True, False)
else:
    u_lut.RescaleTransferFunction(range_min, range_max)
    if u_pwf is not None:
        u_pwf.RescaleTransferFunction(range_min, range_max)

show_colorbar = env_flag("PV_SHOW_COLORBAR", True)
colorbar_title = os.environ.get("PV_COLORBAR_TITLE", colorbar_title_default).strip() or colorbar_title_default
colorbar_unit = os.environ.get("PV_COLORBAR_COMPONENT_TITLE", colorbar_unit_default).strip()
if active_field == "T":
    if colorbar_title in {"|U|", "|U| (m/s)"}:
        colorbar_title = colorbar_title_default
    if colorbar_unit == "m/s":
        colorbar_unit = colorbar_unit_default
colorbar_label_fmt = os.environ.get("PV_COLORBAR_LABEL_FORMAT", "%-#6.3g").strip() or "%-#6.3g"

if show_colorbar:
    display.SetScalarBarVisibility(view, True)
    scalar_bar = GetScalarBar(u_lut, view)
    # Keep unit in title to avoid component-title clipping on narrow right margins.
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

    # Safer defaults so title/labels stay in-frame at 1280x720 and 1920x1080.
    cb_pos_x = float(os.environ.get("PV_COLORBAR_POS_X", "0.83"))
    cb_pos_y = float(os.environ.get("PV_COLORBAR_POS_Y", "0.11"))
    cb_len = float(os.environ.get("PV_COLORBAR_LEN", "0.74"))
    set_prop(scalar_bar, "Position", [cb_pos_x, cb_pos_y])
    set_prop(scalar_bar, "ScalarBarLength", cb_len)

show_meta_overlay = env_flag("PV_SHOW_META_OVERLAY", False)
show_time_overlay = env_flag("PV_SHOW_TIME_OVERLAY", False)

overlay_lines = []
if show_meta_overlay:
    overlay_lines.append(f"case: {case_name_full}")
    if case_meta.get("scene"):
        overlay_lines.append(f"scene: {case_meta['scene']}")
    if case_meta.get("Re") is not None:
        overlay_lines.append(f"Re: {fmt_num(case_meta.get('Re'))}")
    if case_meta.get("nu") is not None:
        overlay_lines.append(f"nu: {fmt_num(case_meta.get('nu'))} m^2/s")

    if solver_name:
        overlay_lines.append(f"solver: {solver_name}")

    if applied_range_mode == "auto":
        overlay_lines.append(f"field: {active_field}, range: auto")
    else:
        overlay_lines.append(f"field: {active_field}, range: [{fmt_num(range_min)}, {fmt_num(range_max)}] {colorbar_unit}")

if overlay_lines:
    meta_text = Text()
    meta_text.Text = "\n".join(overlay_lines)
    meta_display = Show(meta_text, view)
    set_prop(meta_display, "WindowLocation", os.environ.get("PV_META_WINDOW_LOCATION", "UpperLeftCorner"))
    set_prop(meta_display, "FontSize", int(os.environ.get("PV_META_FONT_SIZE", "16")))
    set_prop(meta_display, "Color", [1.0, 1.0, 1.0])
    set_prop(meta_display, "Shadow", 1)

if show_time_overlay:
    time_anno = AnnotateTimeFilter(Input=c2p)
    time_anno.Format = os.environ.get("PV_TIME_FORMAT", "t = %0.4g s")
    time_display = Show(time_anno, view)
    set_prop(time_display, "WindowLocation", os.environ.get("PV_TIME_WINDOW_LOCATION", "LowerLeftCorner"))
    set_prop(time_display, "FontSize", int(os.environ.get("PV_TIME_FONT_SIZE", "15")))
    set_prop(time_display, "Color", [1.0, 1.0, 1.0])
    set_prop(time_display, "Shadow", 1)

# Camera setup:
# - 2D cases: orthographic top-down
# - 3D cases: perspective oblique view
b = c2p.GetDataInformation().GetBounds()
xmin, xmax, ymin, ymax, zmin, zmax = b
cx = 0.5 * (xmin + xmax)
cy = 0.5 * (ymin + ymax)
cz = 0.5 * (zmin + zmax)
sx = max(1e-6, xmax - xmin)
sy = max(1e-6, ymax - ymin)
sz = max(1e-6, zmax - zmin)

camera_mode = os.environ.get("PV_CAMERA_MODE", "auto").strip().lower()
z_ratio_thresh = float(os.environ.get("PV_3D_RATIO_THRESHOLD", "0.12"))
z_ratio = sz / max(sx, sy, 1e-6)
auto_is_3d = ("3d" in case_name) or (z_ratio >= z_ratio_thresh)
if camera_mode == "3d":
    use_3d_camera = True
elif camera_mode == "2d":
    use_3d_camera = False
else:
    use_3d_camera = auto_is_3d

if use_3d_camera:
    dist_factor = float(os.environ.get("PV_3D_DISTANCE_FACTOR", "2.1"))
    yaw_deg = float(os.environ.get("PV_3D_YAW_DEG", "42.0"))
    pitch_deg = float(os.environ.get("PV_3D_PITCH_DEG", "42.0"))
    fov_deg = float(os.environ.get("PV_3D_FOV_DEG", "32.0"))
    fit_margin = float(os.environ.get("PV_3D_FIT_MARGIN", "1.18"))
    base_span = max(sx, sy, sz)
    user_distance = dist_factor * base_span

    # Keep the full domain in frame by fitting the bounding sphere to the FOV.
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
    view.CameraParallelProjection = 0
    view.CameraViewAngle = fov_deg
else:
    view.CameraPosition = [cx, cy, cz + 3.0 * max(sx, sy, sz)]
    view.CameraFocalPoint = [cx, cy, cz]
    view.CameraViewUp = [0.0, 1.0, 0.0]
    view.CameraParallelProjection = 1
    view.CameraParallelScale = 0.6 * max(sx, sy)
Render(view)

# Render PNG sequence with fixed or auto duration; outer pipeline converts to mp4.
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
    # Auto duration keeps temporal detail comparable across cases:
    # 1) enough frames per available time step
    # 2) enough screen time for long physical span
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
        f"camera={'3d' if use_3d_camera else '2d'} (z_ratio={z_ratio:.4f}), "
        f"range_mode={applied_range_mode}, range={range_desc}, colorbar={int(show_colorbar)}, "
        f"meta_overlay={int(show_meta_overlay)}, time_overlay={int(show_time_overlay)}, "
        f"skip_initial_time={int(skip_initial_time)}: {png_glob}"
    )
else:
    raise RuntimeError(f"failed to create PNG sequence for {case_dir}")
