import json
import math
import os
import re
import shutil
import sys
from pathlib import Path


def replace_in_file(path: Path, replacements: dict):
    txt = path.read_text()
    for k, v in replacements.items():
        txt = txt.replace(k, str(v))
    path.write_text(txt)


def set_foam_dict_entries(path: Path, entries: dict[str, object]):
    txt = path.read_text()
    for key, value in entries.items():
        pattern = re.compile(rf"(?m)^([ \t]*{re.escape(key)}[ \t]+)[^;]*;")
        replacement = lambda m, value=value: f"{m.group(1)}{value};"
        txt, n = pattern.subn(replacement, txt, count=1)
        if n == 0:
            if not txt.endswith("\n"):
                txt += "\n"
            txt += f"{key}          {value};\n"
    path.write_text(txt)


def ensure_quality_function_objects(case_dir: Path):
    control = case_dir / "system" / "controlDict"
    if not control.exists():
        return

    txt = control.read_text()
    if "fieldMinMax" in txt:
        return

    fields = []
    for name in ("U", "p", "T"):
        if (case_dir / "0" / name).exists():
            fields.append(name)
    if not fields:
        return

    field_list = " ".join(fields)
    txt = txt.rstrip() + f"""

functions
{{
    fieldMinMax
    {{
        type            fieldMinMax;
        libs            ("libfieldFunctionObjects.so");
        enabled         true;
        executeControl  timeStep;
        executeInterval 10;
        writeControl    timeStep;
        writeInterval   10;
        fields          ({field_list});
        location        yes;
    }}
}}
"""
    control.write_text(txt)


def apply_common_replacements(case_dir: Path, *, u_ref: float, nu: float, dt: float, end_time: float, write_steps: int, mesh: dict, geometry: dict):
    replace_in_file(case_dir / "0" / "U", {
        "__U_VALUE__": f"{u_ref}",
        "__PROFILE_UMAX__": f"{u_ref:.8g}",
        "value uniform (1 0 0);": f"value uniform ({u_ref} 0 0);",
    })

    replace_in_file(case_dir / "constant" / "transportProperties", {
        "__NU__": f"{nu:.8e}",
        "1e-3": f"{nu:.8e}",
    })

    replace_in_file(case_dir / "system" / "controlDict", {
        "__END_TIME__": f"{end_time}",
        "__DELTA_T__": f"{dt}",
        "__WRITE_INTERVAL_STEPS__": f"{write_steps}",
        "endTime         10;": f"endTime         {end_time};",
        "deltaT          0.01;": f"deltaT          {dt};",
        "writeInterval   10;": f"writeInterval   {write_steps};",
    })
    set_foam_dict_entries(
        case_dir / "system" / "controlDict",
        {
            "endTime": end_time,
            "deltaT": dt,
            "adjustTimeStep": "yes",
            "maxCo": 0.5,
            "maxDeltaT": dt,
            "writeInterval": write_steps,
        },
    )

    nx = int(mesh.get("nx", 128))
    ny = int(mesh.get("ny", 128))
    nz = int(mesh.get("nz", 1))
    ny_obstacle = int(mesh.get("ny_obstacle", max(4, ny // 8)))
    ny_jet = int(mesh.get("ny_jet", max(4, ny // 4)))
    ny_outer = int(mesh.get("ny_outer", max(4, ny - ny_jet)))
    lx = float(geometry.get("lx", 1.0))
    ly = float(geometry.get("ly", 1.0))
    lz = float(geometry.get("lz", mesh.get("thickness", 0.01)))
    radius = float(geometry.get("radius", 0.5 * ly))
    obstacle_radius = float(geometry.get("obstacle_radius", radius))
    obstacle_half_width = float(geometry.get("obstacle_half_width", 0.5))
    chord = float(geometry.get("chord", 1.2))
    airfoil_thickness_scale = float(geometry.get("airfoil_thickness_scale", 1.0))
    r_jet = float(geometry.get("r_jet", radius))
    r_outer = float(geometry.get("r_outer", max(radius, ly)))
    wedge_half_angle_deg = float(geometry.get("wedge_half_angle_deg", 2.5))
    wedge_half_angle_rad = math.radians(wedge_half_angle_deg)
    r_cos = radius * math.cos(wedge_half_angle_rad)
    r_sin = radius * math.sin(wedge_half_angle_rad)
    r_jet_cos = r_jet * math.cos(wedge_half_angle_rad)
    r_jet_sin = r_jet * math.sin(wedge_half_angle_rad)
    r_outer_cos = r_outer * math.cos(wedge_half_angle_rad)
    r_outer_sin = r_outer * math.sin(wedge_half_angle_rad)
    r_outer_diag = r_outer / math.sqrt(2.0)
    x_min = float(geometry.get("x_min", -0.25 * lx))
    x_max = float(geometry.get("x_max", 0.75 * lx))
    y_min = float(geometry.get("y_min", -0.5 * ly))
    y_max = float(geometry.get("y_max", 0.5 * ly))
    cyl_diag = obstacle_radius / math.sqrt(2.0)
    square_left = -obstacle_half_width
    square_right = obstacle_half_width
    square_lower = -obstacle_half_width
    square_upper = obstacle_half_width
    air_half_chord = 0.5 * chord
    air_box_half = max(0.45, 0.42 * chord)
    air_le = -air_half_chord
    air_te = air_half_chord
    air_le_t = 0.06 * airfoil_thickness_scale
    air_te_t = 0.01 * airfoil_thickness_scale
    # Use a deliberately rounded airfoil section for benchmark robustness.  A
    # true thin NACA trailing edge produced invalid blockMesh cells for many
    # scaled variants.
    air_thickness = 0.40 * chord * airfoil_thickness_scale
    air_x_quarter = 0.25 * chord
    air_x_eighth = 0.125 * chord
    air_y_quarter = 0.78 * air_thickness
    air_y_mid = 0.96 * air_thickness
    air_y_te = 0.22 * air_thickness
    air_spline_x1 = -0.25 * chord
    air_spline_x2 = 0.0
    air_spline_x3 = 0.25 * chord
    air_spline_y1 = 0.09 * airfoil_thickness_scale
    air_spline_y2 = 0.07 * airfoil_thickness_scale
    air_spline_y3 = 0.035 * airfoil_thickness_scale
    air_le_arc_x = air_le - 0.0667 * chord
    air_te_arc_x = air_te + 0.0167 * chord

    block_mesh = case_dir / "system" / "blockMeshDict"
    block_mesh_text = block_mesh.read_text()
    replacements = {
        "convertToMeters": "scale",
        "__NX__": f"{nx}",
        "__NY__": f"{ny}",
        "__NY_OBSTACLE__": f"{ny_obstacle}",
        "__NY_JET__": f"{ny_jet}",
        "__NY_OUTER__": f"{ny_outer}",
        "__NZ__": f"{nz}",
        "__LX__": f"{lx}",
        "__LY__": f"{ly}",
        "__LZ__": f"{lz}",
        "__R__": f"{radius}",
        "__OBSTACLE_RADIUS__": f"{obstacle_radius}",
        "__OBSTACLE_RADIUS_DIAG__": f"{cyl_diag:.10f}",
        "__OBSTACLE_HALF_WIDTH__": f"{obstacle_half_width}",
        "__R_COS__": f"{r_cos:.10f}",
        "__R_SIN__": f"{r_sin:.10f}",
        "__R_JET__": f"{r_jet}",
        "__R_JET_COS__": f"{r_jet_cos:.10f}",
        "__R_JET_SIN__": f"{r_jet_sin:.10f}",
        "__R_OUTER__": f"{r_outer}",
        "__R_OUTER_COS__": f"{r_outer_cos:.10f}",
        "__R_OUTER_SIN__": f"{r_outer_sin:.10f}",
        "__R_OUTER_DIAG__": f"{r_outer_diag:.10f}",
        "__XMIN__": f"{x_min}",
        "__XMAX__": f"{x_max}",
        "__YMIN__": f"{y_min}",
        "__YMAX__": f"{y_max}",
        "__SQUARE_LEFT__": f"{square_left}",
        "__SQUARE_RIGHT__": f"{square_right}",
        "__SQUARE_LOWER__": f"{square_lower}",
        "__SQUARE_UPPER__": f"{square_upper}",
        "__AIR_BOX_LEFT__": f"{-air_box_half}",
        "__AIR_BOX_RIGHT__": f"{air_box_half}",
        "__AIR_BOX_LOWER__": f"{-air_box_half}",
        "__AIR_BOX_UPPER__": f"{air_box_half}",
        "__AIR_LE__": f"{air_le}",
        "__AIR_TE__": f"{air_te}",
        "__AIR_LE_T__": f"{air_le_t}",
        "__AIR_TE_T__": f"{air_te_t}",
        "__AIR_THICKNESS__": f"{air_thickness:.10f}",
        "__AIR_X_QUARTER__": f"{air_x_quarter:.10f}",
        "__AIR_X_EIGHTH__": f"{air_x_eighth:.10f}",
        "__AIR_Y_QUARTER__": f"{air_y_quarter:.10f}",
        "__AIR_Y_MID__": f"{air_y_mid:.10f}",
        "__AIR_Y_TE__": f"{air_y_te:.10f}",
        "__AIR_LE_ARC_X__": f"{air_le_arc_x}",
        "__AIR_TE_ARC_X__": f"{air_te_arc_x}",
        "__AIR_SPLINE_X1__": f"{air_spline_x1}",
        "__AIR_SPLINE_X2__": f"{air_spline_x2}",
        "__AIR_SPLINE_X3__": f"{air_spline_x3}",
        "__AIR_SPLINE_Y1__": f"{air_spline_y1}",
        "__AIR_SPLINE_Y2__": f"{air_spline_y2}",
        "__AIR_SPLINE_Y3__": f"{air_spline_y3}",
    }

    # Backward compatibility for legacy unit-cube templates that don't use placeholders.
    if "__LX__" not in block_mesh_text and "__LY__" not in block_mesh_text and "__LZ__" not in block_mesh_text:
        replacements.update({
            "(128 128 1)": f"({nx} {ny} {nz})",
            "(0 0 0.01)": f"(0 0 {lz})",
            "(1 0 0.01)": f"({lx} 0 {lz})",
            "(1 1 0.01)": f"({lx} {ly} {lz})",
            "(0 1 0.01)": f"(0 {ly} {lz})",
            "(1 0 0)": f"({lx} 0 0)",
            "(1 1 0)": f"({lx} {ly} 0)",
            "(0 1 0)": f"(0 {ly} 0)",
        })

    replace_in_file(block_mesh, replacements)

    u_field = case_dir / "0" / "U"
    if u_field.exists():
        replace_in_file(
            u_field,
            {
                "__PROFILE_HEIGHT__": f"{ly:.8g}",
            },
        )


def apply_solver_controls(case_dir: Path, *, cfg: dict):
    fv_solution = case_dir / "system" / "fvSolution"
    if not fv_solution.exists():
        return
    text = fv_solution.read_text()

    def patch_entry(block_name: str, key: str, value: str):
        nonlocal text
        pattern = re.compile(
            rf"(?ms)(^\s*{re.escape(block_name)}\s*\{{)(.*?)(^\s*\}})",
            flags=re.MULTILINE,
        )
        match = pattern.search(text)
        if not match:
            return
        head, body, tail = match.group(1), match.group(2), match.group(3)
        entry_pattern = re.compile(rf"(?m)^(\s*{re.escape(key)}\s+)[^;]+;")
        if entry_pattern.search(body):
            body = entry_pattern.sub(rf"\1{value};", body, count=1)
        else:
            body = body.rstrip() + f"\n        {key} {value};\n"
        text = text[: match.start()] + f"{head}{body}{tail}" + text[match.end() :]

    piso_pattern = re.compile(r"(?ms)(^\s*PISO\s*\{)(.*?)(^\s*\})", flags=re.MULTILINE)
    piso_match = piso_pattern.search(text)
    if piso_match:
        head, body, tail = piso_match.group(1), piso_match.group(2), piso_match.group(3)
        if cfg.get("nCorrectors") is not None:
            body = re.sub(r"(?m)^(\s*nCorrectors\s+)[^;]+;", rf"\g<1>{int(cfg['nCorrectors'])};", body, count=1)
        if cfg.get("nNonOrthogonalCorrectors") is not None:
            body = re.sub(
                r"(?m)^(\s*nNonOrthogonalCorrectors\s+)[^;]+;",
                rf"\g<1>{int(cfg['nNonOrthogonalCorrectors'])};",
                body,
                count=1,
            )
        text = text[: piso_match.start()] + f"{head}{body}{tail}" + text[piso_match.end() :]

    block_specs = [
        ("p", "solver", cfg.get("p_solver")),
        ("p", "preconditioner", cfg.get("p_preconditioner")),
        ("p", "tolerance", cfg.get("p_tolerance")),
        ("p", "relTol", cfg.get("p_relTol")),
        ("pFinal", "solver", cfg.get("pFinal_solver", cfg.get("p_solver"))),
        ("pFinal", "preconditioner", cfg.get("pFinal_preconditioner", cfg.get("p_preconditioner"))),
        ("pFinal", "tolerance", cfg.get("pFinal_tolerance", cfg.get("p_tolerance"))),
        ("pFinal", "relTol", cfg.get("pFinal_relTol")),
        ("U", "tolerance", cfg.get("U_tolerance")),
        ("U", "relTol", cfg.get("U_relTol")),
    ]
    for block_name, key, value in block_specs:
        if value is None or str(value).strip() == "":
            continue
        patch_entry(block_name, key, str(value))

    fv_solution.write_text(text)


def write_dye_parabolic_velocity_field(case_dir: Path, *, u_ref: float, mesh: dict, geometry: dict):
    nx = int(mesh.get("nx", 128))
    ny = int(mesh.get("ny", 48))
    nz = int(mesh.get("nz", 1))
    ly = float(geometry.get("ly", 1.0))
    lz = float(geometry.get("lz", mesh.get("thickness", 0.01)))
    dy = ly / max(ny, 1)
    dz = lz / max(nz, 1)

    def parabolic_u(y: float) -> float:
        eta = min(max(y / max(ly, 1e-12), 0.0), 1.0)
        return 4.0 * u_ref * eta * (1.0 - eta)

    cell_vectors: list[str] = []
    for k in range(nz):
        _z = (k + 0.5) * dz
        for j in range(ny):
            y = (j + 0.5) * dy
            ux = parabolic_u(y)
            row = f"({ux:.8g} 0 0)"
            for _i in range(nx):
                cell_vectors.append(row)

    inlet_vectors: list[str] = []
    for k in range(nz):
        _z = (k + 0.5) * dz
        for j in range(ny):
            y = (j + 0.5) * dy
            ux = parabolic_u(y)
            inlet_vectors.append(f"({ux:.8g} 0 0)")

    u_text = (
        "FoamFile { version 2.0; format ascii; class volVectorField; object U; }\n\n"
        "dimensions      [0 1 -1 0 0 0 0];\n\n"
        f"internalField   nonuniform List<vector>\n{len(cell_vectors)}\n(\n"
        + "\n".join(cell_vectors)
        + "\n);\n\n"
        "boundaryField\n"
        "{\n"
        "    inlet\n"
        "    {\n"
        "        type fixedValue;\n"
        f"        value nonuniform List<vector>\n{len(inlet_vectors)}\n(\n"
        + "\n".join(inlet_vectors)
        + "\n        );\n"
        "    }\n"
        "    outlet\n"
        "    {\n"
        "        type zeroGradient;\n"
        "    }\n"
        "    walls\n"
        "    {\n"
        "        type noSlip;\n"
        "    }\n"
        "    frontAndBack\n"
        "    {\n"
        "        type empty;\n"
        "    }\n"
        "}\n"
    )
    write_file(case_dir / "0" / "U", u_text)


def apply_rheology_replacements(case_dir: Path, *, nu_ref: float, cfg: dict):
    tprop = case_dir / "constant" / "transportProperties"
    if not tprop.exists():
        return {}

    rheo_cfg = cfg.get("rheology")
    if not isinstance(rheo_cfg, dict):
        return {}

    model = str(rheo_cfg.get("model", "powerLaw")).strip()
    model_l = model.lower()
    if model_l in {"powerlaw", "power_law"}:
        model_name = "powerLaw"
    elif model_l in {"herschelbulkley", "herschel_bulkley"}:
        model_name = "HerschelBulkley"
    else:
        raise ValueError(f"Unsupported rheology model: {model}")

    n_val = float(rheo_cfg.get("n", 0.6 if model_name == "powerLaw" else 0.8))
    k_factor = float(rheo_cfg.get("k_factor", 1.0))
    nu_min_ratio = float(rheo_cfg.get("nu_min_ratio", 0.1))
    nu_max_ratio = float(rheo_cfg.get("nu_max_ratio", 10.0))
    tau0 = float(rheo_cfg.get("tau0", 0.0 if model_name == "powerLaw" else 0.01))

    nu0 = max(1e-12, float(nu_ref))
    numin = max(1e-12, nu0 * nu_min_ratio)
    numax = max(numin, nu0 * nu_max_ratio)
    k_val = max(1e-12, nu0 * k_factor)

    replace_in_file(
        tprop,
        {
            "__RHEO_MODEL__": model_name,
            "__NU__": f"{nu0:.8e}",
            "__NU0__": f"{nu0:.8e}",
            "__NUMIN__": f"{numin:.8e}",
            "__NUMAX__": f"{numax:.8e}",
            "__K__": f"{k_val:.8e}",
            "__N__": f"{n_val:.8g}",
            "__TAU0__": f"{tau0:.8e}",
        },
    )

    return {
        "model": model_name,
        "nu0": nu0,
        "nuMin": numin,
        "nuMax": numax,
        "k": k_val,
        "n": n_val,
        "tau0": tau0,
    }


def write_file(path: Path, text: str):
    path.write_text(text.rstrip() + "\n")


def patch_type_for_name(name: str) -> str:
    lname = name.lower()
    if "frontandback" in lname or lname == "axis":
        return "empty"
    if "wedge" in lname:
        return "wedge"
    return "generic"


def _set_internal_field(text: str, value: str) -> str:
    pattern = re.compile(r"(?m)^(\s*internalField\s+)[^;]+;")
    if pattern.search(text):
        return pattern.sub(rf"\1{value};", text, count=1)
    return text


def _upsert_keyed_entry(block: str, key: str, value: str) -> str:
    pattern = re.compile(rf"(?m)^(\s*{re.escape(key)}\s+)[^;]+;")
    if pattern.search(block):
        return pattern.sub(rf"\1{value};", block, count=1)

    lines = block.splitlines()
    lines.append(f"        {key} {value};")
    return "\n".join(lines)


def _set_patch_block(text: str, patch_name: str, patch_cfg: dict) -> str:
    patch_re = re.compile(
        rf"(?ms)(^\s*{re.escape(patch_name)}\s*(?:\n\s*)?\{{)(.*?)(^\s*\}})",
        flags=re.MULTILINE,
    )
    m = patch_re.search(text)
    if not m:
        return text

    head, body, tail = m.group(1), m.group(2), m.group(3)
    if "type" in patch_cfg and str(patch_cfg["type"]).strip():
        body = _upsert_keyed_entry(body, "type", str(patch_cfg["type"]).strip())
    if "value" in patch_cfg and str(patch_cfg["value"]).strip():
        body = _upsert_keyed_entry(body, "value", str(patch_cfg["value"]).strip())

    return text[: m.start()] + f"{head}{body}{tail}" + text[m.end() :]


def apply_startup_velocity_ramp(case_dir: Path, *, cfg: dict, u_ref: float):
    ramp_duration = cfg.get("startup_ramp_duration")
    if ramp_duration is None:
        return

    ramp_duration = float(ramp_duration)
    if ramp_duration <= 0.0:
        return

    u_path = case_dir / "0" / "U"
    if not u_path.exists():
        return

    text = u_path.read_text()
    patches = cfg.get("startup_ramp_patches", ["inlet"])
    if isinstance(patches, str):
        patches = [p.strip() for p in patches.split(",") if p.strip()]
    if not isinstance(patches, list):
        patches = ["inlet"]

    def patch_with_ramp(src: str, patch_name: str) -> str:
        patch_re = re.compile(
            rf"(?ms)^(\s*){re.escape(patch_name)}\s*(?:\n\s*)?\{{(?P<body>.*?)^\s*\}}",
            flags=re.MULTILINE,
        )
        patch_match = patch_re.search(src)
        if not patch_match:
            return src
        target = (float(u_ref), 0.0, 0.0)
        value_match = re.search(
            r"value\s+uniform\s+\(\s*([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\s*\)\s*;",
            patch_match.group("body"),
        )
        if value_match:
            target = tuple(float(value_match.group(i)) for i in range(1, 4))
        indent = patch_match.group(1)
        body_indent = indent + "    "
        coded_block = f"""{indent}{patch_name}
{indent}{{
{body_indent}type uniformFixedValue;
{body_indent}value uniform (0 0 0);
{body_indent}uniformValue table
{body_indent}(
{body_indent}    (0 (0 0 0))
{body_indent}    ({0.5 * ramp_duration:.8g} ({0.5 * target[0]:.8g} {0.5 * target[1]:.8g} {0.5 * target[2]:.8g}))
{body_indent}    ({ramp_duration:.8g} ({target[0]:.8g} {target[1]:.8g} {target[2]:.8g}))
{body_indent});
{indent}}}"""
        return src[: patch_match.start()] + coded_block + src[patch_match.end() :]

    for patch in patches:
        text = patch_with_ramp(text, str(patch))

    if bool(cfg.get("startup_zero_internal_U", False)):
        text = re.sub(
            r"(?m)^(\s*internalField\s+uniform\s+)\([^;]+\);",
            r"\g<1>(0 0 0);",
            text,
            count=1,
        )

    u_path.write_text(text)


def apply_bc_ic_overrides(case_dir: Path, *, cfg: dict):
    bc_overrides = cfg.get("bc_overrides")
    ic_overrides = cfg.get("ic_overrides")
    if not isinstance(bc_overrides, dict) and not isinstance(ic_overrides, dict):
        return

    fields = set()
    if isinstance(bc_overrides, dict):
        fields.update(str(k) for k in bc_overrides.keys())
    if isinstance(ic_overrides, dict):
        for k in ic_overrides.keys():
            ks = str(k)
            if ks.endswith("_internal"):
                fields.add(ks[: -len("_internal")])
            else:
                fields.add(ks)

    for field in sorted(fields):
        field_path = case_dir / "0" / field
        if not field_path.exists():
            continue

        txt = field_path.read_text()

        if isinstance(ic_overrides, dict):
            internal_raw = None
            if field in ic_overrides:
                internal_raw = ic_overrides[field]
            elif f"{field}_internal" in ic_overrides:
                internal_raw = ic_overrides[f"{field}_internal"]
            if internal_raw is not None:
                internal_value = str(internal_raw).strip()
                if internal_value:
                    if not internal_value.startswith("uniform"):
                        internal_value = f"uniform {internal_value}"
                    txt = _set_internal_field(txt, internal_value)

        if isinstance(bc_overrides, dict):
            field_bc = bc_overrides.get(field)
            if isinstance(field_bc, dict):
                for patch_name, patch_cfg in field_bc.items():
                    if isinstance(patch_cfg, dict):
                        txt = _set_patch_block(txt, str(patch_name), patch_cfg)

        field_path.write_text(txt)


def sanitize_enclosed_lid_velocity(case_dir: Path, *, cfg: dict):
    case_prefix = str(cfg.get("case_prefix", "")).strip()
    if case_prefix not in {"cavity2d_turb", "lidDrivenCavity3d"}:
        return

    u_path = case_dir / "0" / "U"
    if not u_path.exists():
        return

    text = u_path.read_text()
    patch_re = re.compile(r"(?ms)(^\s*movingWall\s*(?:\n\s*)?\{)(.*?)(^\s*\})", flags=re.MULTILINE)
    value_re = re.compile(
        r"(value\s+uniform\s*\(\s*)([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)(\s*\)\s*;)"
    )

    patch_match = patch_re.search(text)
    if not patch_match:
        return

    body = patch_match.group(2)
    value_match = value_re.search(body)
    if not value_match:
        return

    ux = float(value_match.group(2))
    uy = float(value_match.group(3))
    uz = float(value_match.group(4))
    if abs(uy) <= 1e-12:
        return

    new_value = f"{value_match.group(1)}{ux:.7g} 0 {uz:.7g}{value_match.group(5)}"
    new_body = body[: value_match.start()] + new_value + body[value_match.end() :]
    new_text = text[: patch_match.start(2)] + new_body + text[patch_match.end(2) :]
    u_path.write_text(new_text)

    bc_overrides = cfg.get("bc_overrides")
    if isinstance(bc_overrides, dict):
        field_u = bc_overrides.get("U")
        if isinstance(field_u, dict):
            moving_wall = field_u.get("movingWall")
            if isinstance(moving_wall, dict):
                moving_wall["value"] = f"uniform ({ux:.7g} 0 {uz:.7g})"


def extract_patch_names_from_U(case_dir: Path):
    u_path = case_dir / "0" / "U"
    names = []
    lines = u_path.read_text().splitlines()
    inside_boundary = False
    level = 0
    for ln in lines:
        s = ln.strip()
        if s.startswith("boundaryField"):
            inside_boundary = True
            continue
        if not inside_boundary:
            continue
        level += s.count("{")
        if level == 1 and s and not s.startswith("{") and not s.startswith("}"):
            token = s.split()[0]
            if token.isidentifier():
                names.append(token)
        level -= s.count("}")
        if inside_boundary and level <= 0 and s == "}":
            break
    return names


def setup_turbulence_komega_sst(case_dir: Path, *, u_ref: float, l_ref: float, nu: float, cfg: dict):
    i_turb = float(cfg.get("turbulence_intensity", 0.05))
    l_turb = float(cfg.get("turbulence_length_scale", max(1e-4, 0.07 * l_ref)))
    k0 = max(1e-10, 1.5 * (u_ref * i_turb) ** 2)
    omega0 = max(1e-8, math.sqrt(k0) / (pow(0.09, 0.25) * l_turb))
    nut0 = max(1e-10, k0 / omega0)
    is_buoyant_case = (case_dir / "0" / "p_rgh").exists() and (case_dir / "0" / "T").exists()

    patch_names = extract_patch_names_from_U(case_dir)
    if not patch_names:
        patch_names = ["inlet", "outlet", "walls"]

    def k_bc(name: str) -> str:
        lname = name.lower()
        ptype = patch_type_for_name(name)
        if ptype == "empty":
            return "        type empty;"
        if ptype == "wedge":
            return "        type wedge;"
        if "inlet" in lname:
            return f"        type fixedValue;\n        value uniform {k0:.8e};"
        if "outlet" in lname:
            return "        type zeroGradient;"
        if "wall" in lname:
            return "        type kqRWallFunction;\n        value uniform 0;"
        return "        type zeroGradient;"

    def omega_bc(name: str) -> str:
        lname = name.lower()
        ptype = patch_type_for_name(name)
        if ptype == "empty":
            return "        type empty;"
        if ptype == "wedge":
            return "        type wedge;"
        if "inlet" in lname:
            return f"        type fixedValue;\n        value uniform {omega0:.8e};"
        if "outlet" in lname:
            return "        type zeroGradient;"
        if "wall" in lname:
            return "        type omegaWallFunction;\n        value uniform 1e-08;"
        return "        type zeroGradient;"

    def nut_bc(name: str) -> str:
        lname = name.lower()
        ptype = patch_type_for_name(name)
        if ptype == "empty":
            return "        type empty;"
        if ptype == "wedge":
            return "        type wedge;"
        if "inlet" in lname:
            return f"        type fixedValue;\n        value uniform {nut0:.8e};"
        if "outlet" in lname:
            return "        type zeroGradient;"
        if "wall" in lname:
            return "        type nutkWallFunction;\n        value uniform 0;"
        return "        type zeroGradient;"

    def alphat_bc(name: str) -> str:
        lname = name.lower()
        ptype = patch_type_for_name(name)
        if ptype == "empty":
            return "        type empty;"
        if ptype == "wedge":
            return "        type wedge;"
        if "wall" in lname:
            # OpenFOAM v2506 (buoyantBoussinesqPimpleFoam) does not accept
            # compressible::alphatWallFunction in this context.
            return "        type alphatJayatillekeWallFunction;\n        Prt 0.85;\n        value uniform 0;"
        return "        type calculated;\n        value uniform 0;"

    def build_boundary(section_fn):
        parts = ["boundaryField", "{"]
        for pname in patch_names:
            parts.extend([
                f"    {pname}",
                "    {",
                section_fn(pname),
                "    }",
            ])
        parts.append("}")
        return "\n".join(parts)

    write_file(
        case_dir / "0" / "k",
        f"""FoamFile {{ version 2.0; format ascii; class volScalarField; object k; }}

dimensions      [0 2 -2 0 0 0 0];

internalField   uniform {k0:.8e};

{build_boundary(k_bc)}
""",
    )
    write_file(
        case_dir / "0" / "omega",
        f"""FoamFile {{ version 2.0; format ascii; class volScalarField; object omega; }}

dimensions      [0 0 -1 0 0 0 0];

internalField   uniform {omega0:.8e};

{build_boundary(omega_bc)}
""",
    )
    write_file(
        case_dir / "0" / "nut",
        f"""FoamFile {{ version 2.0; format ascii; class volScalarField; object nut; }}

dimensions      [0 2 -1 0 0 0 0];

internalField   uniform {nut0:.8e};

{build_boundary(nut_bc)}
""",
    )

    write_file(
        case_dir / "constant" / "turbulenceProperties",
        """FoamFile { version 2.0; format ascii; class dictionary; object turbulenceProperties; }

simulationType RAS;

RAS
{
    RASModel        kOmegaSST;
    turbulence      on;
    printCoeffs     on;
}
""",
    )

    if is_buoyant_case:
        write_file(
            case_dir / "0" / "alphat",
            f"""FoamFile {{ version 2.0; format ascii; class volScalarField; object alphat; }}

dimensions      [0 2 -1 0 0 0 0];

internalField   uniform 0;

{build_boundary(alphat_bc)}
""",
        )

        write_file(
            case_dir / "system" / "fvSchemes",
            """FoamFile { version 2.0; format ascii; class dictionary; object fvSchemes; }

ddtSchemes { default Euler; }

gradSchemes { default Gauss linear; }

divSchemes
{
    default                             none;
    div(phi,U)                          Gauss linearUpwind grad(U);
    div(phi,T)                          Gauss upwind;
    div(phi,k)                          Gauss upwind;
    div(phi,omega)                      Gauss upwind;
    div((nuEff*dev2(T(grad(U)))))       Gauss linear;
}

laplacianSchemes { default Gauss linear corrected; }

interpolationSchemes { default linear; }

snGradSchemes { default corrected; }

wallDist
{
    method meshWave;
}
""",
        )

        write_file(
            case_dir / "system" / "fvSolution",
            """FoamFile { version 2.0; format ascii; class dictionary; object fvSolution; }

solvers
{
    p_rgh
    {
        solver          PCG;
        preconditioner  DIC;
        tolerance       1e-7;
        relTol          0.01;
    }
    p_rghFinal
    {
        $p_rgh;
        relTol          0;
    }
    U
    {
        solver          smoothSolver;
        smoother        symGaussSeidel;
        tolerance       1e-8;
        relTol          0.01;
    }
    UFinal
    {
        $U;
        relTol          0;
    }
    T
    {
        solver          smoothSolver;
        smoother        symGaussSeidel;
        tolerance       1e-8;
        relTol          0.01;
    }
    TFinal
    {
        $T;
        relTol          0;
    }
    k
    {
        solver          smoothSolver;
        smoother        symGaussSeidel;
        tolerance       1e-8;
        relTol          0.01;
    }
    kFinal
    {
        $k;
        relTol          0;
    }
    omega
    {
        solver          smoothSolver;
        smoother        symGaussSeidel;
        tolerance       1e-8;
        relTol          0.01;
    }
    omegaFinal
    {
        $omega;
        relTol          0;
    }
}

PIMPLE
{
    nOuterCorrectors             1;
    nCorrectors                  2;
    nNonOrthogonalCorrectors     0;
    pRefCell                     0;
    pRefValue                    0;
}

relaxationFactors
{
    equations
    {
        U       0.7;
        T       0.7;
        k       0.7;
        omega   0.7;
        p_rgh   0.3;
    }
}
""",
        )
    else:
        write_file(
            case_dir / "system" / "fvSchemes",
            """FoamFile { version 2.0; format ascii; class dictionary; object fvSchemes; }

ddtSchemes { default Euler; }

gradSchemes { default Gauss linear; }

divSchemes
{
    default                             none;
    div(phi,U)                          Gauss linearUpwind grad(U);
    div(phi,k)                          Gauss upwind;
    div(phi,omega)                      Gauss upwind;
    div((nuEff*dev2(T(grad(U)))))       Gauss linear;
}

laplacianSchemes { default Gauss linear corrected; }

interpolationSchemes { default linear; }

snGradSchemes { default corrected; }

wallDist
{
    method meshWave;
}
""",
        )

        write_file(
            case_dir / "system" / "fvSolution",
            """FoamFile { version 2.0; format ascii; class dictionary; object fvSolution; }

solvers
{
    p
    {
        solver          PCG;
        preconditioner  DIC;
        tolerance       1e-7;
        relTol          0.01;
    }
    pFinal
    {
        solver          PCG;
        preconditioner  DIC;
        tolerance       1e-7;
        relTol          0;
    }
    U
    {
        solver          smoothSolver;
        smoother        symGaussSeidel;
        tolerance       1e-8;
        relTol          0.01;
    }
    UFinal
    {
        $U;
        relTol          0;
    }
    k
    {
        solver          smoothSolver;
        smoother        symGaussSeidel;
        tolerance       1e-8;
        relTol          0.01;
    }
    kFinal
    {
        $k;
        relTol          0;
    }
    omega
    {
        solver          smoothSolver;
        smoother        symGaussSeidel;
        tolerance       1e-8;
        relTol          0.01;
    }
    omegaFinal
    {
        $omega;
        relTol          0;
    }
}

PIMPLE
{
    nOuterCorrectors             1;
    nCorrectors                  2;
    nNonOrthogonalCorrectors     0;
    pRefCell                     0;
    pRefValue                    0;
}

relaxationFactors
{
    equations
    {
        U       0.7;
        k       0.7;
        omega   0.7;
        p       0.3;
    }
}
""",
        )


def generate_from_config(cfg_path: Path):
    cfg = json.loads(cfg_path.read_text())
    template = Path(cfg["template"])
    out_root = Path(os.environ.get("FOAM_CASES_OUT_ROOT", cfg["out_root"]))
    out_root.mkdir(parents=True, exist_ok=True)

    force_regen = os.environ.get("FORCE_REGEN_CASES", "0") == "1"

    case_prefix = cfg.get("case_prefix") or cfg.get("scene", "case").replace("-", "_")
    raw_suffix = str(cfg.get("case_suffix", "")).strip()
    case_suffix = re.sub(r"[^A-Za-z0-9_-]+", "", raw_suffix)
    re_list = cfg.get("Re_list", [100, 400, 1000])
    representative_re_only = os.environ.get("REPRESENTATIVE_RE_ONLY", "0") == "1"
    if representative_re_only:
        uniq = sorted({float(x) for x in re_list})
        if len(uniq) > 3:
            re_list = [uniq[0], uniq[len(uniq) // 2], uniq[-1]]
        else:
            re_list = uniq

    u_ref = float(cfg.get("U_lid", cfg.get("U_inlet", 1.0)))
    l_ref = float(cfg.get("L", 1.0))
    solver = str(cfg.get("solver", "icoFoam"))
    is_scalar_transport = solver == "scalarTransportFoam"
    turbulence_model = str(cfg.get("turbulence_model", "laminar"))
    laminar_re_max = cfg.get("laminar_re_max")
    auto_turbulence_for_high_re = bool(cfg.get("auto_turbulence_for_high_re", False))
    auto_turbulence_solver = str(cfg.get("auto_turbulence_solver", "pimpleFoam"))
    auto_turbulence_model = str(cfg.get("auto_turbulence_model", "kOmegaSST"))
    turbulence_re_threshold = float(cfg.get("turbulence_re_threshold", 5000.0))
    dt = float(cfg.get("dt", 0.01))
    end_time = float(cfg.get("endTime", 10.0))
    write_interval_sec = float(cfg.get("writeInterval", 0.1))
    write_steps = max(1, int(round(write_interval_sec / dt)))

    mesh = cfg.get("mesh", {})
    geometry = cfg.get("geometry", {})
    if "thickness" in mesh and "lz" not in geometry:
        geometry = {**geometry, "lz": mesh["thickness"]}

    variants: list[dict] = [{"label": case_suffix}] if case_suffix else [{"label": ""}]
    if is_scalar_transport:
        band_cfgs = cfg.get("pulse_bands")
        if isinstance(band_cfgs, list) and band_cfgs:
            band_variants: list[dict] = []
            for idx, item in enumerate(band_cfgs):
                if not isinstance(item, dict):
                    continue
                label = re.sub(r"[^A-Za-z0-9_-]+", "", str(item.get("label", f"band{idx+1}")).strip())
                if not label:
                    label = f"band{idx+1}"
                band_variants.append(
                    {
                        "label": label,
                        "band_label": label,
                        "pulse_y_min": item.get("pulse_y_min"),
                        "pulse_y_max": item.get("pulse_y_max"),
                        "pulse_y_min_frac": item.get("pulse_y_min_frac"),
                        "pulse_y_max_frac": item.get("pulse_y_max_frac"),
                    }
                )
            if band_variants:
                variants = band_variants

        duration_cfgs = cfg.get("pulse_variants")
        if isinstance(duration_cfgs, list) and duration_cfgs:
            duration_variants: list[dict] = []
            for idx, item in enumerate(duration_cfgs):
                if not isinstance(item, dict):
                    continue
                label = re.sub(r"[^A-Za-z0-9_-]+", "", str(item.get("label", f"dur{idx+1}")).strip())
                if not label:
                    label = f"dur{idx+1}"
                duration_variants.append(
                    {
                        "label": label,
                        "duration_label": label,
                        "pulse_duration": item.get("pulse_duration"),
                        "pulse_ramp_duration": item.get("pulse_ramp_duration"),
                    }
                )
            if duration_variants:
                combined: list[dict] = []
                for base in variants:
                    for duration in duration_variants:
                        base_label = str(base.get("label", "")).strip()
                        dur_label = str(duration.get("label", "")).strip()
                        combo_label = "_".join(part for part in [base_label, dur_label] if part)
                        merged = dict(base)
                        merged.update(duration)
                        merged["label"] = combo_label
                        combined.append(merged)
                variants = combined or variants

    generated = 0
    skipped = 0
    skipped_high_re_laminar = 0
    for variant in variants:
        variant_label = str(variant.get("label", "")).strip()
        variant_suffix = re.sub(r"[^A-Za-z0-9_-]+", "", variant_label)
        if is_scalar_transport and case_suffix:
            variant_suffix = "_".join(part for part in [variant_suffix, case_suffix] if part)
        for re_value in re_list:
            re_value = float(re_value)
            nu = u_ref * l_ref / re_value
            scalar_diffusivity = None
            if is_scalar_transport:
                scalar_diffusivity = nu

            case_solver = solver
            case_turbulence_model = turbulence_model
            if (
                auto_turbulence_for_high_re
                and case_turbulence_model.lower() == "laminar"
                and re_value >= turbulence_re_threshold
            ):
                case_solver = auto_turbulence_solver
                case_turbulence_model = auto_turbulence_model

            if (
                case_turbulence_model.lower() == "laminar"
                and laminar_re_max is not None
                and re_value > float(laminar_re_max)
            ):
                print(
                    f"[gen_cases] skip high-Re laminar case: Re={re_value:g} > "
                    f"laminar_re_max={float(laminar_re_max):g} ({cfg_path})"
                )
                skipped += 1
                skipped_high_re_laminar += 1
                continue

            case_name = f"{case_prefix}_Re{int(re_value)}"
            if variant_suffix:
                case_name = f"{case_name}_{variant_suffix}"
            case_dir = out_root / case_name
            if case_dir.exists():
                if force_regen:
                    shutil.rmtree(case_dir)
                else:
                    print(f"[gen_cases] skip existing case: {case_dir}")
                    skipped += 1
                    continue

            shutil.copytree(template, case_dir)
            generated += 1

            apply_common_replacements(
                case_dir,
                u_ref=u_ref,
                nu=nu,
                dt=dt,
                end_time=end_time,
                write_steps=write_steps,
                mesh=mesh,
                geometry=geometry,
            )
            apply_solver_controls(case_dir, cfg=cfg)
            if is_scalar_transport:
                pulse_duration = float(variant.get("pulse_duration", cfg.get("pulse_duration", max(dt, 0.4 * l_ref / max(u_ref, 1e-8)))))
                pulse_ramp_duration = float(variant.get("pulse_ramp_duration", cfg.get("pulse_ramp_duration", min(0.25 * pulse_duration, max(dt, 0.08 * l_ref / max(u_ref, 1e-8))))))
                pulse_value = float(cfg.get("pulse_value", 1.0))
                pulse_outlet_value = float(cfg.get("pulse_outlet_value", 0.0))
                ly = float(geometry.get("ly", 1.0))
                if variant.get("pulse_y_min") is not None:
                    pulse_y_min = float(variant.get("pulse_y_min", 0.0))
                elif "pulse_y_min" in cfg:
                    pulse_y_min = float(cfg.get("pulse_y_min", 0.0))
                else:
                    pulse_y_min = float(variant.get("pulse_y_min_frac", cfg.get("pulse_y_min_frac", 0.35))) * ly
                if variant.get("pulse_y_max") is not None:
                    pulse_y_max = float(variant.get("pulse_y_max", ly))
                elif "pulse_y_max" in cfg:
                    pulse_y_max = float(cfg.get("pulse_y_max", ly))
                else:
                    pulse_y_max = float(variant.get("pulse_y_max_frac", cfg.get("pulse_y_max_frac", 0.65))) * ly
                pulse_y_min = max(0.0, min(pulse_y_min, ly))
                pulse_y_max = max(pulse_y_min, min(pulse_y_max, ly))
                write_dye_parabolic_velocity_field(case_dir, u_ref=u_ref, mesh=mesh, geometry=geometry)
                replace_in_file(
                    case_dir / "0" / "T",
                    {
                        "__PULSE_T_END__": f"{pulse_duration:.8g}",
                        "__PULSE_RAMP__": f"{pulse_ramp_duration:.8g}",
                        "__PULSE_VALUE__": f"{pulse_value:.8g}",
                        "__PULSE_OUTLET_VALUE__": f"{pulse_outlet_value:.8g}",
                        "__PULSE_Y_MIN__": f"{pulse_y_min:.8g}",
                        "__PULSE_Y_MAX__": f"{pulse_y_max:.8g}",
                    },
                )
            if is_scalar_transport:
                replace_in_file(
                    case_dir / "constant" / "transportProperties",
                    {
                        "DT              DT [0 2 -1 0 0 0 0] 1e-4;": f"DT              DT [0 2 -1 0 0 0 0] {scalar_diffusivity:.8e};",
                        "DT [0 2 -1 0 0 0 0] 1e-4;": f"DT [0 2 -1 0 0 0 0] {scalar_diffusivity:.8e};",
                    },
                )
            rheology_meta = apply_rheology_replacements(case_dir, nu_ref=nu, cfg=cfg)

            replace_in_file(case_dir / "system" / "controlDict", {
                "application     icoFoam;": f"application     {case_solver};",
                "application     pimpleFoam;": f"application     {case_solver};",
                "application     buoyantBoussinesqPimpleFoam;": f"application     {case_solver};",
            })

            if case_turbulence_model.lower() == "komegasst":
                setup_turbulence_komega_sst(
                    case_dir,
                    u_ref=u_ref,
                    l_ref=l_ref,
                    nu=nu,
                    cfg=cfg,
                )

            apply_bc_ic_overrides(case_dir, cfg=cfg)
            apply_startup_velocity_ramp(case_dir, cfg=cfg, u_ref=u_ref)
            sanitize_enclosed_lid_velocity(case_dir, cfg=cfg)
            ensure_quality_function_objects(case_dir)

            meta = {
                "domain": cfg.get("domain", "fluid"),
                "scene": cfg.get("scene", case_prefix),
                "case_prefix": case_prefix,
                "Re": re_value,
                "U_ref": u_ref,
                "L_ref": l_ref,
                "nu": nu,
                "Pe": re_value if is_scalar_transport else None,
                "dt": dt,
                "endTime": end_time,
                "writeInterval": write_interval_sec,
                "mesh": mesh,
                "geometry": geometry,
                "template": str(template),
                "solver": case_solver,
                "turbulence_model": case_turbulence_model,
            }
            if scalar_diffusivity is not None:
                meta["scalar_diffusivity"] = scalar_diffusivity
                meta["pulse_duration"] = pulse_duration
                meta["pulse_ramp_duration"] = pulse_ramp_duration
                meta["pulse_value"] = float(cfg.get("pulse_value", 1.0))
                meta["pulse_y_min"] = pulse_y_min
                meta["pulse_y_max"] = pulse_y_max
                if variant.get("band_label"):
                    meta["pulse_band"] = str(variant.get("band_label"))
                if variant.get("duration_label"):
                    meta["pulse_duration_label"] = str(variant.get("duration_label"))
            if variant_suffix:
                meta["case_suffix"] = variant_suffix
            if rheology_meta:
                meta["rheology"] = rheology_meta
            if isinstance(cfg.get("bc_overrides"), dict):
                meta["bc_overrides"] = cfg.get("bc_overrides")
            if isinstance(cfg.get("ic_overrides"), dict):
                meta["ic_overrides"] = cfg.get("ic_overrides")
            (case_dir / "case_meta.json").write_text(json.dumps(meta, indent=2))

    print(
        f"Generated {generated} cases from {cfg_path} in {out_root} "
        f"(skipped {skipped}, skipped_high_re_laminar={skipped_high_re_laminar}, "
        f"representative_re_only={int(representative_re_only)}, force_regen={int(force_regen)})"
    )


def default_config_paths():
    defaults = [
        Path("configs/sweep.json"),
        Path("configs/sweep_cavity2d_turbulent.json"),
        Path("configs/sweep_channel2d.json"),
        Path("configs/sweep_pipe3d.json"),
        Path("configs/sweep_jetaxi.json"),
        Path("configs/sweep_channel2d_turbulent.json"),
        Path("configs/sweep_pipe3d_turbulent.json"),
        Path("configs/sweep_jetaxi_turbulent.json"),
        Path("configs/sweep_suddenExpansion2d.json"),
        Path("configs/sweep_suddenContraction2d.json"),
        Path("configs/sweep_flowPastSquare2d.json"),
        Path("configs/sweep_flowPastCylinder2d.json"),
        Path("configs/sweep_naca0012Airfoil2d.json"),
        Path("configs/sweep_wakeInteraction2d.json"),
        Path("configs/sweep_backwardFacingStep2d.json"),
        Path("configs/sweep_dyeAdvection2d.json"),
        Path("configs/sweep_buoyancyCavity2d.json"),
        Path("configs/sweep_rayleighBenard2d.json"),
        Path("configs/sweep_lidDrivenCavity3d.json"),
        Path("configs/sweep_channel2d_nonNewtonian_powerLaw.json"),
        Path("configs/sweep_channel2d_nonNewtonian_herschelBulkley.json"),
        Path("configs/sweep_suddenExpansion2d_nonNewtonian_powerLaw.json"),
        Path("configs/sweep_suddenExpansion2d_nonNewtonian_herschelBulkley.json"),
    ]
    return [p for p in defaults if p.exists()]


def main(args=None):
    args = list(sys.argv[1:] if args is None else args)
    cfg_paths = [Path(p) for p in args] if args else default_config_paths()
    if not cfg_paths:
        raise FileNotFoundError("No sweep config files found")

    for cfg_path in cfg_paths:
        generate_from_config(cfg_path)


if __name__ == "__main__":
    main()
