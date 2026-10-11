"""Physical descriptions from simulation inputs and computed quantities."""

from __future__ import annotations

import math

import re

from typing import Any, Sequence


def _num(value: Any, default: float | None = None) -> float | None:
    try:
        out = float(value)
    except Exception:
        return default
    if math.isfinite(out):
        return out
    return default


def _fmt_num(value: Any, nd: int = 5, fallback: str = "not recorded") -> str:
    x = _num(value)
    if x is None:
        return fallback
    ax = abs(x)
    if ax != 0.0 and (ax < 1e-3 or ax >= 1e5):
        return f"{x:.{max(1, nd - 1)}e}"
    return f"{x:.{nd}g}"


def _case_dim(meta: dict[str, Any]) -> str:
    raw = str(meta.get("dim", "")).strip().lower()
    if raw in {"2d", "3d"}:
        return raw.upper()
    geom = meta.get("geometry") if isinstance(meta.get("geometry"), dict) else {}
    return "3D" if str(geom.get("shape_3d", "")).strip() else "2D"


_PROBLEM_PHRASES = {
    "biaxialTension": "biaxial tensile loading of a thin plate",
    "bucklingSnapthrough": "compressive buckling of a slender column",
    "cantileverLocalization": "cantilever bending with localized deformation",
    "contactPatchCompression": "localized patch compression of a block",
    "cubeCompression3D": "compression of a three-dimensional cube",
    "cylinderTorsion3D": "torsion of a cylindrical shaft",
    "hyperelasticNeoHookean3D": "large-deformation Neo-Hookean response in a three-dimensional solid",
    "hyperelasticNeoHookeanCuboid3D": "large-deformation Neo-Hookean response in a cuboid",
    "hyperelasticNeoHookeanSphere3D": "large-deformation Neo-Hookean response in a sphere-like body",
    "loadUnloadHysteresis": "load-unload plastic hysteresis in tension",
    "longCuboidBending3D": "bending of a long three-dimensional cuboid",
    "nearIncompressibleBlock": "near-incompressible deformation of a block",
    "necking": "plastic necking in a tensile bar",
    "neoHookean": "large-deformation Neo-Hookean response in a block",
    "notchedBarTension3D": "tension of a three-dimensional notched bar",
    "plasticJ2CylinderTension3D": "J2 plastic tension of a cylindrical specimen",
    "plasticJ2LargeStrain3D": "large-strain J2 plastic deformation of a three-dimensional solid",
    "plasticJ2NotchedBar3D": "J2 plastic tension of a three-dimensional notched bar",
    "plateWithHole": "remote tension of a plate with a circular hole",
    "plateWithHole3D": "remote tension of a three-dimensional plate with a circular hole",
    "pureShear": "pure shear deformation of a rectangular solid",
    "sphereIndentation3D": "indentation of a sphere-like three-dimensional solid",
    "threePointBending": "three-point bending of a beam",
    "torsionShear": "torsion-shear deformation of a planar solid",
    "uniaxialCompression": "uniaxial compression of a solid body",
    "uniaxialTension": "uniaxial tension of a dogbone specimen",
    "yieldHardening": "plastic yielding with isotropic hardening in tension",
}

_GEOMETRY_LABELS = {
    "block": "rectangular block",
    "cube": "cube",
    "cuboid": "cuboid",
    "cylinder": "cylinder",
    "dogbone": "dogbone specimen",
    "notched_bar_3d": "three-dimensional notched bar",
    "plate": "thin plate",
    "plate_with_hole": "plate with a circular hole",
    "slender_column": "slender column",
    "solid_3d": "three-dimensional solid",
    "sphere": "sphere-like solid",
}

_MATERIAL_LABELS = {
    "linear_elastic": "linear elastic",
    "j2_plastic": "J2 plasticity",
    "neo_hookean": "Neo-Hookean hyperelastic",
    "neoHookean": "Neo-Hookean hyperelastic",
}

_LOADING_LABELS = {
    "axial_compression": "axial compression",
    "biaxial_displacement": "biaxial displacement control",
    "bending_control": "bending displacement control",
    "compression_control": "compression displacement control",
    "displacement_control": "displacement-controlled tension",
    "indentation_control": "indentation displacement control",
    "load_control": "load-controlled tension",
    "patch_compression": "localized patch compression",
    "pure_shear": "pure shear loading",
    "remote_tension": "remote tensile loading",
    "tip_force": "tip-force loading",
    "torsion_control": "torsional displacement control",
    "torsion_shear": "combined torsion-shear loading",
}

_FORBIDDEN_QUERY_TERMS = (
    "case_id",
    "release signature",
    "file name",
    "filename",
    "solver",
    "backend",
    "mesh resolution",
    "mesh",
    "fenicsx",
    "paraview",
)


def _plain_token(value: Any, fallback: str = "not recorded") -> str:
    if value is None:
        return fallback
    s = str(value).strip()
    return s if s else fallback


def _humanize_name(value: Any) -> str:
    s = _plain_token(value, "")
    s = s.replace("_", " ").replace("-", " ")
    s = re.sub(r"(?<=[a-z])(?=[A-Z0-9])", " ", s)
    s = re.sub(r"(?<=[0-9])(?=[A-Za-z])", " ", s)
    s = re.sub(r"\b([23])\s*d\b", lambda m: f"{m.group(1)}D", s, flags=re.IGNORECASE)
    s = re.sub(r"\bj\s*2\b", "J2", s, flags=re.IGNORECASE)
    s = re.sub(r"\bneo\s*hookean\b", "Neo-Hookean", s, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", s).strip()


def _clean_fragment(value: Any, fallback: str = "visible deformation develops") -> str:
    s = _plain_token(value, fallback)
    s = re.sub(r"\s+", " ", s).strip()
    return s.rstrip(".")


def _positive_num(value: Any) -> float | None:
    x = _num(value)
    if x is None or x <= 0.0:
        return None
    return x


def _fmt_about(value: Any, nd: int = 5) -> str:
    return _fmt_num(value, nd=nd)


def _join_natural(parts: Sequence[str]) -> str:
    items = [p.strip() for p in parts if p and p.strip()]
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + f", and {items[-1]}"


def _with_indefinite_article(phrase: str) -> str:
    s = phrase.strip()
    if not s or s.lower().startswith(("a ", "an ", "the ")):
        return s
    article = "an" if s[0].lower() in {"a", "e", "i", "o", "u"} else "a"
    return f"{article} {s}"


def _problem_phrase(case: dict[str, Any]) -> str:
    family = str(case.get("family") or case.get("scenario") or "").strip()
    if family in _PROBLEM_PHRASES:
        return _PROBLEM_PHRASES[family]
    return _humanize_name(family) or "a solid mechanics deformation problem"


def _geometry_label(scene: dict[str, Any]) -> str:
    geom = (
        scene.get("geometry_params")
        if isinstance(scene.get("geometry_params"), dict)
        else {}
    )
    raw = geom.get("shape_3d") or geom.get("kind") or scene.get("geometry_id")
    key = str(raw or "").strip()
    return _GEOMETRY_LABELS.get(key, _humanize_name(key) or "solid body")


def _geometry_dimension_value(scene: dict[str, Any]) -> str:
    geom = (
        scene.get("geometry_params")
        if isinstance(scene.get("geometry_params"), dict)
        else {}
    )
    lx = _positive_num(geom.get("lx"))
    ly = _positive_num(geom.get("ly"))
    lz = _positive_num(geom.get("lz"))
    thickness = _positive_num(geom.get("thickness"))
    parts: list[str] = []
    if lx is not None and ly is not None and lz is not None:
        parts.append(
            f"overall dimensions about {_fmt_about(lx)} by {_fmt_about(ly)} by {_fmt_about(lz)}"
        )
    elif lx is not None and ly is not None:
        parts.append(f"in-plane dimensions about {_fmt_about(lx)} by {_fmt_about(ly)}")
    elif lx is not None:
        parts.append(f"length about {_fmt_about(lx)}")
    if thickness is not None:
        parts.append(f"thickness about {_fmt_about(thickness)}")
    return _join_natural(parts)


def _ratio_value(num: Any, den: Any, label: str) -> str:
    a = _positive_num(num)
    b = _positive_num(den)
    if a is None or b is None:
        return ""
    return f"{label} about {_fmt_about(a / b, 4)}"


def _geometry_feature_values(scene: dict[str, Any]) -> list[str]:
    geom = (
        scene.get("geometry_params")
        if isinstance(scene.get("geometry_params"), dict)
        else {}
    )
    lx = geom.get("lx")
    ly = geom.get("ly")
    features: list[str] = []

    for key, label in [
        ("hole_radius", "hole-radius-to-length ratio"),
        ("sphere_radius", "sphere-radius-to-length ratio"),
        ("cylinder_radius", "cylinder-radius-to-length ratio"),
        ("neck_region_width", "neck-width-to-length ratio"),
        ("patch_half_width", "patch-half-width-to-length ratio"),
        ("support_span", "support-span-to-length ratio"),
        ("load_width", "load-width-to-length ratio"),
    ]:
        val = _ratio_value(geom.get(key), lx, label)
        if val:
            features.append(val)

    for key, label in [
        ("notch_depth_ratio", "notch depth ratio"),
        ("notch_width_ratio", "notch width ratio"),
        ("imperfection", "initial imperfection amplitude"),
        ("biaxial_ratio", "biaxial stretch ratio"),
        ("bulk_bias", "near-incompressibility bias"),
        ("warping_coeff", "warping coefficient"),
    ]:
        if geom.get(key) is not None:
            features.append(f"{label} about {_fmt_about(geom.get(key), 4)}")

    if geom.get("localization_band_x") is not None:
        val = _ratio_value(
            geom.get("localization_band_x"),
            lx,
            "localization-band position over length",
        )
        if val:
            features.append(val)
    if geom.get("twist_axis_x") is not None:
        val = _ratio_value(
            geom.get("twist_axis_x"), lx, "twist-axis position over length"
        )
        if val:
            features.append(val)
    if geom.get("hole_cx") is not None:
        val = _ratio_value(
            geom.get("hole_cx"), lx, "hole-center x-position over length"
        )
        if val:
            features.append(val)
    if geom.get("hole_cy") is not None:
        val = _ratio_value(
            geom.get("hole_cy"), ly, "hole-center y-position over height"
        )
        if val:
            features.append(val)
    return features


def _geometry_summary(scene: dict[str, Any]) -> str:
    label = _with_indefinite_article(_geometry_label(scene))
    dim_value = _geometry_dimension_value(scene)
    features = _geometry_feature_values(scene)
    parts = [label]
    if dim_value:
        parts.append(dim_value)
    if features:
        parts.append("with " + _join_natural(features[:3]))
    return ", ".join(parts)


def _material_label(raw: Any) -> str:
    key = str(raw or "").strip()
    return _MATERIAL_LABELS.get(key, _humanize_name(key) or "solid material")


def _material_summary(params: dict[str, Any]) -> str:
    parts = [_material_label(params.get("material_model"))]
    if params.get("E") is not None:
        parts.append(f"Young's modulus about {_fmt_about(params.get('E'), 5)}")
    if params.get("nu") is not None:
        parts.append(f"Poisson's ratio about {_fmt_about(params.get('nu'), 5)}")
    if params.get("yield_stress") is not None:
        parts.append(f"yield stress about {_fmt_about(params.get('yield_stress'), 5)}")
    if params.get("hardening_modulus") is not None:
        parts.append(
            f"hardening modulus about {_fmt_about(params.get('hardening_modulus'), 5)}"
        )
    return _join_natural(parts)


def _loading_label(raw: Any) -> str:
    key = str(raw or "").strip()
    return _LOADING_LABELS.get(key, _humanize_name(key) or "mechanical loading")


def _detail_value_text(value: Any, nd: int = 5) -> str:
    if isinstance(value, (list, tuple)):
        return "(" + ", ".join(_fmt_about(v, nd) for v in value) + ")"
    return _fmt_about(value, nd)


def _loading_detail_values(scene: dict[str, Any]) -> list[str]:
    loading = scene.get("loading") if isinstance(scene.get("loading"), dict) else {}
    out: list[str] = []
    for key, label in [
        ("twist_angle_deg", "twist angle"),
        ("indent_depth_ratio", "indentation-depth ratio"),
        ("biaxial_ratio", "biaxial loading ratio"),
        ("shear_ratio_xy", "shear ratio"),
        ("tip_force_direction", "tip-force direction"),
        ("center_drop_ratio", "center-drop ratio"),
        ("critical_load_factor", "critical-load factor"),
        ("damage_onset", "damage-onset threshold"),
        ("patch_peak_ratio", "patch-peak ratio"),
        ("twist_strength", "twist-strength parameter"),
        ("volumetric_weight", "volumetric penalty weight"),
    ]:
        if loading.get(key) is not None:
            value = loading.get(key)
            prefix = "" if isinstance(value, (list, tuple)) else "about "
            out.append(f"{label} {prefix}{_detail_value_text(value, 5)}")
    if loading.get("compress_axis") is not None:
        out.append(
            f"compression axis {_detail_value_text(loading.get('compress_axis'), 4)}"
        )
    return out


def _loading_summary(params: dict[str, Any], scene: dict[str, Any]) -> str:
    loading = scene.get("loading") if isinstance(scene.get("loading"), dict) else {}
    mode = _loading_label(params.get("load_mode") or loading.get("mode"))
    parts = [mode]
    load_factor = params.get("load_factor", loading.get("load_factor"))
    if load_factor is not None:
        parts.append(f"load factor about {_fmt_about(load_factor, 5)}")
    sign = _num(loading.get("load_sign"))
    if sign is not None and sign < 0.0 and "compression" not in mode:
        parts.append("compressive sign convention")
    details = _loading_detail_values(scene)
    parts.extend(details[:2])
    return _join_natural(parts)


def _visible_response(case: dict[str, Any]) -> str:
    scene = case.get("scene_spec") if isinstance(case.get("scene_spec"), dict) else {}
    return _clean_fragment(
        scene.get("expected_motion_or_pattern"),
        "the displacement field evolves visibly",
    )


def _rendered_field_value(_: dict[str, Any]) -> str:
    return "displacement magnitude on the deformed geometry"


_EVAL_PHYSICS_MECHANISM = {
    "biaxialTension": "in-plane stretching along two directions produces a smooth displacement field that grows toward the pulled edges",
    "bucklingSnapthrough": "beyond the critical load the slender column snaps sideways, trading axial compression for a sudden lateral bow",
    "cantileverLocalization": "the clamped end stays fixed while bending concentrates deformation in a localized band along the beam",
    "contactPatchCompression": "pressure applied over a small patch drives indentation that decays away from the contact zone",
    "cubeCompression3D": "axial compression shortens the cube while its free faces bulge outward through the Poisson effect",
    "cylinderTorsion3D": "the applied twist rotates cross-sections about the axis, so displacement grows with radius and axial position",
    "hyperelasticNeoHookean3D": "the rubber-like material sustains large strains, so displacement grows nonlinearly as the solid deforms",
    "hyperelasticNeoHookeanCuboid3D": "the rubber-like cuboid sustains large strains, so displacement grows nonlinearly as it deforms",
    "hyperelasticNeoHookeanSphere3D": "the rubber-like sphere sustains large strains, so displacement grows nonlinearly as it deforms",
    "loadUnloadHysteresis": "plastic straining during loading is not recovered on unloading, leaving residual deformation behind",
    "longCuboidBending3D": "bending makes displacement grow from the supported region toward the deflected end of the long cuboid",
    "nearIncompressibleBlock": "the nearly incompressible material preserves volume, so shape change is accommodated by lateral bulging",
    "necking": "past the yield point plastic strain localizes at the neck, which thins faster than the rest of the bar",
    "neoHookean": "the rubber-like block sustains large strains, so displacement grows nonlinearly as it deforms",
    "notchedBarTension3D": "the notch concentrates strain, so deformation localizes around the reduced cross-section under tension",
    "plasticJ2CylinderTension3D": "J2 plastic flow lets the cylinder stretch permanently once the yield stress is exceeded",
    "plasticJ2LargeStrain3D": "large-strain J2 plasticity accumulates permanent deformation once loading passes the yield stress",
    "plasticJ2NotchedBar3D": "the notch concentrates stress, so J2 plastic strain accumulates around the reduced section",
    "plateWithHole": "the circular hole concentrates stress, so deformation intensifies near the hole under remote tension",
    "plateWithHole3D": "the circular hole concentrates stress through the thickness, so deformation intensifies near the hole",
    "pureShear": "opposite tangential motion of the faces produces a nearly uniform shear deformation across the solid",
    "sphereIndentation3D": "the indenter presses into the sphere-like body, producing localized contact deformation",
    "threePointBending": "the midspan load bends the beam between the two supports, with the largest deflection at the center",
    "torsionShear": "in-plane twisting shears the solid, so displacement increases away from the rotation center",
    "uniaxialCompression": "axial compression shortens the body while lateral expansion follows the Poisson effect",
    "uniaxialTension": "axial stretching elongates the dogbone, with strain concentrating in the gauge section",
    "yieldHardening": "after yield, isotropic hardening lets stress keep rising while plastic strain accumulates",
}

_EVAL_COLOR_CUE = {
    "biaxialTension": "colors brighten toward the edges that are pulled outward in both directions",
    "bucklingSnapthrough": "the strongest colors appear where the column bows sideways",
    "cantileverLocalization": "colors brighten from the clamped end toward the deflected tip",
    "contactPatchCompression": "the brightest region sits under the compressed patch",
    "cubeCompression3D": "colors intensify on the faces that move the most as the cube squashes",
    "cylinderTorsion3D": "colors grow with radius as outer material sweeps farther during the twist",
    "hyperelasticNeoHookean3D": "the most stretched regions carry the brightest colors",
    "hyperelasticNeoHookeanCuboid3D": "the most stretched regions of the cuboid carry the brightest colors",
    "hyperelasticNeoHookeanSphere3D": "the most stretched regions of the sphere carry the brightest colors",
    "loadUnloadHysteresis": "bright residual colors remain after the load is removed",
    "longCuboidBending3D": "colors peak where the cuboid deflects farthest from its supports",
    "nearIncompressibleBlock": "bulging faces light up while the constrained interior stays dark",
    "necking": "the neck region brightens as it thins faster than the rest of the bar",
    "neoHookean": "the most stretched regions of the block carry the brightest colors",
    "notchedBarTension3D": "colors concentrate around the notch as strain localizes there",
    "plasticJ2CylinderTension3D": "colors grow along the cylinder as permanent stretch accumulates",
    "plasticJ2LargeStrain3D": "heavily strained regions carry the brightest colors",
    "plasticJ2NotchedBar3D": "colors concentrate at the notch where plastic strain localizes",
    "plateWithHole": "colors intensify around the hole where deformation concentrates",
    "plateWithHole3D": "colors intensify around the hole where deformation concentrates",
    "pureShear": "colors form a smooth gradient across the sheared solid",
    "sphereIndentation3D": "the contact zone under the indenter carries the brightest colors",
    "threePointBending": "the midspan carries the brightest colors at maximum deflection",
    "torsionShear": "colors increase away from the rotation center",
    "uniaxialCompression": "colors mark the faces that displace most as the body shortens",
    "uniaxialTension": "colors grow toward the pulled end of the specimen",
    "yieldHardening": "colors accumulate in the plastically stretched gauge region",
}


def _fmt_modulus(value: Any) -> str:
    x = _positive_num(value)
    if x is None:
        return ""
    if x >= 1e9:
        return f"about {_fmt_num(x / 1e9, 4)} GPa"
    return f"about {_fmt_num(x / 1e6, 4)} MPa"


def _fmt_stress(value: Any) -> str:
    x = _positive_num(value)
    if x is None:
        return ""
    if x >= 1e6:
        return f"about {_fmt_num(x / 1e6, 4)} MPa"
    return f"about {_fmt_num(x / 1e3, 4)} kPa"


def _eval_raw_query_text(case: dict[str, Any]) -> str:
    params = case.get("params") if isinstance(case.get("params"), dict) else {}
    scene = case.get("scene_spec") if isinstance(case.get("scene_spec"), dict) else {}
    dim = _plain_token(case.get("dimension") or scene.get("dim"), "solid mechanics")
    problem = _problem_phrase(case)
    response = _visible_response(case)
    geometry = _geometry_summary(scene)
    material = _material_summary(params)
    loading = _loading_summary(params, scene)
    rendered = _rendered_field_value(case)
    text = (
        f"Identify the {dim} solid mechanics benchmark video showing {problem}, where {response}, "
        f"the specimen is {geometry}, the material is {material}, "
        f"the applied boundary condition is {loading}, and the rendered scalar field is {rendered}."
    )
    _assert_query_text_allowed(text)
    return text


def _eval_parsed_query_text(case: dict[str, Any]) -> str:
    family = str(case.get("family") or case.get("scenario") or "").strip()
    params = case.get("params") if isinstance(case.get("params"), dict) else {}
    scene = case.get("scene_spec") if isinstance(case.get("scene_spec"), dict) else {}
    dim = _plain_token(case.get("dimension") or scene.get("dim"), "solid mechanics")
    problem = _problem_phrase(case)
    response = _visible_response(case)
    mechanism = _EVAL_PHYSICS_MECHANISM.get(family, response)
    color_cue = _EVAL_COLOR_CUE.get(
        family, "the largest displacements carry the brightest colors"
    )
    rendered = _rendered_field_value(case)

    geometry_label = _with_indefinite_article(_geometry_label(scene))
    features = _geometry_feature_values(scene)
    variant_parts = [geometry_label]
    if features:
        variant_parts.append(features[0])
    loading_mode = _loading_label(
        params.get("load_mode") or (scene.get("loading") or {}).get("mode")
    )
    sign = _num((scene.get("loading") or {}).get("load_sign"))
    if sign is not None and sign < 0.0 and "compression" not in loading_mode:
        loading_mode = (
            loading_mode.replace("tension", "loading") + " applied in compression"
        )
    load_factor = params.get(
        "load_factor", (scene.get("loading") or {}).get("load_factor")
    )
    loading_phrase = loading_mode
    if load_factor is not None:
        loading_phrase += f" with load factor about {_fmt_num(load_factor, 4)}"

    details: list[str] = []
    e_txt = _fmt_modulus(params.get("E"))
    if e_txt:
        details.append(f"Young's modulus {e_txt}")
    if params.get("nu") is not None:
        details.append(f"Poisson's ratio about {_fmt_num(params.get('nu'), 4)}")
    y_txt = _fmt_stress(params.get("yield_stress"))
    if y_txt:
        details.append(f"yield stress {y_txt}")
    if len(details) < 3 and load_factor is not None:
        details.append(f"load factor about {_fmt_num(load_factor, 4)}")

    sentences = [
        f"Find the {dim} solid mechanics video of {problem}, showing {response}.",
        f"Physically, {mechanism}.",
        f"The color map represents {rendered}, so {color_cue}.",
        f"Use the variant with {_join_natural(variant_parts)}; the loading is {loading_phrase}.",
        f"To identify the exact video, look for {_join_natural(details[:3])}.",
    ]
    text = " ".join(sentences)
    _assert_query_text_allowed(text)
    return text


def _assert_query_text_allowed(text: str) -> None:
    lower = text.lower()
    for term in _FORBIDDEN_QUERY_TERMS:
        if term in lower:
            raise ValueError(f"Forbidden query term {term!r} found in: {text}")
