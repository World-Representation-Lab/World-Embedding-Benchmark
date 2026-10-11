"""Generate retrieval descriptions using the published evaluation templates."""

import json
from pathlib import Path


def _read_object(path):
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _solid_record(meta):
    from .solid import _case_dim
    from .solid_responses import RESPONSES

    family = meta.get("family") or meta.get("case_prefix")
    scenario = meta.get("scene") or family
    material = meta.get("material", {})
    geometry = meta.get("geometry", {})
    loading = meta.get("loading", {})
    nonlinear = meta.get("nonlinear", {})
    if scenario not in RESPONSES:
        raise ValueError(f"No Solid description registered for {scenario}")
    return {
        "family": family,
        "scenario": scenario,
        "dimension": _case_dim(meta),
        "params": {
            **{
                key: material.get(key)
                for key in ("E", "nu", "yield_stress", "hardening_modulus")
            },
            "material_model": material.get("model", "unknown"),
            "load_mode": loading.get("mode", "unknown"),
            "load_factor": loading.get("load_factor"),
            "nonlinear_model": nonlinear.get("model", "none"),
        },
        "scene_spec": {
            "geometry_id": geometry.get("kind", "unknown"),
            "geometry_params": geometry,
            "loading": loading,
            "dim": _case_dim(meta),
            "model": material.get("model", "unknown"),
            "nonlinear_model": nonlinear.get("model", "none"),
            "expected_motion_or_pattern": RESPONSES[scenario],
        },
    }


def _fluid_record(case, meta):
    from .fluid_fields import _infer_dim, _infer_model, _parse_field_minmax
    from .fluid_utils import extract_u_uniform, parse_foam_scalar_dict

    geometry = meta.get("geometry", {})
    mesh = meta.get("mesh", {})
    transport = parse_foam_scalar_dict(case / "constant/transportProperties")
    velocity = extract_u_uniform(case / "0/U") or meta.get("U_ref") or meta.get("U_lid")
    viscosity = (
        meta.get("nu")
        if isinstance(meta.get("nu"), (int, float))
        else transport.get("nu")
    )
    reynolds = meta.get("Re")
    if (
        reynolds is None
        and isinstance(velocity, (int, float))
        and isinstance(viscosity, (int, float))
        and viscosity
    ):
        reynolds = (
            velocity * (meta.get("L_ref") or geometry.get("lx") or 1.0) / viscosity
        )
    rheology = meta.get("rheology", {})
    return {
        "case_id": case.name,
        "params": {
            "scenario_prefix": meta.get("case_prefix", meta.get("family")),
            "U_ref": velocity,
            "nu": viscosity,
            "rheology_model": rheology.get("model"),
            "rheology_n": rheology.get("n"),
            **{
                key: meta.get(key)
                for key in (
                    "Pe",
                    "pulse_band",
                    "pulse_duration",
                    "pulse_duration_label",
                )
            },
        },
        "scene_spec": {
            "dim": _infer_dim(meta, case),
            "model": _infer_model(case, meta),
            "geometry_params": {
                key: float(value)
                for key, value in {
                    **geometry,
                    **{
                        f"mesh_{k}": v
                        for k, v in mesh.items()
                        if isinstance(v, (int, float))
                    },
                }.items()
                if isinstance(value, (int, float))
            },
            "nondim_numbers": {"Re": reynolds},
        },
        "observables": _parse_field_minmax(case),
    }


def generate_query(domain, case):
    """Read one completed numerical case and return its evaluation-format row.

    The video uses a relative path instead of embedding video bytes in JSON.
    Published templates determine which inputs and computed quantities are stated.
    """
    case = Path(case)
    meta = _read_object(case / "case_meta.json")
    if domain == "fluid":
        from . import fluid

        record = _fluid_record(case, meta)
        raw = fluid._eval_raw_query_text(record)
        parsed = fluid._eval_parsed_query_text(record)
    elif domain == "solid":
        from . import solid

        # Require the solver output even though this template describes inputs.
        _read_object(case / "postProcessing/metrics.json")
        record = _solid_record(meta)
        raw = solid._eval_raw_query_text(record)
        parsed = solid._eval_parsed_query_text(record)
    elif domain == "dynamics":
        from . import dynamics
        from .dynamics_eval import case_from_eval_row
        from .dynamics_parsed import generate_template_text

        metrics = _read_object(case / "postProcessing/metrics.json")
        record = dynamics._case_record(
            case_dir=case, meta=meta, metrics=metrics, metadata_row={}
        )
        tokens = dynamics._case_tokens(record)
        base, extra = dynamics._query_field_pool(tokens)
        raw = dynamics._full_query_text(tokens, tuple(base + extra), case.name)
        parsed = generate_template_text(
            case_from_eval_row(
                record["family"], {"case_id": case.name, "raw_text": raw}
            )
        )
    elif domain == "optics":
        from . import optics

        metrics = _read_object(case / "postProcessing/metrics.json")
        record = optics._case_record(
            case_dir=case, meta=meta, metrics=metrics, metadata_row={}, self_check={}
        )
        tokens = optics._case_tokens(record)
        raw = optics._full_query_text(
            tokens, tuple(optics._query_field_order(tokens)), case.name
        )
        parsed = optics._parsed_query_text(record, tokens)
    else:
        raise ValueError(f"Unknown domain: {domain}")
    if not raw.strip() or not parsed.strip():
        raise ValueError("The query template returned empty text")
    return {
        "query_id": case.name
        + ("__full__00" if domain in {"dynamics", "optics"} else "__full__0"),
        "case_id": case.name,
        "raw_text": raw,
        "parsed_text": parsed,
        "video": {"bytes": None, "path": "video.mp4"},
    }


def write_query(domain, case, output):
    """Write query.json without replacing a different existing description."""
    path = Path(output) / "query.json"
    row = generate_query(domain, case)
    if path.exists():
        if _read_object(path) != row:
            raise FileExistsError(f"A different query already exists: {path}")
        return path
    with path.open("x") as stream:
        stream.write(
            json.dumps(row, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
        )
    return path


def describe_output(domain, output, family=None):
    """Generate text for one existing example output, without running a solver."""
    output = Path(output)
    cases = list((output / "cases").rglob("case_meta.json"))
    if len(cases) != 1:
        raise ValueError(f"Expected one numerical case in {output / 'cases'}")
    if not (output / "video.mp4").is_file():
        raise FileNotFoundError(output / "video.mp4")
    if family is not None:
        meta = _read_object(cases[0])
        if domain == "fluid":
            # The retrieval family groups several lower-level flow configurations.
            from .fluid_families import FAMILIES

            actual = FAMILIES.get(str(meta.get("case_prefix", "")).lower())
        else:
            actual = meta.get("family") or meta.get("case_prefix")
        if actual != family:
            raise ValueError(
                f"Output family {actual!r} differs from requested {family!r}"
            )
    return write_query(domain, cases[0].parent, output)
