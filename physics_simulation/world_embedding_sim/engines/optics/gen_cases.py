#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any


WAVE_FAMILIES = {
    "singleSlitFraunhofer",
    "doubleSlitInterference",
    "circularApertureAiry",
    "knifeEdgeFresnel",
    "diffractionGrating",
    "thinFilmInterference",
    "gaussianBeamPropagation",
    "waveplatePolarization",
}
MEEP_FAMILIES = {
    "meepCylinderScattering",
    "meepWaveguideMode",
    "meepPhotonicCrystalDefect",
    "meepPhotonicCrystalWaveguide",
    "meepGratingDiffraction",
    "meepPlasmonicDimer",
    "meepDipoleRadiation",
    "meepEvanescentTIR",
}


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def _iter_sweep_files(items: list[str]) -> list[Path]:
    if not items:
        root = Path(__file__).resolve().parents[1] / "configs"
        return sorted(root.glob("sweep_*.json"))

    out: list[Path] = []
    for raw in items:
        p = Path(raw).expanduser().resolve()
        if p.is_dir():
            out.extend(sorted(p.rglob("sweep_*.json")))
        elif p.is_file():
            out.append(p)
        else:
            raise FileNotFoundError(f"Sweep not found: {raw}")
    return sorted(dict.fromkeys(out))


def _with_defaults(case: dict[str, Any], fallback_prefix: str, idx: int) -> dict[str, Any]:
    family = str(case.get("family") or case.get("case_prefix") or fallback_prefix)
    case_prefix = str(case.get("case_prefix") or family)
    case_name = str(case.get("case_name") or f"{case_prefix}_S{idx:03d}")
    dim = str(case.get("dim", "2D")).strip().upper()
    if dim not in {"2D", "3D"}:
        dim = "2D"

    optics = case.get("optics") if isinstance(case.get("optics"), dict) else {}
    illumination = case.get("illumination") if isinstance(case.get("illumination"), dict) else {}
    receiver = case.get("receiver") if isinstance(case.get("receiver"), dict) else {}
    scan = case.get("scan") if isinstance(case.get("scan"), dict) else {}
    solver = case.get("solver") if isinstance(case.get("solver"), dict) else {}
    visualization = case.get("visualization") if isinstance(case.get("visualization"), dict) else {}

    if family in MEEP_FAMILIES:
        default_backends = ["meep"]
    elif family in WAVE_FAMILIES:
        default_backends = ["hcipy", "surrogate"]
    else:
        default_backends = ["mitsuba3", "mitsuba2", "surrogate"]
    raw_backends = solver.get("backend_preference")
    if isinstance(raw_backends, list):
        prefs = [str(x).strip().lower() for x in raw_backends if str(x).strip()]
        if family not in WAVE_FAMILIES and prefs == ["mitsuba2"]:
            prefs = list(default_backends)
        solver["backend_preference"] = prefs or list(default_backends)
    else:
        solver["backend_preference"] = list(default_backends)
    solver.setdefault("variant_preferences", ["scalar_rgb"])
    solver.setdefault("samples_per_pixel", 96)
    solver.setdefault("reference_frames", 3)
    solver.setdefault("strict_mitsuba_default", True)

    scan.setdefault("parameter", "scan_value")
    scan.setdefault("start", 0.0)
    scan.setdefault("stop", 1.0)
    scan.setdefault("count", 24)
    scan.setdefault("units", "")

    receiver.setdefault("screen_distance", 1.0)
    receiver.setdefault("screen_half_width", 1.2)
    receiver.setdefault("screen_half_height", 0.6)

    illumination.setdefault("source", {})
    source = illumination["source"] if isinstance(illumination.get("source"), dict) else {}
    source.setdefault("type", "collimated_spot")
    source.setdefault("source_distance", 1.0)
    source.setdefault("divergence_deg", 0.4)
    source.setdefault("beam_half_width", 0.08)
    illumination["source"] = source

    visualization.setdefault("field_resolution", [160, 96])
    visualization.setdefault("field_extent", [-1.25, 1.25, -0.6, 0.6])
    visualization.setdefault("ray_bundle_count", 9)

    case["family"] = family
    case["case_prefix"] = case_prefix
    case["case_name"] = case_name
    case["dim"] = dim
    case["optics"] = optics
    case["illumination"] = illumination
    case["receiver"] = receiver
    case["scan"] = scan
    case["solver"] = solver
    case["visualization"] = visualization
    return case


def _normalize_cases(payload: Any, fallback_prefix: str) -> list[dict[str, Any]]:
    if isinstance(payload, dict) and isinstance(payload.get("cases"), list):
        rows = [x for x in payload["cases"] if isinstance(x, dict)]
    elif isinstance(payload, list):
        rows = [x for x in payload if isinstance(x, dict)]
    elif isinstance(payload, dict):
        rows = [payload]
    else:
        rows = []

    out: list[dict[str, Any]] = []
    for i, row in enumerate(rows):
        out.append(_with_defaults(dict(row), fallback_prefix, i))
    return out


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _init_case_dir(case_dir: Path, case: dict[str, Any]) -> None:
    (case_dir / "system").mkdir(parents=True, exist_ok=True)
    (case_dir / "constant").mkdir(parents=True, exist_ok=True)
    (case_dir / "results" / "reference").mkdir(parents=True, exist_ok=True)
    (case_dir / "results" / "sensor_fields").mkdir(parents=True, exist_ok=True)
    (case_dir / "results" / "ray_paths").mkdir(parents=True, exist_ok=True)
    (case_dir / "postProcessing").mkdir(parents=True, exist_ok=True)
    (case_dir / "scene_assets").mkdir(parents=True, exist_ok=True)

    scan = case.get("scan") if isinstance(case.get("scan"), dict) else {}
    _write_text(
        case_dir / "system" / "controlDict",
        "\n".join(
            [
                "application physics_bench_optics;",
                "startTime 0;",
                f"endTime {max(1, int(scan.get('count', 24)) - 1)};",
                "deltaT 1;",
                "",
            ]
        ),
    )
    _write_text(case_dir / "constant" / "README.txt", "Optics case generated by physics_bench_optics.\n")
    payload = json.dumps(case, indent=2, ensure_ascii=False) + "\n"
    (case_dir / "case_meta.json").write_text(payload, encoding="utf-8")
    (case_dir / "solver_input.json").write_text(payload, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate optics benchmark case folders from sweep configs")
    parser.add_argument("sweeps", nargs="*", help="Sweep json files or directories")
    args = parser.parse_args()

    out_root = Path(os.environ.get("OPTICS_CASES_OUT_ROOT", "./outputs/cases")).expanduser().resolve()
    force_regen = os.environ.get("FORCE_REGEN_CASES", "0").strip() not in {"", "0", "false", "False"}
    sweeps = _iter_sweep_files(args.sweeps)
    if not sweeps:
        raise RuntimeError("No sweep files found.")

    created = 0
    skipped = 0
    for sweep in sweeps:
        payload = _read_json(sweep)
        fallback_prefix = sweep.stem.replace("sweep_", "").replace("_s000", "")
        cases = _normalize_cases(payload, fallback_prefix=fallback_prefix)
        for case in cases:
            family = str(case["family"])
            case_name = str(case["case_name"])
            case_dir = out_root / family / case_name
            if case_dir.exists() and not force_regen:
                skipped += 1
                continue
            if case_dir.exists() and force_regen:
                shutil.rmtree(case_dir)
            _init_case_dir(case_dir, case)
            created += 1

    print(
        "[optics_bench] gen_cases "
        f"sweeps={len(sweeps)} created={created} skipped={skipped} "
        f"out_root={out_root} force_regen={int(force_regen)}"
    )


if __name__ == "__main__":
    main()
