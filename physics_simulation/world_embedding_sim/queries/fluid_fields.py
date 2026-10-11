"""Physical descriptions from simulation inputs and computed quantities."""

from __future__ import annotations

import re

from pathlib import Path

from typing import Any, Dict, Optional

from .fluid_utils import latest_file_under_time_dirs


def _infer_dim(meta: Dict[str, Any], case_dir: Path) -> str:
    nz = (
        ((meta.get("mesh") or {}).get("nz"))
        if isinstance(meta.get("mesh"), dict)
        else None
    )
    if isinstance(nz, int) and nz <= 1:
        return "2D"
    name = case_dir.name.lower()
    if "2d" in name:
        return "2D"
    if "3d" in name:
        return "3D"
    return "3D" if (case_dir / "constant" / "polyMesh").exists() else "2D"


def _infer_model(case_dir: Path, meta: Dict[str, Any]) -> str:
    if "turb" in case_dir.name.lower():
        return "RANS"
    tprop = case_dir / "constant" / "turbulenceProperties"
    if not tprop.exists():
        return "laminar"
    text = tprop.read_text().lower()
    if "simulationtype" in text and "laminar" in text:
        return "laminar"
    if "les" in text:
        return "LES"
    if "ras" in text or "komega" in text or "kepsilon" in text:
        return "RANS"
    return "laminar"


def _parse_field_minmax(case_dir: Path) -> Dict[str, Optional[float]]:
    root = case_dir / "postProcessing" / "fieldMinMax"
    dat = latest_file_under_time_dirs(root, "fieldMinMax.dat")
    out = {"U_min": None, "U_max": None, "p_min": None, "p_max": None}
    if not dat:
        return out

    def location_end(tokens: list[str], start: int) -> int | None:
        balance = 0
        for i in range(start, len(tokens)):
            balance += tokens[i].count("(") - tokens[i].count(")")
            if ")" in tokens[i] and balance <= 0:
                return i + 1
        return None

    for line in dat.read_text().splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        tokens = s.split()
        if len(tokens) < 5:
            continue
        field = tokens[1]
        try:
            fmin = float(tokens[2])
        except ValueError:
            continue
        max_idx = location_end(tokens, 3)
        if max_idx is None or max_idx >= len(tokens):
            continue
        # Parallel OpenFOAM can insert a processor id between the min location
        # and the max value.  Skip it when present.
        if re.fullmatch(r"\d+", tokens[max_idx]) and max_idx + 1 < len(tokens):
            max_idx += 1
        try:
            fmax = float(tokens[max_idx])
        except ValueError:
            continue
        if field in {"mag(U)", "U"}:
            out["U_min"] = fmin
            out["U_max"] = fmax
        if field == "p":
            out["p_min"] = fmin
            out["p_max"] = fmax
    return out
