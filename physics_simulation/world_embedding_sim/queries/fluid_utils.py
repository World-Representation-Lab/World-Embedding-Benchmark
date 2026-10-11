"""Physical descriptions from simulation inputs and computed quantities."""

from __future__ import annotations

import math

import re

from pathlib import Path

from typing import Any, Dict, List, Optional


def _numeric(v: Any) -> Optional[float]:
    try:
        if v is None:
            return None
        return float(v)
    except (ValueError, TypeError):
        return None


def parse_foam_scalar_dict(path: Path) -> Dict[str, float]:
    """Very lightweight OpenFOAM dictionary parser for `key value;` scalars."""
    out: Dict[str, float] = {}
    if not path.exists():
        return out

    text = path.read_text()
    text = re.sub(r"//.*", "", text)
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    for key, raw in re.findall(r"(?m)^\s*([A-Za-z0-9_]+)\s+([^;]+);", text):
        token = raw.strip().split()[-1]
        value = _numeric(token)
        if value is not None:
            out[key] = value
    return out


def extract_u_uniform(path: Path) -> Optional[float]:
    if not path.exists():
        return None
    text = path.read_text()
    m = re.search(
        r"uniform\s*\(\s*([0-9eE+\-.]+)\s+([0-9eE+\-.]+)\s+([0-9eE+\-.]+)\s*\)", text
    )
    if not m:
        return None
    ux, uy, uz = (float(m.group(i)) for i in (1, 2, 3))
    return math.sqrt(ux * ux + uy * uy + uz * uz)


def numeric_time_dirs(path: Path) -> List[Path]:
    if not path.exists():
        return []
    tdirs = []
    for p in path.iterdir():
        if p.is_dir():
            try:
                float(p.name)
                tdirs.append(p)
            except ValueError:
                continue
    return sorted(tdirs, key=lambda x: float(x.name))


def latest_file_under_time_dirs(root: Path, filename: str) -> Optional[Path]:
    for td in reversed(numeric_time_dirs(root)):
        fp = td / filename
        if fp.exists():
            return fp
    return None
