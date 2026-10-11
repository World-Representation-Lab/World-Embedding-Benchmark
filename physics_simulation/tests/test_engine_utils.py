"""Tests for engine utilities that require no external solver."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

import world_embedding_sim


ENGINES = Path(world_embedding_sim.__file__).parent / "engines"


def load_utility(branch, filename):
    path = ENGINES / branch / filename
    spec = importlib.util.spec_from_file_location(f"{branch}_{path.stem}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class EngineUtilityTests(unittest.TestCase):
    def test_engine_imports(self):
        modules = {
            "dynamics": ["run_case"],
            "solid": [
                "run_case",
                "fe_fenicsx_solver",
                "fe_fenicsx_solver_2d",
                "fe_fenicsx_solver_2d_general",
            ],
            "optics": ["run_case", "run_em_case", "run_meep_case"],
        }
        for branch, names in modules.items():
            with self.subTest(branch=branch):
                result = subprocess.run(
                    [sys.executable, "-c", "import " + ", ".join(names)],
                    cwd=ENGINES / branch,
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(result.returncode, 0, result.stderr)

    def test_loading_history(self):
        solid = load_utility("solid", "fe_utils.py")
        self.assertEqual(solid._resample_path([], 1), [1.0])
        self.assertEqual(solid._resample_path([], 3), [0.0, 0.5, 1.0])
        self.assertEqual(solid._resample_path([0.0, 1.0], 3), [0.0, 0.5, 1.0])
        case = {"loading": {"load_path": [-1.0, 0.5, 2.0]}}
        self.assertEqual(solid._load_history(case, 3), [0.0, 0.5, 1.0])

    def test_principal_values(self):
        solid = load_utility("solid", "fe_utils.py")
        high, low = solid._principal_2d(
            np.array([3.0, 0.0]), np.array([1.0, 0.0]), np.array([0.0, 2.0])
        )
        np.testing.assert_array_equal(high, [3.0, 2.0])
        np.testing.assert_array_equal(low, [1.0, -2.0])

    def test_triangle_field_output(self):
        solid = load_utility("solid", "fe_utils.py")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "field.vtu"
            solid._write_vtu_tri(
                path,
                np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float),
                np.array([[0, 1, 2]], dtype=int),
                {"displacement": np.array([0.0, 0.5, 1.0])},
            )
            piece = ET.parse(path).find("./UnstructuredGrid/Piece")
            self.assertEqual(
                piece.attrib, {"NumberOfPoints": "3", "NumberOfCells": "1"}
            )
            values = piece.find("./PointData/DataArray").text.split()
            self.assertEqual(values, ["0", "0.5", "1"])
            series = Path(directory) / "fields.pvd"
            solid._write_pvd(series, [0.0, 0.5], ["first.vtu", "second.vtu"])
            entries = ET.parse(series).findall("./Collection/DataSet")
            self.assertEqual([row.attrib["timestep"] for row in entries], ["0", "0.5"])
            self.assertEqual(
                [row.attrib["file"] for row in entries], ["first.vtu", "second.vtu"]
            )

    def test_optical_field_statistics(self):
        optics = load_utility("optics", "field_utils.py")
        zero = optics._summary(np.zeros((3, 3)), [-1.0, 1.0, -1.0, 1.0])
        self.assertEqual(
            zero,
            {
                "peak": 0.0,
                "power": 0.0,
                "centroid_x": 0.0,
                "centroid_y": 0.0,
                "spot_radius": 0.0,
            },
        )
        values = np.zeros((3, 3))
        values[1, 1] = 2.0
        point = optics._summary(values, [-1.0, 1.0, -1.0, 1.0])
        self.assertEqual(point, dict(zero, peak=2.0, power=2.0))
        self.assertEqual(optics._clamp(-1.0, 0.0, 1.0), 0.0)
        self.assertEqual(optics._clamp(2.0, 0.0, 1.0), 1.0)

    def test_trajectory_speed(self):
        dynamics = load_utility("dynamics", "trajectory.py")
        times = np.array([0.0, 0.5, 1.0])
        positions = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=float)
        velocity = dynamics._compute_speed_series(times, positions)
        np.testing.assert_array_equal(velocity, [[2, 0, 0], [2, 0, 0], [2, 0, 0]])


if __name__ == "__main__":
    unittest.main()
