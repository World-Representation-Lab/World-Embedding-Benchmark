"""Check description structure and numerical input binding without a solver."""

import json
from pathlib import Path
import tempfile
import unittest

import world_embedding_sim
from world_embedding_sim.queries import describe_output, generate_query, write_query


EXAMPLES = Path(world_embedding_sim.__file__).parent / "examples"


class QueryTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.output = Path(temporary.name)

    def make_case(self, name, metrics=None):
        domain, family = name.split("/")
        config = json.loads((EXAMPLES / name / "case.json").read_text())
        simulation = config["simulation"]
        meta = simulation["cases"][0] if "cases" in simulation else simulation
        case = self.output / "cases" / family / meta["case_name"]
        (case / "postProcessing").mkdir(parents=True)
        (case / "case_meta.json").write_text(json.dumps(meta))
        self.write_metrics(case, metrics or {})
        return domain, case

    def write_metrics(self, case, metrics):
        (case / "postProcessing/metrics.json").write_text(json.dumps(metrics))

    def assert_schema(self, row):
        self.assertEqual(
            set(row), {"query_id", "case_id", "raw_text", "parsed_text", "video"}
        )
        self.assertEqual(row["video"], {"bytes": None, "path": "video.mp4"})
        self.assertTrue(row["raw_text"])
        self.assertTrue(row["parsed_text"])
        self.assertNotIn(str(self.output), json.dumps(row))

    def test_dynamics_reads_computed_speed(self):
        domain, case = self.make_case(
            "dynamics/pendulum",
            {"max_speed": 2.125, "max_angle_deg": 35, "settling_time": 6},
        )
        first = generate_query(domain, case)
        self.assert_schema(first)
        self.assertIn("2.125 m/s", first["parsed_text"])
        self.write_metrics(
            case, {"max_speed": 7.875, "max_angle_deg": 35, "settling_time": 6}
        )
        second = generate_query(domain, case)
        self.assertIn("7.875 m/s", second["parsed_text"])
        self.assertNotEqual(first["raw_text"], second["raw_text"])
        self.assertNotEqual(first["parsed_text"], second["parsed_text"])

    def test_solid_reads_material_parameters(self):
        domain, case = self.make_case("solid/plateWithHole3D")
        first = generate_query(domain, case)
        self.assert_schema(first)
        meta = json.loads((case / "case_meta.json").read_text())
        meta["material"]["E"] = 4.2e9
        (case / "case_meta.json").write_text(json.dumps(meta))
        second = generate_query(domain, case)
        self.assertIn("4.2 GPa", second["parsed_text"])
        self.assertNotEqual(first["parsed_text"], second["parsed_text"])

    def test_optics_reads_computed_fringe_spacing(self):
        domain, case = self.make_case(
            "optics/doubleSlitInterference", {"fringe_spacing": 0.1234}
        )
        first = generate_query(domain, case)
        self.assert_schema(first)
        self.assertIn("0.1234", first["parsed_text"])
        self.write_metrics(case, {"fringe_spacing": 0.4321})
        second = generate_query(domain, case)
        self.assertIn("0.4321", second["parsed_text"])
        self.assertNotEqual(first["parsed_text"], second["parsed_text"])

    def test_fluid_reads_openfoam_statistics(self):
        case = self.output / "cases/dye2d"
        stats = case / "postProcessing/fieldMinMax/0/fieldMinMax.dat"
        stats.parent.mkdir(parents=True)
        meta = {
            "case_prefix": "dye2d",
            "geometry": {"lx": 2, "ly": 1},
            "mesh": {"nz": 1},
            "U_ref": 1.0,
            "nu": 0.01,
            "Re": 200,
            "Pe": 100,
        }
        (case / "case_meta.json").write_text(json.dumps(meta))
        stats.write_text("0 U 0 (0 0 0) 4.25 (1 1 0)\n")
        first = generate_query("fluid", case)
        self.assert_schema(first)
        self.assertIn("4.25", first["parsed_text"])
        stats.write_text("1 U 0 (0 0 0) 7.75 (1 1 0)\n")
        second = generate_query("fluid", case)
        self.assertIn("7.75", second["parsed_text"])
        self.assertNotEqual(first["parsed_text"], second["parsed_text"])

    def test_existing_query_is_not_silently_replaced(self):
        domain, case = self.make_case("dynamics/pendulum", {"max_speed": 2.125})
        path = write_query(domain, case, self.output)
        self.assertEqual(write_query(domain, case, self.output), path)
        previous = path.read_bytes()
        self.write_metrics(case, {"max_speed": 7.875})
        with self.assertRaises(FileExistsError):
            write_query(domain, case, self.output)
        self.assertEqual(path.read_bytes(), previous)

    def test_describe_rejects_another_family(self):
        self.make_case("dynamics/pendulum", {"max_speed": 2.125})
        (self.output / "video.mp4").write_bytes(b"video")
        with self.assertRaisesRegex(ValueError, "differs from requested"):
            describe_output("dynamics", self.output, "rigidCollision")

    def test_missing_solver_metrics_fail(self):
        domain, case = self.make_case("dynamics/pendulum")
        (case / "postProcessing/metrics.json").unlink()
        with self.assertRaises(FileNotFoundError):
            generate_query(domain, case)


if __name__ == "__main__":
    unittest.main()
