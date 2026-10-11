import contextlib
import io
import json
import runpy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from world_embedding_sim.cli import main
from world_embedding_sim.fluid import solve_fluid
from world_embedding_sim.config import validate_example
import world_embedding_sim


class CommandTests(unittest.TestCase):
    def test_describe_does_not_require_solver_runtime(self):
        output = Path("outputs/pendulum")
        with patch(
            "sys.argv",
            ["web-sim", "describe", "dynamics/pendulum", "--output", str(output)],
        ), patch(
            "world_embedding_sim.cli.describe_output"
        ) as describe, contextlib.redirect_stdout(
            io.StringIO()
        ):
            main()
        describe.assert_called_once_with("dynamics", output, "pendulum")

    def test_root_script_uses_existing_command(self):
        script = Path(__file__).resolve().parents[1] / "run_simulation.py"
        with patch("world_embedding_sim.cli.main") as command:
            runpy.run_path(str(script), run_name="__main__")
        command.assert_called_once_with()

    def test_run_dispatch_preserves_example_and_runtime(self):
        config = (
            Path(world_embedding_sim.__file__).parent
            / "examples/dynamics/pendulum/case.json"
        )
        runtime = {"dynamics": {"python": {"chrono": "/example/bin/python"}}}
        with tempfile.TemporaryDirectory() as directory:
            runtime_path = Path(directory) / "runtime.json"
            runtime_path.write_text(json.dumps(runtime))
            output = Path(directory) / "output"
            arguments = [
                "web-sim",
                "run",
                "dynamics/pendulum",
                "--runtime",
                str(runtime_path),
                "--output",
                str(output),
            ]
            with patch("sys.argv", arguments), patch(
                "world_embedding_sim.cli.run_example"
            ) as run:
                main()
            run.assert_called_once_with(
                json.loads(config.read_text()), runtime, output, 2
            )

    def test_bundled_examples_equal_source(self):
        source = Path(__file__).resolve().parents[1] / "examples"
        bundled = Path(world_embedding_sim.__file__).parent / "examples"
        paths = list(source.glob("*/*/case.json"))
        self.assertEqual(len(paths), 80)
        for path in paths:
            with self.subTest(example=str(path.relative_to(source))):
                self.assertEqual(
                    json.loads(path.read_text()),
                    json.loads((bundled / path.relative_to(source)).read_text()),
                )

    def test_all_example_inputs(self):
        examples = Path(world_embedding_sim.__file__).parent / "examples"
        paths = list(examples.glob("*/*/case.json"))
        self.assertEqual(len(paths), 80)
        for path in paths:
            with self.subTest(example=str(path)):
                validate_example(json.loads(path.read_text()))

    def test_reject_parameter_sweep(self):
        with self.assertRaises(ValueError):
            validate_example(
                {
                    "domain": "fluid",
                    "family": "cavity",
                    "backend": None,
                    "simulation": {"Re_list": [100, 200]},
                }
            )

    def test_invalid_video_dimensions(self):
        with self.assertRaisesRegex(ValueError, "even"):
            validate_example(
                {
                    "domain": "fluid",
                    "family": "cavity",
                    "backend": None,
                    "simulation": {},
                    "render": {"width": 1279},
                }
            )

    def test_nonfinite_input(self):
        with self.assertRaisesRegex(ValueError, "finite"):
            validate_example(
                {
                    "domain": "fluid",
                    "family": "cavity",
                    "backend": None,
                    "simulation": {"Re_list": [float("nan")]},
                }
            )

    def test_list_covers_all_families(self):
        output = io.StringIO()
        with patch("sys.argv", ["web-sim", "list"]), contextlib.redirect_stdout(output):
            main()
        names = output.getvalue().splitlines()
        self.assertEqual(len(names), 80)
        self.assertEqual(len(set(names)), 80)

    def test_unknown_example_fails(self):
        with patch(
            "sys.argv",
            [
                "web-sim",
                "run",
                "missing/family",
                "--runtime",
                "none.json",
                "--output",
                "unused",
            ],
        ), contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as caught:
                main()
        self.assertEqual(caught.exception.code, 2)

    def test_fluid_parallel_commands(self):
        with tempfile.TemporaryDirectory() as directory:
            case = Path(directory)
            (case / "system").mkdir()
            (case / "system/controlDict").write_text("application icoFoam;\n")
            calls = []
            solve_fluid(
                case, 2, lambda name, command, **kw: calls.append((name, command))
            )
            self.assertEqual(
                [x[0] for x in calls],
                ["blockMesh", "decompose", "solve", "reconstruct"],
            )
            self.assertEqual(
                calls[2][1], ["mpirun", "-np", "2", "icoFoam", "-parallel"]
            )
            self.assertIn(
                "numberOfSubdomains 2;", (case / "system/decomposeParDict").read_text()
            )

    def test_fluid_serial_commands(self):
        with tempfile.TemporaryDirectory() as directory:
            case = Path(directory)
            (case / "system").mkdir()
            (case / "system/controlDict").write_text("application icoFoam;\n")
            calls = []
            solve_fluid(case, 1, lambda name, command, **kw: calls.append(name))
            self.assertEqual(calls, ["blockMesh", "solve"])


if __name__ == "__main__":
    unittest.main()
