"""Check process commands without starting external solvers or renderers."""

from __future__ import annotations

import contextlib
import copy
import io
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import world_embedding_sim
from world_embedding_sim.runner import run_example


EXAMPLES = Path(world_embedding_sim.__file__).parent / "examples"


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.output = Path(self.directory.name) / "output"
        self.calls = []
        query_patch = patch("world_embedding_sim.runner.write_query")
        self.write_query = query_patch.start()
        self.addCleanup(query_patch.stop)
        self.runtime = {
            "dynamics": {
                "python": {
                    "chrono": "/tools/chrono/python",
                    "surrogate": "/tools/numpy/python",
                }
            },
            "solid": {"python": "/tools/dolfinx/python"},
            "optics": {
                "python": {
                    "hcipy": "/tools/hcipy/python",
                    "mitsuba3": "/tools/mitsuba/python",
                    "meep": "/tools/meep/python",
                }
            },
            "render": {
                "pvpython": "/tools/paraview/pvpython",
                "pvbatch": "/tools/paraview/pvbatch",
                "ffmpeg": "/tools/ffmpeg/ffmpeg",
            },
        }

    def _process(self, command, **kwargs):
        """Create only the files the runner expects from each process."""
        stage = Path(kwargs["stdout"].name).stem
        self.calls.append((stage, command, kwargs))
        if stage == "prepare":
            case = self.output / "cases" / "example"
            (case / "system").mkdir(parents=True)
            (case / "case_meta.json").write_text("{}")
            (case / "system/controlDict").write_text("application icoFoam;\n")
        if stage == "render":
            frames = self.output / "frames"
            if frames.is_dir():
                (frames / "frame_0000.png").write_bytes(b"test frame")
            (self.output / "video.mp4").write_bytes(b"test video")

    def _run(self, config, process=None):
        with patch(
            "world_embedding_sim.runner.subprocess.run",
            side_effect=process or self._process,
        ), contextlib.redirect_stdout(io.StringIO()):
            run_example(config, self.runtime, self.output)

    def _config(self, example):
        return json.loads((EXAMPLES / example / "case.json").read_text())

    def test_all_examples_keep_solver_and_render_commands(self):
        paths = sorted(EXAMPLES.glob("*/*/case.json"))
        self.assertEqual(len(paths), 80)
        for index, path in enumerate(paths):
            with self.subTest(example=str(path.relative_to(EXAMPLES))):
                self.output = Path(self.directory.name) / str(index)
                self.calls = []
                config = json.loads(path.read_text())
                original = copy.deepcopy(config)
                self._run(config)
                self.assertEqual(config, original)
                stages = [stage for stage, _, _ in self.calls]
                commands = {stage: command for stage, command, _ in self.calls}
                self.assertEqual(stages[0], "prepare")
                self.assertEqual(stages[-1], "decode")
                self.write_query.assert_called_with(
                    config["domain"],
                    self.output / "cases/example",
                    self.output.resolve(),
                )
                self.assertEqual(
                    commands["decode"][0], self.runtime["render"]["ffmpeg"]
                )
                self.assertIn("-xerror", commands["decode"])
                domain = config["domain"]
                if domain == "fluid":
                    self.assertEqual(
                        stages,
                        [
                            "prepare",
                            "blockMesh",
                            "decompose",
                            "solve",
                            "reconstruct",
                            "render",
                            "encode",
                            "decode",
                        ],
                    )
                    self.assertEqual(
                        commands["render"][0], self.runtime["render"]["pvbatch"]
                    )
                    self.assertEqual(
                        commands["solve"],
                        ["mpirun", "-np", "2", "icoFoam", "-parallel"],
                    )
                    self.assertEqual(commands["encode"][3:5], ["-framerate", "30"])
                else:
                    solver = self.runtime[domain]["python"]
                    if isinstance(solver, dict):
                        solver = solver[config["backend"]]
                    self.assertEqual(commands["solve"][0], solver)
                    self.assertIn(config["backend"], commands["solve"])
                    if domain == "solid":
                        self.assertEqual(
                            stages, ["prepare", "solve", "render", "encode", "decode"]
                        )
                        self.assertIn("--strict-fe", commands["solve"])
                        self.assertEqual(
                            commands["render"][0], self.runtime["render"]["pvbatch"]
                        )
                        self.assertEqual(
                            Path(commands["render"][1]).name, "render_paraview.py"
                        )
                        self.assertEqual(commands["encode"][3:5], ["-framerate", "30"])
                    else:
                        self.assertEqual(
                            stages, ["prepare", "solve", "render", "encode", "decode"]
                        )
                        display = config["render"]
                        self.assertEqual(
                            commands["render"][0], self.runtime["render"]["pvpython"]
                        )
                        self.assertEqual(
                            commands["render"][4:],
                            [
                                "--fps",
                                str(display["fps"]),
                                "--width",
                                str(display["width"]),
                                "--height",
                                str(display["height"]),
                                "--target-seconds",
                                str(display["seconds"]),
                            ],
                        )
                        self.assertEqual(
                            commands["encode"][3:5], ["-framerate", str(display["fps"])]
                        )

    def test_solid_container_invocation(self):
        self.runtime["solid"] = {
            "container": "/tools/apptainer",
            "image": "/images/dolfinx.sif",
            "container_python": "/dolfinx-env/bin/python3",
        }
        self._run(self._config("solid/uniaxialTension"))
        commands = {stage: command for stage, command, _ in self.calls}
        command = commands["solve"]
        self.assertEqual(command[:3], ["/tools/apptainer", "exec", "--cleanenv"])
        self.assertIn("/images/dolfinx.sif", command)
        self.assertIn("/dolfinx-env/bin/python3", command)
        self.assertIn(f"{self.output}:{self.output}", command)
        self.assertEqual(commands["render"][0], "/tools/paraview/pvbatch")
        self.assertNotIn("/images/dolfinx.sif", commands["render"])

    def test_solid_render_environment_and_frame_rate(self):
        self.runtime["render"]["environment"] = {"PV_FPS": "24", "PV_WIDTH": "1280"}
        self.runtime["render"]["pvbatch_args"] = ["--force-offscreen-rendering"]
        self._run(self._config("solid/plateWithHole3D"))
        commands = {stage: command for stage, command, _ in self.calls}
        self.assertEqual(commands["render"][1], "--force-offscreen-rendering")
        self.assertEqual(commands["encode"][3:5], ["-framerate", "24"])
        for stage, _, kwargs in self.calls:
            if stage in {"render", "encode"}:
                self.assertEqual(kwargs["env"]["PV_FPS"], "24")
                self.assertEqual(kwargs["env"]["PV_WIDTH"], "1280")

    def test_render_failure_reports_log_and_stops(self):
        def fail_render(command, **kwargs):
            self._process(command, **kwargs)
            if Path(kwargs["stdout"].name).stem == "render":
                raise subprocess.CalledProcessError(1, command)

        with self.assertRaisesRegex(RuntimeError, r"render failed.*logs/render.log"):
            self._run(self._config("dynamics/pendulum"), fail_render)
        self.assertEqual(self.calls[-1][0], "render")

    def test_missing_frames_stops_encoding(self):
        def omit_frames(command, **kwargs):
            if Path(kwargs["stdout"].name).stem != "render":
                self._process(command, **kwargs)

        with self.assertRaisesRegex(RuntimeError, "Renderer produced no frames"):
            self._run(self._config("dynamics/pendulum"), omit_frames)
        self.assertNotIn("encode", [stage for stage, _, _ in self.calls])

    def test_existing_output_is_preserved(self):
        self.output.mkdir()
        marker = self.output / "keep.txt"
        marker.write_text("existing output")
        with self.assertRaises(FileExistsError):
            self._run(self._config("dynamics/pendulum"))
        self.assertEqual(marker.read_text(), "existing output")
        self.assertEqual(self.calls, [])


if __name__ == "__main__":
    unittest.main()
