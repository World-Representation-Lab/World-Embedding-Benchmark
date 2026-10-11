"""Simulation and rendering process execution."""
import json
import os
from pathlib import Path
import subprocess
import sys

from .fluid import solve_fluid
from .config import validate_example
from .queries import write_query


def run_example(config, runtime, output, cpus=2):
    """Generate one case, invoke its solver and renderer, and verify its video."""
    validate_example(config)
    if cpus < 1:
        raise ValueError("cpus must be positive")
    domain, family, backend = (config[k] for k in ("domain", "family", "backend"))
    if domain not in {"fluid", "solid", "dynamics", "optics"}:
        raise ValueError(f"Unsupported domain: {domain}")
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    assets = Path(__file__).parent / "engines" / domain
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env.update(OMP_NUM_THREADS=str(cpus), LP_NUM_THREADS="1", VTK_SMP_MAX_THREADS="1")
    env.update(runtime.get(domain, {}).get("environment", {}))
    render = runtime["render"]
    ffmpeg = render["ffmpeg"]
    env["PATH"] = str(Path(ffmpeg).parent) + os.pathsep + env["PATH"]
    logs = output / "logs"
    logs.mkdir()

    def execute(name, command, extra=None, cwd=None):
        process_env = dict(env)
        process_env.update(extra or {})
        log_path = logs / (name + ".log")
        try:
            with log_path.open("w") as log:
                subprocess.run(
                    list(map(str, command)),
                    cwd=cwd or output,
                    env=process_env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                )
        except (subprocess.CalledProcessError, OSError) as error:
            raise RuntimeError(f"{name} failed. See {log_path}: {error}") from error

    # Select the solver interpreter or the Solid container.
    settings = runtime.get(domain, {})
    if domain == "solid" and settings.get("image"):
        interpreter = settings.get("container_python", "/dolfinx-env/bin/python3")
        prefix = [
            settings.get("container", "apptainer"),
            "exec",
            "--cleanenv",
            "--bind",
            f"{assets}:{assets}",
            "--bind",
            f"{output}:{output}",
            "--bind",
            f"{Path(ffmpeg).parent}:{Path(ffmpeg).parent}",
            settings["image"],
            "env",
            "PATH="
            + str(Path(interpreter).parent)
            + ":/usr/local/bin:/usr/bin:/bin:"
            + str(Path(ffmpeg).parent),
            interpreter,
        ]
    else:
        python = settings.get("python", sys.executable)
        if isinstance(python, dict):
            python = python[backend]
        prefix = [python]

    # Case generators receive a separate copy of the simulation configuration.
    simulation = json.loads(json.dumps(config["simulation"]))
    if domain == "fluid":
        template = Path(simulation["template"])
        if template.is_absolute() or ".." in template.parts:
            raise ValueError("Fluid template must be a bundled relative path")
        simulation["template"] = str(assets / template)
        simulation["out_root"] = str(output / "cases")
    sweep = output / "input.json"
    sweep.write_text(json.dumps(simulation, indent=2) + "\n")
    key = {
        "fluid": "FOAM",
        "solid": "SOLID",
        "optics": "OPTICS",
        "dynamics": "DYNAMICS",
    }[domain]
    execute(
        "prepare",
        [sys.executable, assets / "gen_cases.py", sweep],
        {key + "_CASES_OUT_ROOT": str(output / "cases")},
    )
    cases = list((output / "cases").rglob("case_meta.json"))
    if len(cases) != 1:
        raise RuntimeError(f"Expected one case; generated {len(cases)}")
    case = cases[0].parent
    if domain == "fluid":
        solve_fluid(case, cpus, execute)
    else:
        command = prefix + [assets / "run_case.py", case, "--backend", backend]
        if domain == "dynamics":
            command += [
                "--no-strict-chrono" if backend == "surrogate" else "--strict-chrono"
            ]
        if domain == "solid":
            command += ["--strict-fe"]
        if domain == "optics":
            command += ["--strict-mitsuba"]
        execute("solve", command)

    # Each renderer retains its branch-specific sampling and visual defaults.
    movie = output / "video.mp4"
    rendering_env = dict(render.get("environment", {}))
    if domain in {"fluid", "solid"}:
        pv = render["pvbatch"]
        execute(
            "render",
            [pv]
            + render.get("pvbatch_args", [])
            + [assets / "render_paraview.py", case, movie],
            rendering_env,
        )
        execute(
            "encode",
            [
                ffmpeg,
                "-v",
                "error",
                "-framerate",
                str(rendering_env.get("PV_FPS", env.get("PV_FPS", "30")))
                if domain == "solid"
                else "30",
                "-i",
                output / "video_%04d.png",
                "-c:v",
                "libx264",
                "-crf",
                "18",
                "-pix_fmt",
                "yuv420p",
                "-threads",
                str(cpus),
                movie,
            ],
            rendering_env,
        )
    else:
        frames = output / "frames"
        frames.mkdir()
        display = config.get("render", {})
        fps = int(display.get("fps", 24))
        command = (
            [render["pvpython"]]
            + render.get("pvpython_args", [])
            + [
                assets / "render_paraview.py",
                case,
                frames,
                "--fps",
                str(fps),
                "--width",
                str(display.get("width", 1280)),
                "--height",
                str(display.get("height", 720)),
                "--target-seconds",
                str(display.get("seconds", 15)),
            ]
        )
        execute("render", command, rendering_env)
        if not list(frames.glob("frame_*.png")):
            raise RuntimeError("Renderer produced no frames")
        execute(
            "encode",
            [
                ffmpeg,
                "-v",
                "error",
                "-framerate",
                str(fps),
                "-i",
                frames / "frame_%04d.png",
                "-c:v",
                "libx264",
                "-crf",
                "18",
                "-pix_fmt",
                "yuv420p",
                "-threads",
                str(cpus),
                movie,
            ],
            rendering_env,
        )
    execute(
        "decode", [ffmpeg, "-v", "error", "-xerror", "-i", movie, "-f", "null", "-"]
    )
    if not movie.is_file() or not movie.stat().st_size:
        raise RuntimeError("No video was produced")
    write_query(domain, case, output)
    print(movie)
