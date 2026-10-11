# Environment setup

Set up the command-line package, install the tools needed by your chosen example,
and record their paths in `runtime.json`. You can start with one branch and add
the others later.

## 1. Install the command-line package

From the repository root:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e .
python run_simulation.py list
```

The command-line package supports Python 3.10 or newer. It generates simulation
inputs using the Python standard library. Solver dependencies belong in separate
environments; `runtime.json` selects the interpreter for each solver.

## 2. Choose a solver environment

Check the `domain` and `backend` fields in the example's `case.json`. The versions
below were used for the example runs. Solver names link to official installation
instructions or release downloads. Use the listed versions when reproducing these
examples; the linked projects may also offer newer releases.

| Branch / backend | Solver version | Python packages used with it |
| --- | --- | --- |
| Fluid Mechanics | [OpenFOAM v2506](https://dl.openfoam.com/source/v2506/) | Uses OpenFOAM executables |
| Solid Mechanics | [FEniCSx/DOLFINx 0.10.0.post2](https://docs.fenicsproject.org/dolfinx/v0.10.0.post2/python/installation.html) | See dependencies below |
| Dynamics / `chrono` | [PyChrono 9.0.1](https://projectchrono.org/download/), Python 3.12 | NumPy 2.4.2 |
| Dynamics / `surrogate` | Analytical / ODE integration | NumPy |
| Optics / `hcipy` | [HCIPy 0.7.0](https://docs.hcipy.org/0.7.0/installation.html) | NumPy 1.26.4, SciPy 1.13.1 |
| Optics / `mitsuba3` | [Mitsuba 3.9.0](https://github.com/mitsuba-renderer/mitsuba3/releases/tag/v3.9.0) | Dr.Jit 1.4.0, NumPy 2.0.2, Pillow |
| Electromagnetism / `meep` | [Python Meep 1.33.0](https://meep.readthedocs.io/en/latest/Installation/) | NumPy 2.2.6 |

Optics and electromagnetism share the `optics` configuration key. The `surrogate`
key names an existing Dynamics execution route; computation methods are described
in [physical formulations and outputs](models.md).

PyChrono must include the modules required by the selected example. The Solid
environment needs DOLFINx, UFL, Basix, PETSc/petsc4py, MPI/mpi4py, and NumPy.
Installation options are described in the
[DOLFINx documentation](https://docs.fenicsproject.org/dolfinx/v0.10.0.post2/python/installation.html).
For PyChrono, follow the [9.x installation guide](https://api.projectchrono.org/9.0.0/pychrono_installation.html).
The OpenFOAM link points to the OpenCFD v2506 release used by these examples.

For video generation, install:

- [FFmpeg downloads](https://ffmpeg.org/download.html), with the `libx264` encoder
  for every branch. The official page provides source releases and links to
  third-party prebuilt binaries.
- [ParaView downloads](https://www.paraview.org/download/): select **6.0.1** for
  all four branches. Dynamics and Optics use `pvpython`; Fluid and Solid use
  `pvbatch`.

## 3. Set executable paths

For a Dynamics pendulum example, save the following as `runtime.json` and replace
the placeholder paths with absolute paths to your installations:

```json
{
  "dynamics": {
    "python": {
      "chrono": "/path/to/chrono/bin/python"
    }
  },
  "render": {
    "pvpython": "/path/to/paraview/bin/pvpython",
    "ffmpeg": "/path/to/ffmpeg"
  }
}
```

The [README runtime configuration](../README.md#runtime-configuration) includes
paths for all Python backends. Each run only needs the selected solver and its
rendering tools. The following settings cover branch-specific requirements.

### Fluid Mechanics: initialize OpenFOAM

Source the initialization script supplied by your OpenFOAM installation, then
check the executable in the same shell:

```bash
blockMesh -help
```

OpenFOAM uses its own compiler and MPI libraries. Keep these consistent with the
installed binaries. `--cpus 1` runs the solver directly; larger values use
`mpirun`, domain decomposition, and reconstruction. The default is `--cpus 2`.

In `runtime.json`, set `render.pvbatch` to the ParaView `pvbatch` executable and
`render.ffmpeg` to FFmpeg. Fluid examples find OpenFOAM commands through the
initialized shell environment.

### Solid Mechanics: use a Python environment or a container

For a native installation, set `solid.python` to the interpreter that has DOLFINx
installed. Set `render.pvbatch` to ParaView and `render.ffmpeg` to FFmpeg.

Official container images are published at
[dolfinx/dolfinx](https://hub.docker.com/r/dolfinx/dolfinx), with available versions
listed under [image tags](https://hub.docker.com/r/dolfinx/dolfinx/tags).
The [project's container instructions](https://github.com/FEniCS/dolfinx#docker-images)
describe the `stable` and `nightly` images. These tags change over time; retain
the selected image's digest and check the DOLFINx version inside it. The tested
version for these examples is **0.10.0.post2**.

Install the container runtime using the
[Apptainer installation guide](https://apptainer.org/docs/admin/main/installation.html).
To use an existing SIF image, configure:

```json
{
  "solid": {
    "container": "apptainer",
    "image": "/path/to/dolfinx.sif",
    "container_python": "/path/inside/image/bin/python"
  },
  "render": {
    "ffmpeg": "/path/to/ffmpeg"
  }
}
```

`container_python` is the interpreter path inside the image. The runner mounts
the engine, output directory, and FFmpeg directory into the container. FFmpeg
must work both on the host and inside the image; a compatible static build is
useful here. The example runs used a DOLFINx 0.10.0.post2 image.

### Optional: headless rendering and environment variables

Choose a ParaView build that supports your machine's graphics configuration.
Additional launcher arguments go in `render.pvpython_args` or
`render.pvbatch_args`, as JSON arrays of strings.

Use a domain's `environment` object for installation-specific variables and
`render.environment` for ParaView rendering settings. Solid rendering runs outside
the solver container, using `render.pvbatch`. Keep machine-specific library paths
in this configuration file.

### Rendering settings by branch

The renderers use the following settings. The example's `render` object supplies
command-line options to the Dynamics and Optics renderers.

| Branch | Renderer | Resolution and frame rate |
| --- | --- | --- |
| Dynamics | ParaView | Example `render.width`, `render.height`, and `render.fps` |
| Optics & Electromagnetism | ParaView | Example `render.width`, `render.height`, and `render.fps` |
| Fluid Mechanics | ParaView | `PV_WIDTH` and `PV_HEIGHT`; defaults to 1920 × 1080, encoded at 30 fps |
| Solid Mechanics | ParaView | `PV_WIDTH`, `PV_HEIGHT`, and `PV_FPS`; defaults to 1920 × 1080 at 30 fps |

Fluid and Solid use renderer-specific timing based on the available simulation
or loading states. Their renderers do not read the example's `render` object.
Keep Fluid's `PV_FPS` at 30 to agree with its encoder.

For the included examples, start with the existing defaults. Changing playback
duration changes the relationship between video time and simulation time;
physical interpretation should use the exported time or loading history.

## 4. Run an example

With the pendulum runtime configuration above:

```bash
python run_simulation.py run dynamics/pendulum \
  --runtime runtime.json \
  --output outputs/pendulum
```

Use a new output directory for every run. A successful run produces `video.mp4`
and decodes the complete video to check for encoding errors.

If a command fails, inspect the log named in the error message. Stage logs are
under `logs/`; numerical cases are under `cases/`. Keep the failed output for
inspection and choose a new directory when repeating the run.
