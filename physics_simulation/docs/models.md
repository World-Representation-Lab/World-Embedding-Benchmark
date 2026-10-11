# Physical formulations and outputs

The branch names and computation methods follow
[Section 2.1](https://arxiv.org/html/2610.03632v1#S2.SS1) and
[Appendix B](https://arxiv.org/html/2610.03632v1#A2) of the paper.

Each example specifies its implementation through `backend` and its physical
parameters through `simulation`. Backend selection is explicit; a missing
dependency stops the run.

## Dynamics

The dynamics engine combines Project Chrono with analytical formulations and
ordinary differential equation (ODE) integration. The `backend` selects an
execution route. Both `chrono` and `surrogate` can call family-specific analytical
implementations. These configuration keys are retained for compatibility; the
family implementation determines the computation method.

Pendulum, block tower, stacking and sphere drop examples use Chrono body dynamics.
Gear motion follows tooth-count kinematics. Rigid collision uses an impulse
calculation. Mass-spring-damper,
unbalanced rotor, articulated chain and rolling-sliding transition use their
family-specific equations. The structural families, including stress-wave
propagation, transient cantilever, harmonic response, impact, plastic beam, damage
bar, rubber compression, flexible crank-slider and shell response, use the included
analytical, ODE, or parameterized-response implementations. Their code is organized
under `world_embedding_sim/engines/dynamics/families`.

For stress-wave propagation, a prescribed triangular pulse travels at
`sqrt(E / density)` and is combined with an opposite-sign reflected pulse. The
video colors show stress magnitude; the exported probe histories retain its sign.

## Fluid Mechanics

OpenFOAM uses finite-volume discretization. Cases specify the application, mesh, material coefficients and initial
and boundary conditions. The velocity and reference length used to derive
viscosity accompany the Reynolds number. Dye examples describe scalar transport
with a prescribed velocity field; scalar diffusivity determines its spreading.

## Solid Mechanics

FEniCSx/DOLFINx uses finite-element discretization to solve the configured
constitutive and boundary-value problem. Videos primarily show displacement
magnitude on the deformed geometry. Quasi-static load sequences represent
successive equilibrium states. Material and geometric parameters must be
interpreted together with the prescribed load or displacement.

The two-dimensional `yieldHardening`, `loadUnloadHysteresis` and `necking`
implementations use a history-dependent effective-stiffness model. A scalar
history field accumulates excess equivalent strain and reduces the effective
shear modulus. The exported `plastic_strain_*` fields describe that internal
scalar history. The load-unload example illustrates the resulting force versus
displacement cycle. The necking example uses a spatially varying specimen and
material fields; its interpretation requires inspecting the computed geometry
and displacement localization. The three-dimensional J2-related families also
use a history-dependent internal-variable update; its coefficients follow their
implementation and loading family.

## Optics & Electromagnetism

The 19 optics families use analytical optical relations or HCIPy propagation.
Mitsuba 3 is an auxiliary backend for selected geometric-optics configurations.
The eight electromagnetic families use Meep's finite-difference time-domain
(FDTD) method.
The Meep length scale and source frequency are specified in the example;
normalized solver units require that scale for conversion to physical units.

Optical videos display irradiance and may progress through a parameter scan.
Meep videos follow simulation time and display the field quantity
`Q = (Re Ex)^2 + (Re Ey)^2 + (Re Ez)^2 + 0.15 (Re Hz)^2`, normalized over the
sampled frames of each case.

## Rendering

All four branches use ParaView for rendering. FFmpeg encodes the
videos. Camera settings, colors, frame rate and video duration describe visualization.
Numerical integration time steps are specified in the solver settings. Use solver outputs to
inspect physical time, load history and field values.
