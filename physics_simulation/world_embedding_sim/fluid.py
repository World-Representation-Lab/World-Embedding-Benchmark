"""OpenFOAM execution using the caller's configured environment."""
import re


def solve_fluid(case, cpus, execute):
    """Mesh and solve an OpenFOAM case, then reconstruct parallel results."""
    control = (case / "system/controlDict").read_text()
    match = re.search(r"(?m)^\s*application\s+(\w+)\s*;", control)
    if not match:
        raise ValueError("OpenFOAM application is missing from controlDict")
    solver = match.group(1)
    execute("blockMesh", ["blockMesh"], cwd=case)
    if (case / "system/setFieldsDict").is_file():
        execute("setFields", ["setFields"], cwd=case)
    if cpus == 1:
        execute("solve", [solver], cwd=case)
        return
    (case / "system/decomposeParDict").write_text(
        "FoamFile {version 2.0; format ascii; class dictionary; object decomposeParDict;}\n"
        f"numberOfSubdomains {cpus};\nmethod scotch;\ndistributed no;\nroots ();\n"
    )
    execute("decompose", ["decomposePar"], cwd=case)
    execute("solve", ["mpirun", "-np", str(cpus), solver, "-parallel"], cwd=case)
    execute("reconstruct", ["reconstructPar"], cwd=case)
