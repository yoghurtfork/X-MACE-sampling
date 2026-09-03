#!/usr/bin/env python3
"""Minimal, local-only replacement for SHARC's setup_traj.py.

It reads selected MCH singlet conditions from an ``initconds.excited`` file and
creates SHARC trajectory folders.  Unlike setup_traj.py it does not copy a QM
directory into every trajectory: each ``TRAJ_* / QM`` is a symlink to one shared
``QM_shared`` directory.  It is intentionally limited to this X-MACE workflow.
"""

from __future__ import annotations

import argparse
import os
import random
import re
import shutil
from pathlib import Path


INPUT_TEMPLATE = """printlevel 2

geomfile "geom"
veloc external
velocfile "veloc"
nstates {nstates}
actstates {nstates}
state {state} mch
coeff auto
rngseed {rngseed}
charge {charge}
ezero    {eref:.10f}
tmax {tmax:.6f}
stepsize {stepsize:.6f}
nsubsteps 25
surf diagonal
coupling ktdc
gradcorrect
ekincorrect parallel_vel
reflect_frustrated none
decoherence_scheme edc
decoherence_param 0.1
hopping_procedure sharc
grad_all
nospinorbit
output_format ascii
output_dat_steps 1
"""


MACE_TEMPLATE_SETTINGS = ("model_file", "device", "energy_unit", "distance_unit")


def render_mace_template(template: Path, settings: dict[str, str]) -> str:
    """Return a base MACE template with this run's settings substituted."""
    lines = template.read_text(encoding="utf-8").splitlines(keepends=True)
    occurrences = {setting: [] for setting in MACE_TEMPLATE_SETTINGS}

    for index, line in enumerate(lines):
        body = line.rstrip("\r\n")
        stripped = body.lstrip()
        if not stripped or stripped.startswith("#"):
            continue
        key = stripped.split(None, 1)[0]
        if key in occurrences:
            occurrences[key].append(index)

    for setting, indices in occurrences.items():
        if not indices:
            raise ValueError(f"MACE template is missing required setting: {setting}")
        if len(indices) > 1:
            raise ValueError(f"MACE template contains duplicate setting: {setting}")

    for setting, indices in occurrences.items():
        index = indices[0]
        line = lines[index]
        body = line.rstrip("\r\n")
        newline = line[len(body):]
        leading = body[: len(body) - len(body.lstrip())]
        suffix_match = re.search(r"\s+#.*$", body)
        suffix = suffix_match.group(0) if suffix_match else ""
        lines[index] = f"{leading}{setting} {settings[setting]}{suffix}{newline}"

    return "".join(lines)


def parse_conditions(path: Path) -> tuple[int, float, str, list[tuple[int, list[str], list[int]]]]:
    """Return atom count, reference energy, state count, and selected blocks."""
    lines = path.read_text(encoding="utf-8").splitlines()
    natom = int(next(line.split()[1] for line in lines if line.startswith("Natom")))
    eref = float(next(line.split()[1] for line in lines if line.startswith("Eref")))
    states = " ".join(next(line.split()[1:] for line in lines if line.startswith("States")))

    blocks: list[tuple[int, list[str], list[int]]] = []
    cursor = 0
    while cursor < len(lines):
        if not lines[cursor].startswith("Index"):
            cursor += 1
            continue
        index = int(lines[cursor].split()[1])
        if cursor + 1 >= len(lines) or lines[cursor + 1].strip() != "Atoms":
            raise ValueError(f"Index {index}: expected an Atoms line")
        atoms = lines[cursor + 2 : cursor + 2 + natom]
        if len(atoms) != natom:
            raise ValueError(f"Index {index}: expected {natom} atom lines")
        state_header = cursor + 2 + natom
        if state_header >= len(lines) or lines[state_header].strip() != "States":
            raise ValueError(f"Index {index}: expected a States line")
        selected: list[int] = []
        pos = state_header + 1
        while pos < len(lines) and not lines[pos].startswith("Ekin"):
            fields = lines[pos].split()
            if len(fields) > 11 and fields[11] == "True":
                selected.append(int(fields[0]))
            pos += 1
        blocks.append((index, atoms, selected))
        cursor = pos
    return natom, eref, states, blocks


def write_trajectory(path: Path, atoms: list[str], state: int, rngseed: int, eref: float,
                     nstates: str, charge: str, tmax: float, stepsize: float, qm_shared: Path) -> None:
    path.mkdir(parents=True)
    geom: list[str] = []
    veloc: list[str] = []
    for line in atoms:
        fields = line.split()
        if len(fields) < 9:
            raise ValueError(f"Malformed atom record: {line}")
        # SHARC's geom file contains symbol, atomic number, Cartesian position,
        # and mass only.  Do not slice by character position: long coordinate
        # fields can otherwise leak a partial velocity into the geometry record.
        # Match SHARC's conventional fixed-width geometry-table layout.
        geom.append(
            "%2s %5.1f %12.8f %12.8f %12.8f %12.8f"
            % (fields[0], *(float(value) for value in fields[1:6]))
        )
        veloc.append("% 12.8f % 12.8f % 12.8f" % tuple(float(x) for x in fields[-3:]))
    (path / "geom").write_text("\n".join(geom) + "\n", encoding="utf-8")
    (path / "veloc").write_text("\n".join(veloc) + "\n", encoding="utf-8")
    (path / "input").write_text(
        INPUT_TEMPLATE.format(nstates=nstates, state=state, rngseed=rngseed,
                              charge=charge, eref=eref, tmax=tmax, stepsize=stepsize),
        encoding="utf-8",
    )
    qm_link = path / "QM"
    qm_link.symlink_to(os.path.relpath(qm_shared, path), target_is_directory=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initconds", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--resources", type=Path, required=True)
    parser.add_argument("--model-file", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--energy-unit", default="eV")
    parser.add_argument("--distance-unit", default="Ang")
    parser.add_argument("--nstates", help="SHARC singlet/doublet/triplet counts; default: file header")
    parser.add_argument("--charge", default="0 0 0")
    parser.add_argument("--tmax", type=float, default=10.0)
    parser.add_argument("--stepsize", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--reset", action="store_true", help="replace an existing output directory")
    args = parser.parse_args()

    if not args.initconds.is_file() or not args.template.is_file() or not args.resources.is_file():
        parser.error("--initconds, --template, and --resources must name existing files")
    if not args.model_file.is_file():
        parser.error("--model-file must name an existing regular file")
    args.model_file = args.model_file.resolve()
    try:
        rendered_mace_template = render_mace_template(
            args.template,
            {
                "model_file": str(args.model_file),
                "device": args.device,
                "energy_unit": args.energy_unit,
                "distance_unit": args.distance_unit,
            },
        )
    except ValueError as error:
        parser.error(str(error))
    if args.output.exists():
        if not args.reset:
            parser.error(f"output already exists: {args.output} (use --reset to replace it)")
        shutil.rmtree(args.output)

    _, eref, header_states, blocks = parse_conditions(args.initconds)
    nstates = args.nstates or header_states
    args.output.mkdir(parents=True)
    qm_shared = args.output / "QM_shared"
    qm_shared.mkdir()
    (qm_shared / "MACE.template").write_text(rendered_mace_template, encoding="utf-8")
    shutil.copy2(args.resources, qm_shared / "MACE.resources")

    rng = random.Random(args.seed)
    counters: dict[int, int] = {}
    created = 0
    for _, atoms, selected_states in blocks:
        for state in selected_states:
            if state < 2:
                continue  # this helper is for initially excited singlet states
            singlet = state - 1  # MCH state 2 (S1) maps to Singlet_1
            counters[singlet] = counters.get(singlet, 0) + 1
            target = args.output / f"Singlet_{singlet}" / f"TRAJ_{counters[singlet]:05d}"
            write_trajectory(target, atoms, state, rng.randint(-999999, 999999) or 1,
                             eref, nstates, args.charge, args.tmax, args.stepsize, qm_shared)
            created += 1
    if not created:
        raise SystemExit("No selected excited singlet states found in initconds.excited")
    print(f"Created {created} local trajectory folder(s) in {args.output}")


if __name__ == "__main__":
    main()
