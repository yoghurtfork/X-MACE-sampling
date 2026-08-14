"""Run a GFN2-xTB relaxation followed by fixed-setting NVE molecular dynamics."""

from __future__ import annotations

import argparse
from pathlib import Path


TEMPERATURE_K = 300
MD_STEPS = 1010
MD_TIMESTEP_FS = 1
SAVE_INTERVAL = 10


def run(input_xyz: Path) -> None:
    from ase import units
    from ase.io import read
    from ase.io.trajectory import Trajectory
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary, ZeroRotation
    from ase.md.verlet import VelocityVerlet
    from ase.optimize import BFGS
    from xtb.ase.calculator import XTB

    atoms = read(input_xyz)
    atoms.calc = XTB(method="GFN2-xTB")
    print(f"Initial energy: {atoms.get_potential_energy():.4f} eV")
    optimizer = BFGS(atoms, logfile="relaxation.log")
    optimizer.run(fmax=0.1)
    print(f"Relaxed energy: {atoms.get_potential_energy():.4f} eV")
    atoms.write("relaxed_mol.xyz")

    MaxwellBoltzmannDistribution(atoms, temperature_K=TEMPERATURE_K)
    Stationary(atoms)
    ZeroRotation(atoms)
    dynamics = VelocityVerlet(atoms, MD_TIMESTEP_FS * units.fs)
    trajectory = Trajectory("md_trajectory.traj", "w", atoms)

    def print_status() -> None:
        print(
            f"Step {dynamics.nsteps:5d}  E_pot={atoms.get_potential_energy():.4f} eV  "
            f"E_kin={atoms.get_kinetic_energy():.4f} eV"
        )

    dynamics.attach(trajectory.write, interval=SAVE_INTERVAL)
    dynamics.attach(print_status, interval=1000)
    dynamics.run(MD_STEPS)
    trajectory.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_xyz", type=Path, help="single-geometry XYZ input file")
    args = parser.parse_args(argv)
    if args.input_xyz.suffix.lower() != ".xyz" or not args.input_xyz.is_file():
        parser.error(f"input_xyz must be an existing XYZ file: {args.input_xyz}")
    run(args.input_xyz.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
