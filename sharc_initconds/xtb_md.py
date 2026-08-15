"""Run a GFN2-xTB relaxation followed by fixed-setting NVE molecular dynamics."""

from __future__ import annotations

import argparse
from pathlib import Path


TEMPERATURE_K = 300
MD_STEPS = 1010
MD_TIMESTEP_FS = 1
SAVE_INTERVAL = 10


def run(
    input_xyz: Path,
    *,
    temperature: float = TEMPERATURE_K,
    md_steps: int = MD_STEPS,
    md_timestep_fs: float = MD_TIMESTEP_FS,
    save_interval: int = SAVE_INTERVAL,
) -> None:
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

    MaxwellBoltzmannDistribution(atoms, temperature_K=temperature)
    Stationary(atoms)
    ZeroRotation(atoms)
    dynamics = VelocityVerlet(atoms, md_timestep_fs * units.fs)
    trajectory = Trajectory("md_trajectory.traj", "w", atoms)

    def print_status() -> None:
        print(
            f"Step {dynamics.nsteps:5d}  E_pot={atoms.get_potential_energy():.4f} eV  "
            f"E_kin={atoms.get_kinetic_energy():.4f} eV"
        )

    dynamics.attach(trajectory.write, interval=save_interval)
    dynamics.attach(print_status, interval=1000)
    dynamics.run(md_steps)
    trajectory.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_xyz", type=Path, help="single-geometry XYZ input file")
    parser.add_argument("--temperature", type=float, default=TEMPERATURE_K, help=f"initial temperature in K (default: {TEMPERATURE_K})")
    parser.add_argument("--md-steps", type=int, default=MD_STEPS, help=f"number of MD steps (default: {MD_STEPS})")
    parser.add_argument("--md-timestep-fs", type=float, default=MD_TIMESTEP_FS, help=f"MD timestep in fs (default: {MD_TIMESTEP_FS})")
    parser.add_argument("--save-interval", type=int, default=SAVE_INTERVAL, help=f"trajectory save interval in MD steps (default: {SAVE_INTERVAL})")
    args = parser.parse_args(argv)
    if args.input_xyz.suffix.lower() != ".xyz" or not args.input_xyz.is_file():
        parser.error(f"input_xyz must be an existing XYZ file: {args.input_xyz}")
    if args.temperature <= 0 or args.md_steps < 1 or args.md_timestep_fs <= 0 or args.save_interval < 1:
        parser.error("--temperature, --md-timestep-fs, and --save-interval must be positive; --md-steps must be at least 1")
    run(
        args.input_xyz.resolve(),
        temperature=args.temperature,
        md_steps=args.md_steps,
        md_timestep_fs=args.md_timestep_fs,
        save_interval=args.save_interval,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
