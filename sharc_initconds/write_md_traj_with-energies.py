"""Annotate selected MD frames with X-MACE multi-state energy predictions."""

from __future__ import annotations

import argparse
from pathlib import Path


def run(model: Path, n_states: int, n_frames: int) -> None:
    import numpy as np
    from ase.io import read, write
    from mace.calculators import MACECalculator

    write("md_trajectory.xyz", read("md_trajectory.traj", index=":"))
    atoms_list = read("md_trajectory.xyz", index=f"0:{n_frames}")
    calculator = MACECalculator(model_paths=str(model), n_energies=n_states, device="cpu")
    all_energies = []
    for atoms in atoms_list:
        calculator.calculate(atoms)
        energy_array = np.asarray(calculator.results["energy"]).reshape(1, -1)
        if energy_array.shape[1] != n_states:
            raise ValueError(f"Model returned {energy_array.shape[1]} states; expected {n_states}")
        all_energies.append(energy_array.ravel())
        atoms.info = {"REF_energy": energy_array}
    np.savetxt("predicted_energies.txt", np.asarray(all_energies), header="Energies (eV): rows=geometries, columns=states")
    write("md_traj_with-energies.xyz", atoms_list)
    print(f"Written {len(atoms_list)} frames to 'md_traj_with-energies.xyz'")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--n-states", type=int, default=3)
    parser.add_argument("--n-frames", type=int, default=101)
    args = parser.parse_args(argv)
    if not args.model.is_file():
        parser.error(f"--model does not exist: {args.model}")
    if args.n_states < 1 or args.n_frames < 2:
        parser.error("--n-states must be >= 1 and --n-frames must be >= 2")
    run(args.model.resolve(), args.n_states, args.n_frames)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
