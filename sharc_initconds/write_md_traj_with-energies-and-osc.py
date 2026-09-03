"""Add X-MACE oscillator strengths to the energy-annotated MD trajectory."""

from __future__ import annotations

import argparse
from pathlib import Path


def run(model: Path, n_osc: int) -> None:
    import numpy as np
    from ase.io import read, write
    from mace.calculators import MACECalculator

    atoms_list = read("md_traj_with-energies.xyz", index=":")
    calculator = MACECalculator(model_paths=str(model), n_energies=n_osc, device="cuda")
    all_osc = []
    for atoms in atoms_list:
        calculator.calculate(atoms)
        osc_array = np.asarray(calculator.results["energy"]).reshape(-1)
        if len(osc_array) != n_osc:
            raise ValueError(f"Model returned {len(osc_array)} oscillator strengths; expected {n_osc}")
        all_osc.append(osc_array)
        atoms.info["REF_osc-strength"] = osc_array
    np.savetxt("predicted_osc-strengths.txt", np.asarray(all_osc), header="Oscillator strengths: rows=geometries, columns=transitions")
    write("md_traj_with-energies-and-osc.xyz", atoms_list)
    print(f"Written {len(atoms_list)} frames to 'md_traj_with-energies-and-osc.xyz'")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--n-osc", type=int, required=True)
    args = parser.parse_args(argv)
    if not args.model.is_file():
        parser.error(f"--model does not exist: {args.model}")
    if args.n_osc < 1:
        parser.error("--n-osc must be >= 1")
    run(args.model.resolve(), args.n_osc)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
