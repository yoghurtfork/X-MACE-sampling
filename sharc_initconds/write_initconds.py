"""Write a SHARC ``initconds`` file from energy/oscillator annotated frames."""

from __future__ import annotations

import argparse


EV_TO_HA = 0.036749322
VEL_CONV = 0.00448998


def run(n_states: int, n_osc: int) -> None:
    import numpy as np
    from ase import units
    from ase.data import atomic_masses
    from ase.io import read

    frames = read("md_traj_with-energies-and-osc.xyz", index=":")
    if len(frames) < 2:
        raise ValueError("Fewer than two MD frames are available.")
    reference_energies = np.asarray(frames[0].info["REF_energy"]).ravel()
    if len(reference_energies) != n_states:
        raise ValueError(f"Energy trajectory contains {len(reference_energies)} states; expected {n_states}")

    with open("initconds", "w", encoding="utf-8") as output:
        output.write("SHARC Initial conditions file, version 4.0   <Excited>\n")
        output.write(f"Ninit     {len(frames) - 1}\n")
        output.write(f"Natom     {len(frames[0])}\nRepr      MCH\n")
        output.write(f"Eref         {reference_energies[0] * EV_TO_HA:.10f}\n")
        output.write("Eharm           0.0000000000\n")
        output.write(f"States    {n_states} 0 0\n\n\nEquilibrium\n")
        for index, frame in enumerate(frames):
            energies = np.asarray(frame.info["REF_energy"]).ravel()
            oscillator_strengths = np.asarray(frame.info["REF_osc-strength"]).ravel()
            if len(energies) != n_states or len(oscillator_strengths) != n_osc:
                raise ValueError("Annotated trajectory does not match the requested state counts.")
            positions = frame.get_positions() / units.Bohr
            velocities = frame.get_velocities() * VEL_CONV
            kinetic = frame.get_kinetic_energy() / units.Hartree
            potential = (reference_energies[0] - energies[0]) * EV_TO_HA
            atom_lines = [
                f"{atom.symbol} {atom.number:.1f} {position[0]:.8f} {position[1]:.8f} {position[2]:.8f} "
                f"{atomic_masses[atom.number]:.8f} {velocity[0]:.8f} {velocity[1]:.8f} {velocity[2]:.8f}\n"
                for atom, position, velocity in zip(frame, positions, velocities)
            ]
            if index == 0:
                output.writelines(atom_lines)
                output.write("\n\n")
                continue
            output.write(f"Index     {index}\nAtoms\n")
            output.writelines(atom_lines)
            output.write("States\n")
            for state in range(n_states):
                energy = float(energies[state]) * EV_TO_HA
                ground = float(energies[0]) * EV_TO_HA
                gap = float(energies[state] - energies[0]) * EV_TO_HA
                oscillator = 0.0 if state == 0 else float(oscillator_strengths[state - 1])
                output.write(
                    f"{state + 1:03}    {energy:.8f} {ground:.8f}   0.00000000   0.00000000   "
                    f"0.00000000   0.00000000   0.00000000   0.00000000   {gap:.8f}   {oscillator:.8f}\n"
                )
            total = potential + kinetic
            output.write(
                f"Ekin    {kinetic:.8f} a.u.\nEpot_harm    0.00000000 a.u.\n"
                f"Epot    {potential:.8f} a.u.\nEtot_harm    {kinetic:.8f} a.u.\n"
                f"Etot    {total:.8f} a.u.\n\n\n"
            )
    print(f"Written {len(frames) - 1} initial conditions + 1 equilibrium geometry to 'initconds'")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-states", type=int, default=3)
    parser.add_argument("--n-osc", type=int, default=2)
    args = parser.parse_args(argv)
    if args.n_states < 2 or args.n_osc != args.n_states - 1:
        parser.error("--n-states must be >= 2 and --n-osc must equal --n-states minus one")
    run(args.n_states, args.n_osc)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
