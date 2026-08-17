"""Write a SHARC ``initconds`` file from energy/oscillator annotated frames."""

from __future__ import annotations

import argparse
import math


VEL_CONV = 0.00448998


def effective_transition_dipole(oscillator_strength: float, gap_ha: float) -> tuple[float, bool]:
    """Return an x-directed effective dipole (a.u.) for SHARC selection.

    SHARC's ``excite.py`` derives oscillator strengths from transition dipoles,
    so an X-MACE oscillator-strength prediction must be represented by an
    equivalent dipole magnitude. Negative predictions are treated as dark
    transitions; non-finite values and non-positive excitation gaps are not
    physically usable and therefore fail early.
    """
    if not math.isfinite(oscillator_strength):
        raise ValueError(f"Oscillator strength must be finite; got {oscillator_strength!r}")
    if not math.isfinite(gap_ha):
        raise ValueError(f"Excitation gap must be finite; got {gap_ha!r}")
    if gap_ha <= 0.0:
        raise ValueError(f"Excitation gap must be positive; got {gap_ha:.8g} Ha")
    if oscillator_strength <= 0.0:
        return 0.0, oscillator_strength < 0.0
    return math.sqrt(3.0 * oscillator_strength / (2.0 * gap_ha)), False


def run(n_states: int, n_osc: int, *, without_oscillator_strengths: bool = False) -> None:
    import numpy as np
    from ase import units
    from ase.data import atomic_masses
    from ase.io import read

    input_filename = (
        "md_traj_with-energies.xyz"
        if without_oscillator_strengths
        else "md_traj_with-energies-and-osc.xyz"
    )
    frames = read(input_filename, index=":")
    if len(frames) < 2:
        raise ValueError("Fewer than two MD frames are available.")
    reference_energies = np.asarray(frames[0].info["REF_energy"]).ravel()
    if len(reference_energies) != n_states:
        raise ValueError(f"Energy trajectory contains {len(reference_energies)} states; expected {n_states}")
    if not np.isfinite(reference_energies).all():
        raise ValueError("Equilibrium-frame predicted state energies must all be finite.")

    clamped_oscillators = 0
    with open("initconds", "w", encoding="utf-8") as output:
        output.write("SHARC Initial conditions file, version 4.0   <Excited>\n")
        output.write(f"Ninit     {len(frames) - 1}\n")
        output.write(f"Natom     {len(frames[0])}\nRepr      MCH\n")
        output.write(f"Eref         {reference_energies[0]:.10f}\n")
        output.write("Eharm           0.0000000000\n")
        output.write(f"States    {n_states} 0 0\n\n\nEquilibrium\n")
        for index, frame in enumerate(frames):
            energies = np.asarray(frame.info["REF_energy"]).ravel()
            oscillator_strengths = (
                np.zeros(n_states - 1)
                if without_oscillator_strengths
                else np.asarray(frame.info["REF_osc-strength"]).ravel()
            )
            if len(energies) != n_states or (
                not without_oscillator_strengths and len(oscillator_strengths) != n_osc
            ):
                raise ValueError("Annotated trajectory does not match the requested state counts.")
            if not np.isfinite(energies).all():
                raise ValueError(f"Frame {index}: predicted state energies must all be finite.")
            if not np.isfinite(oscillator_strengths).all():
                raise ValueError(f"Frame {index}: predicted oscillator strengths must all be finite.")
            positions = frame.get_positions() / units.Bohr
            velocities = frame.get_velocities() * VEL_CONV
            kinetic = frame.get_kinetic_energy() / units.Hartree
            potential = (reference_energies[0] - energies[0]) # * EV_TO_HA
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
                energy = float(energies[state]) # * EV_TO_HA
                ground = float(energies[0]) # * EV_TO_HA
                gap = float(energies[state] - energies[0]) # * EV_TO_HA
                oscillator = 0.0 if state == 0 else float(oscillator_strengths[state - 1])
                dipole_x = 0.0
                if state > 0 and not without_oscillator_strengths:
                    try:
                        dipole_x, clamped = effective_transition_dipole(oscillator, gap)
                    except ValueError as error:
                        raise ValueError(f"Frame {index}, state {state + 1}: {error}") from error
                    clamped_oscillators += clamped
                    oscillator = max(oscillator, 0.0)
                output.write(
                    f"{state + 1:03}    {energy:.8f} {ground:.8f}   {dipole_x:.8f}   0.00000000   "
                    f"0.00000000   0.00000000   0.00000000   0.00000000   {gap:.8f}   {oscillator:.8f}\n"
                )
            total = potential + kinetic
            output.write(
                f"Ekin    {kinetic:.8f} a.u.\nEpot_harm    0.00000000 a.u.\n"
                f"Epot    {potential:.8f} a.u.\nEtot_harm    {kinetic:.8f} a.u.\n"
                f"Etot    {total:.8f} a.u.\n\n\n"
            )
    print(f"Written {len(frames) - 1} initial conditions + 1 equilibrium geometry to 'initconds'")
    if without_oscillator_strengths:
        print("Wrote zero transition dipoles and oscillator strengths for explicit-state selection")
    else:
        print(f"Clamped {clamped_oscillators} negative oscillator-strength predictions to zero")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-states", type=int, default=3)
    parser.add_argument("--n-osc", type=int, default=2)
    parser.add_argument(
        "--without-oscillator-strengths",
        action="store_true",
        help="read energy-only frames and write zero transition dipoles",
    )
    args = parser.parse_args(argv)
    if args.n_states < 2 or (
        not args.without_oscillator_strengths and args.n_osc != args.n_states - 1
    ):
        parser.error("--n-states must be >= 2 and --n-osc must equal --n-states minus one")
    run(
        args.n_states,
        args.n_osc,
        without_oscillator_strengths=args.without_oscillator_strengths,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
