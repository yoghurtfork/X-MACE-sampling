from ase.io import read
from ase.data import atomic_masses
from ase import units
import numpy as np

N_STATES    = 3       # number of electronic states to write (S0, S1, S2)
STATES_LINE = "5 0 0" # singlets doublets triplets declared in the file header
                       # must be consistent with what excite.py expects
EV_TO_HA    = 0.036749322
VEL_CONV    = 0.00448998   # ASE velocity units → Bohr / a.u. time

frames        = read("md_traj_with-energies-and-osc.xyz", index=":")
ref_energies0 = np.array(frames[0].info["REF_energy"]).flatten()   # eV, frame 0
Eref_ha       = ref_energies0[0] * EV_TO_HA                         # S0 of frame 0, Hartree

with open("initconds", "w") as f:

    # ── File header ───────────────────────────────────────────────────────
    f.write("SHARC Initial conditions file, version 4.0   <Excited>\n")
    f.write(f"Ninit     {len(frames) - 1}\n")   # frame 0 = equilibrium, not an IC
    f.write(f"Nato     {frames[0].get_number_of_atoms()}\n")
    f.write("Repr      MCH\n")
    f.write(f"Eref         {Eref_ha:.10f}\n")
    f.write("Eharm           0.0000000000\n")
    f.write(f"States    {STATES_LINE}\n")
    f.write("\n\nEquilibrium\n")

    for i, frame in enumerate(frames):
        ref_energies  = np.array(frame.info["REF_energy"]).flatten()        # (N_STATES,) eV
        osc_strengths = np.array(frame.info["REF_osc-strength"]).flatten()  # (N_OSC,)

        pos_bohr = frame.get_positions() / units.Bohr
        vel_au   = frame.get_velocities() * VEL_CONV
        ekin_ha  = frame.get_kinetic_energy() / units.Hartree
        epot_ha  = (ref_energies0[0] - ref_energies[0]) * EV_TO_HA
        etot_ha  = epot_ha + ekin_ha

        atom_lines = [
            f"{atom.symbol} {atom.number:.1f} "
            f"{pos[0]:.8f} {pos[1]:.8f} {pos[2]:.8f} "
            f"{atomic_masses[atom.number]:.8f} "
            f"{vel[0]:.8f} {vel[1]:.8f} {vel[2]:.8f}\n"
            for atom, pos, vel in zip(frame, pos_bohr, vel_au)
        ]

        if i == 0:
            # Equilibrium geometry block — no Index / States header
            f.writelines(atom_lines)
            f.write("\n\n")
        else:
            f.write(f"Index     {i}\n")
            f.write("Atoms\n")
            f.writelines(atom_lines)
            f.write("States\n")
            for j in range(N_STATES):
                e_ha     = float(ref_energies[j])                    * EV_TO_HA
                ref_e_ha = float(ref_energies[0])                    * EV_TO_HA
                delta_ha = float(ref_energies[j] - ref_energies[0]) * EV_TO_HA
                osc      = 0.0 if j == 0 else float(osc_strengths[j - 1])
                f.write(
                    f"{j+1:03}    {e_ha:.8f} {ref_e_ha:.8f}   "
                    f"0.00000000   0.00000000   0.00000000   "
                    f"0.00000000   0.00000000   0.00000000   "
                    f"{delta_ha:.8f}   {osc:.8f}\n"
                )
            f.write(f"Ekin    {ekin_ha:.8f} a.u.\n")
            f.write(f"Epot_harm    0.00000000 a.u.\n")
            f.write(f"Epot    {epot_ha:.8f} a.u.\n")
            f.write(f"Etot_harm    {ekin_ha:.8f} a.u.\n")
            f.write(f"Etot    {etot_ha:.8f} a.u.\n")
            f.write("\n\n")

print(f"Written {len(frames)-1} initial conditions + 1 equilibrium geometry to 'initconds'")