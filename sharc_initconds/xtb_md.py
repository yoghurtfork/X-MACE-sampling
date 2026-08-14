# Run a ground-state GFN2-xTB geometry relaxation followed by NVE MD.
# Usage: python xtb_md.py <input.xyz>

import sys
from ase.io import read
from ase.optimize import BFGS
from ase.md.verlet import VelocityVerlet
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary, ZeroRotation
from ase.io.trajectory import Trajectory
from ase import units
from xtb.ase.calculator import XTB

if len(sys.argv) < 2:
    raise ValueError("Usage: python xtb_md.py <input.xyz>")

# ── Load structure ────────────────────────────────────────────────────────
atoms = read(sys.argv[1])
atoms.calc = XTB(method="GFN2-xTB")

# ── Geometry relaxation ───────────────────────────────────────────────────
print(f"Initial energy: {atoms.get_potential_energy():.4f} eV")
optimizer = BFGS(atoms, logfile="relaxation.log")
optimizer.run(fmax=0.1)   # converge when max force < 0.1 eV/Å
print(f"Relaxed energy: {atoms.get_potential_energy():.4f} eV")
atoms.write("relaxed_mol.xyz")

# ── Initialise velocities at 300 K ───────────────────────────────────────
MaxwellBoltzmannDistribution(atoms, temperature_K=300)
Stationary(atoms)     # remove net linear momentum
ZeroRotation(atoms)   # remove net angular momentum

# ── NVE MD ───────────────────────────────────────────────────────────────
dyn  = VelocityVerlet(atoms, 1 * units.fs)
traj = Trajectory("md_trajectory.traj", "w", atoms)

def print_status():
    print(f"Step {dyn.nsteps:5d}  E_pot={atoms.get_potential_energy():.4f} eV  "
          f"E_kin={atoms.get_kinetic_energy():.4f} eV")

dyn.attach(traj.write,    interval=10)    # save every 10 steps → 10 000 frames
dyn.attach(print_status, interval=1000)
dyn.run(100_000)