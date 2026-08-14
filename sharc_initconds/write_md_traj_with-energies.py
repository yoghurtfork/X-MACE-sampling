from ase.io import read, write
import numpy as np
from mace.calculators import MACECalculator

ENERGY_MODEL = "/path/to/energy_model.model"   # <-- update
N_STATES     = 3                                # S0, S1, S2
N_FRAMES     = 101                              # frames to use (index 0:N_FRAMES)

# Convert .traj → .xyz so individual frames can be re-read
write("md_trajectory.xyz", read("md_trajectory.traj", index=":"))

atoms_list = read("md_trajectory.xyz", index=f"0:{N_FRAMES}")

calculator = MACECalculator(
    model_paths=ENERGY_MODEL,
    n_energies=N_STATES,
    device='cpu'
)

all_energies = []
for atoms in atoms_list:
    calculator.calculate(atoms)
    energy_array = np.array(calculator.results['energy']).reshape(1, -1)   # (1, N_STATES)
    all_energies.append(energy_array.flatten())
    # Replace atoms.info — velocities in atoms.arrays['momenta'] are unaffected
    atoms.info = {"REF_energy": energy_array}

np.savetxt("predicted_energies.txt", np.array(all_energies),
           header="Energies (eV): rows=geometries, columns=states (S0 S1 S2)")
write("md_traj_with-energies.xyz", atoms_list)
print(f"Written {len(atoms_list)} frames to 'md_traj_with-energies.xyz'")
print(f"REF_energy shape per frame: (1, {N_STATES})")