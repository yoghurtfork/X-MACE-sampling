from ase.io import read, write
import numpy as np
from mace.calculators import MACECalculator

OSC_MODEL = "/path/to/oscillator_model.model"   # <-- update
N_OSC     = 2   # S0→S1, S0→S2 — must match --n_energies used during oscillator training

atoms_list = read("md_traj_with-energies.xyz", index=":")

calculator = MACECalculator(
    model_paths=OSC_MODEL,
    n_energies=N_OSC,
    device='cpu'
)

all_osc = []
for atoms in atoms_list:
    calculator.calculate(atoms)
    osc_array = np.array(calculator.results['energy']).reshape(-1)   # (N_OSC,)
    all_osc.append(osc_array)
    # Add to atoms.info — does NOT overwrite REF_energy from Step 2
    atoms.info["REF_osc-strength"] = osc_array

np.savetxt("predicted_osc-strengths.txt", np.array(all_osc),
           header="Oscillator strengths: rows=geometries, columns=transitions (S0->S1, S0->S2)")
write("md_traj_with-energies-and-osc.xyz", atoms_list)
print(f"Written {len(atoms_list)} frames to 'md_traj_with-energies-and-osc.xyz'")

# Verify both keys are present before proceeding
test = atoms_list[1]
for k, v in test.info.items():
    print(f"  atoms.info['{k}']: shape {np.array(v).shape}")