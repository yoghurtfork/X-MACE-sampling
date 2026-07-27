import numpy as np
from ase.neighborlist import NeighborList, natural_cutoffs
import torch
from dscribe.descriptors import SOAP, ACSF, MBTR
from dscribe.core import system as dscribe_system

def _patch_dscribe_system_init_keyword_args():
    """Patch DScribe System.__init__ to pass keyword args to ASE.

    DScribe creates `System` objects using positional args, but ASE's
    `_LimitedAtoms.__init__` has a `velocities` parameter after `info`.
    This mismatch can cause `TypeError: Use only one of "momenta" and "velocities"`.
    """
    if getattr(dscribe_system.System, "_patched_init_with_keyword_args", False):
        return

    def _patched_init(
        self,
        symbols=None,
        positions=None,
        numbers=None,
        tags=None,
        momenta=None,
        masses=None,
        magmoms=None,
        charges=None,
        scaled_positions=None,
        cell=None,
        pbc=None,
        celldisp=None,
        constraint=None,
        calculator=None,
        info=None,
        wyckoff_positions=None,
        equivalent_atoms=None,
    ):
        super(dscribe_system.System, self).__init__(
            symbols=symbols,
            positions=positions,
            numbers=numbers,
            tags=tags,
            momenta=momenta,
            masses=masses,
            magmoms=magmoms,
            charges=charges,
            scaled_positions=scaled_positions,
            cell=cell,
            pbc=pbc,
            celldisp=celldisp,
            constraint=constraint,
            calculator=calculator,
            info=info,
        )
        self.wyckoff_positions = wyckoff_positions
        self.equivalent_atoms = equivalent_atoms
        self._cell_inverse = None
        self._displacement_tensor = None
        self._distance_matrix = None
        self._inverse_distance_matrix = None

    dscribe_system.System.__init__ = _patched_init
    dscribe_system.System._patched_init_with_keyword_args = True

_patch_dscribe_system_init_keyword_args()

def get_descriptor(descriptor_type, atoms, encoder=None, force_weight=1.0, energy_weight=1.0):
    """
    Router to the appropriate descriptor function. 
        - "bond_lengths": C-C bond lengths
        - "bond_angles": C-C-C bond angles
        - "dihedral": dihedral angle about the C=C bond
        - "energies": energies of the ASE Atoms object
        - "encoded_energies": energies encoded into a latent space representation using the XMACE encoder
        - "soap": SOAP descriptor using the DScribe library
        - "acsf": ACSF descriptor using the DScribe library
        - "mbtr": MBTR descriptor using the DScribe library
        - "ci_score": score which qualifies how close the geometry is to a conical intersection
    """
    if descriptor_type == "bond_lengths":
        return get_bond_lengths(atoms)
    elif descriptor_type == "bond_angles":
        return get_bond_angles(atoms)
    elif descriptor_type == "dihedral":
        return get_dihedral(atoms)
    elif descriptor_type == "soap":
        return get_soap(atoms)
    elif descriptor_type == "acsf":
        return get_acsf(atoms)
    elif descriptor_type == "mbtr":
        return get_mbtr(atoms)
    elif descriptor_type == "energies":
        return get_energies(atoms)
    elif descriptor_type == "encoded_energies":
        return get_encoded_energies(atoms, encoder)
    elif descriptor_type == "ci_score":
        return get_ci_score(atoms, force_weight, energy_weight)
    else:
        raise ValueError(
            f"Unknown descriptor type: {descriptor_type}. "
            f"Supported types: 'bond_lengths', 'bond_angles', 'soap', 'acsf', 'mbtr', 'energies', 'encoded_energies', 'ci_score'"
        )

def _get_bond_graph(atoms):
    """Return a deterministic covalent-bond adjacency map."""
    cutoffs = natural_cutoffs(atoms, mult=1.2)
    neighbor_list = NeighborList(
        cutoffs,
        self_interaction=False,
        bothways=True,
        skin=0.0,
    )
    neighbor_list.update(atoms)

    adjacency = {atom_idx: set() for atom_idx in range(len(atoms))}
    for atom_idx in range(len(atoms)):
        neighbor_indices, _ = neighbor_list.get_neighbors(atom_idx)
        adjacency[atom_idx].update(int(idx) for idx in neighbor_indices)
    return adjacency


def _get_carbon_bonds(atoms, adjacency):
    """Return adjacent carbon-index pairs in stable atom-index order."""
    symbols = atoms.get_chemical_symbols()
    return sorted(
        {
            tuple(sorted((atom_idx, neighbor_idx)))
            for atom_idx, neighbor_indices in adjacency.items()
            if symbols[atom_idx] == "C"
            for neighbor_idx in neighbor_indices
            if symbols[neighbor_idx] == "C"
        }
    )


def get_bond_lengths(atoms):
    """Return all covalently bonded C-C distances in deterministic order.

    The carbon skeleton is inferred from covalent-radius neighbor cutoffs, so
    this supports ethene, propene, butene, and larger hydrocarbons without
    relying on fixed atom indices. Distances are returned in Angstrom.
    """
    adjacency = _get_bond_graph(atoms)
    carbon_bonds = _get_carbon_bonds(atoms, adjacency)
    if not carbon_bonds:
        raise ValueError("Need at least one covalently bonded pair of carbon atoms.")

    return [atoms.get_distance(*pair, mic=True) for pair in carbon_bonds]

def get_bond_angles(atoms):
    """Return C-C-C bond angles for carbon triplets in an ASE Atoms object.

    Adjacent carbon atoms are detected with ASE covalent-radius neighbor cutoffs.
    Angles are calculated for all C-C-C triplets where the middle atom is bonded to
    the outer two atoms. Angles are returned in degrees.
    
    Raises:
        ValueError: If there are fewer than 3 carbon atoms in the structure.
    """
    symbols = atoms.get_chemical_symbols()
    c_count = sum(1 for s in symbols if s == "C")
    
    if c_count < 3:
        raise ValueError(f"Need at least 3 C atoms, but found {c_count}")
    
    cutoffs = natural_cutoffs(atoms, mult=1.2)
    neighbor_list = NeighborList(
        cutoffs,
        self_interaction=False,
        bothways=True,
        skin=0.0,
    )
    neighbor_list.update(atoms)

    bond_angles = []
    seen_triplets = set()

    for center_idx, symbol in enumerate(symbols):
        if symbol != "C":
            continue

        neighbor_indices, _ = neighbor_list.get_neighbors(center_idx)
        c_neighbors = [
            int(idx) for idx in neighbor_indices if symbols[int(idx)] == "C"
        ]

        # Calculate angles for all pairsof carbon neighbors
        for i, neighbor1_idx in enumerate(c_neighbors):
            for neighbor2_idx in c_neighbors[i + 1 :]:
                triplet = tuple(sorted((neighbor1_idx, center_idx, neighbor2_idx)))
                if triplet in seen_triplets:
                    continue

                seen_triplets.add(triplet)
                angle = atoms.get_angle(neighbor1_idx, center_idx, neighbor2_idx)
                bond_angles.append(angle)

    return bond_angles

def get_energies(atoms):
    """Return the energies of the ASE Atoms object."""
    return atoms.info["REF_energy"][0]


def get_ci_score(atoms, force_weight=1.0, energy_weight=1.0):
    """Calculate a conical-intersection score for one geometry.

    Scores are calculated for the two adjacent pairs of electronic states:
        (force_weight * RMS(F_j - F_i)) / (energy_weight * (E_j - E_i) + 1e-6)

    The larger of the two scores is returned. Each pair's intermediate values
    are printed for diagnostics.
    """
    if energy_weight < 0 or force_weight < 0:
        raise ValueError("energy_weight and force_weight must be non-negative.")

    energies = np.asarray(atoms.info["REF_energy"], dtype=float)
    if energies.shape != (1, 3):
        raise ValueError("atoms must provide exactly three energy levels.")
    energies = energies[0]

    if "REF_forces" in atoms.arrays:
        forces = np.asarray(atoms.arrays["REF_forces"], dtype=float)
    elif "REF_forces" in atoms.info:
        forces = np.asarray(atoms.info["REF_forces"], dtype=float)
    else:
        raise KeyError("atoms must provide state-resolved 'REF_forces'.")

    n_atoms = len(atoms)
    if forces.shape != (n_atoms, 3, 3):
        raise ValueError(
            "REF_forces must have shape (n_atoms, 3, 3); "
            f"got {forces.shape}."
        )

    pair_scores = []
    for first, second in ((0, 1), (1, 2)):
        gap = energies[second] - energies[first]
        delta_forces = forces[:, second, :] - forces[:, first, :]
        force_diff = np.sqrt(np.mean(delta_forces**2))
        score = (force_weight * force_diff) + (1/(energy_weight * gap + 1e-6))
        pair_scores.append(score)

    return [float(max(pair_scores))]

def get_encoded_energies(atoms, encoder):
    """
    Return the encoded energies of the ASE Atoms object.
    This uses the same encoder that X-MACE uses to encode the energies into a latent space representation.
    """
    energies = get_energies(atoms)

    # Prepare tensor with shape [batch=1, num_items, feature_dim=1]
    energies_tensor = torch.tensor(energies, dtype=torch.get_default_dtype()).reshape(1, -1, 1)
    
    # Move to encoder device if possible
    '''
    try:
        params = list(encoder.parameters())
        device = params[0].device if params else torch.device("cpu")
    except Exception:
        device = torch.device("cpu")
    energies_tensor = energies_tensor.to(device)
    '''

    # Run encoder and return a Python list of encoded values for the single geometry
    encoded_energies = encoder(energies_tensor)  # [1, latent_dim]
    return encoded_energies.squeeze(0).cpu().tolist()

def get_soap(atoms):
    """
    Return the SOAP descriptor using the DScribe library.
    SOAP (Smooth Overlap of Atomic Positions) captures the local environment around each atom
    by replacing each neighbour with a smooth Gaussian density.
    The SOAP descriptors for each atom are concatenated into a single vector.
    """

    # Create a SOAP descriptor object
    soap = SOAP(
        species=["C","H"],
        periodic=False,
        r_cut=5.0,
        n_max=8,
        l_max=6,
        sigma=0.3,
    )

    # Compute the SOAP descriptor for the atoms
    soap_descriptor = soap.create(atoms)

    # Concatenate into a list
    soap_descriptor = soap_descriptor.flatten().tolist()

    return soap_descriptor

def get_acsf(atoms):
    """
    Return the ACSF descriptor using the DScribe library.
    ACSF (Atom-Centered Symmetry Functions) captures the local environment around each atom
    by using symmetry functions.
    The ACSF descriptors for each atom are concatenated into a single vector.
    """

    # Create an ACSF descriptor object
    acsf = ACSF(
    species=["H", "C"],
    r_cut=5.0,
    g2_params=[
        [1.0, 0.0],
        [1.0, 1.0],
        [1.0, 2.0],
        [1.0, 3.0],
        [1.0, 4.0],
    ],
    g4_params=[
        [1.0, 1.0,  1.0],
        [1.0, 1.0, -1.0],
        [1.0, 2.0,  1.0],
        [1.0, 2.0, -1.0],
        [1.0, 4.0,  1.0],
        [1.0, 4.0, -1.0],
    ],
    periodic=False
    )

    # Compute the ACSF descriptor for the atoms
    acsf_descriptor = acsf.create(atoms)

    # Concatenate into a list
    acsf_descriptor = acsf_descriptor.flatten().tolist()

    return acsf_descriptor

def get_mbtr(atoms):
    """
    Return the MBTR descriptor using the DScribe library.
    MBTR (Many-Body Tensor Representation) describes the molecular geometry
    by capturing the distributions of distances/angles.
    The distribution for distances and distribution for angles are concatenated together.
    """

    # Create an MBTR descriptor object for distances
    mbtr_k2 = MBTR(
        species=["C", "H"],
        geometry={"function": "inverse_distance"},
        grid={
            "min": 0.0,
            "max": 1.1,
            "sigma": 0.02,
            "n": 75,
        },
        weighting={"function": "unity"},
        normalization="l2",
        periodic=False
    )
    
    # Create an MBTR descriptor object for angles
    mbtr_k3 = MBTR(
        species=["C", "H"],
        geometry={"function": "angle"},
        grid={
            "min": 0.0,
            "max": 180.0,
            "sigma": 3.0,
            "n": 90,
        },
        weighting={"function": "unity"},
        normalization="l2",
        periodic=False
    )
    
    # Compute the MBTR descriptors
    mbtr_descriptor_k2 = mbtr_k2.create(atoms)
    mbtr_descriptor_k3 = mbtr_k3.create(atoms)

    # Concatenate
    mbtr_descriptor = np.concatenate([mbtr_descriptor_k2, mbtr_descriptor_k3])

    return mbtr_descriptor

def get_dihedral(atoms):
    """Return the torsion angle about the most substituted C-C bond.

    Candidate C-C bonds are ranked by the number of hydrogens attached to
    their endpoints. This selects the alkene bond in the ethene, propene, and
    butene datasets. For each side of that bond, a carbon substituent is
    preferred over other elements, giving H-C-C-H, C-C-C-H, and C-C-C-C
    torsions respectively for those three molecules.
    """
    symbols = atoms.get_chemical_symbols()
    adjacency = _get_bond_graph(atoms)
    carbon_bonds = _get_carbon_bonds(atoms, adjacency)
    if not carbon_bonds:
        raise ValueError("Need at least one covalently bonded pair of carbon atoms.")

    def attached_hydrogen_count(bond):
        first, second = bond
        return sum(
            symbols[neighbor] == "H"
            for endpoint, other in ((first, second), (second, first))
            for neighbor in adjacency[endpoint]
            if neighbor != other
        )

    central_first, central_second = min(
        carbon_bonds,
        key=lambda bond: (attached_hydrogen_count(bond), bond),
    )

    def choose_substituent(endpoint, other_endpoint):
        candidates = [
            neighbor
            for neighbor in adjacency[endpoint]
            if neighbor != other_endpoint
        ]
        if not candidates:
            raise ValueError(
                "Cannot define a dihedral because a central carbon has "
                "no substituent."
            )
        return min(
            candidates,
            key=lambda idx: (symbols[idx] != "C", idx),
        )

    outer_first = choose_substituent(central_first, central_second)
    outer_second = choose_substituent(central_second, central_first)
    angle = atoms.get_dihedral(
        outer_first,
        central_first,
        central_second,
        outer_second,
        mic=bool(np.any(atoms.pbc)),
    )
    # ASE reports dihedrals in [0, 360). Fold equivalent clockwise and
    # anticlockwise rotations into [0, 180] so values close to 360 remain
    # adjacent to values close to 0 in descriptor space and scatter plots.
    unsigned_angle = min(angle, 360.0 - angle)
    return [unsigned_angle]
