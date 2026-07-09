import numpy as np
from ase.neighborlist import NeighborList, natural_cutoffs

def get_descriptor(descriptor_type, atoms):
    """
    Router to the appropriate descriptor function. 
        - "bond_lengths": C-C bond lengths
        - "bond_angles": C-C-C bond angles
    """
    if descriptor_type == "bond_lengths":
        return get_bond_lengths(atoms)
    elif descriptor_type == "bond_angles":
        return get_bond_angles(atoms)
    else:
        raise ValueError(
            f"Unknown descriptor type: {descriptor_type}. "
            f"Supported types: 'bond_lengths', 'bond_angles'"
        )

def get_bond_lengths(atoms):
    """Return C-C bond lengths for adjacent carbon atoms in an ASE Atoms object.

    Adjacent carbon atoms are detected with ASE covalent-radius neighbor cutoffs.
    Distances are returned in Angstrom, matching ASE position units.
    """
    symbols = atoms.get_chemical_symbols()
    cutoffs = natural_cutoffs(atoms, mult=1.2)
    neighbor_list = NeighborList(
        cutoffs,
        self_interaction=False,
        bothways=True,
        skin=0.0,
    )
    neighbor_list.update(atoms)

    bond_lengths = []
    seen_pairs = set()

    for atom_idx, symbol in enumerate(symbols):
        if symbol != "C":
            continue

        neighbor_indices, _ = neighbor_list.get_neighbors(atom_idx)
        for neighbor_idx in neighbor_indices:
            if symbols[neighbor_idx] != "C":
                continue

            pair = tuple(sorted((atom_idx, int(neighbor_idx))))
            if pair in seen_pairs:
                continue

            seen_pairs.add(pair)
            bond_lengths.append(atoms.get_distance(*pair, mic=True))

    return bond_lengths

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
