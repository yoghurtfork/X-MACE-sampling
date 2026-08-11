"""Export Z/E azobenzene geometries with S0 and S1 properties to extxyz."""

import argparse
import json
from collections import Counter
from pathlib import Path

import msgpack


AZOBENZENE_INCHIKEYS = {
    "Z": "DMLAVOWQYNRWNQ-YPKPFQOONA-N",
    "E": "DMLAVOWQYNRWNQ-BUHFOSPRNA-N",
}
INCHIKEY_TO_ISOMER = {key: isomer for isomer, key in AZOBENZENE_INCHIKEYS.items()}
ATOMIC_SYMBOLS = {1: "H", 6: "C", 7: "N"}
DEFAULT_INPUT_FILENAME = "switches.msgpack"
DEFAULT_OUTPUT_FILENAME = "cis_trans_azobenzenes.xyz"


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments for the MessagePack input and XYZ output."""
    parser = argparse.ArgumentParser(
        description="Export Z/E azobenzene S0/S1 energies and forces to extended XYZ."
    )
    parser.add_argument(
        "input_path",
        nargs="?",
        type=Path,
        default=None,
        help=f"MessagePack input path (default: ./{DEFAULT_INPUT_FILENAME})",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help=(
            "Extended-XYZ output path "
            f"(default: {DEFAULT_OUTPUT_FILENAME} next to the input file)"
        ),
    )
    return parser.parse_args()


def has_complete_state_data(
    coordinates: list, ground_energy: object, first_excited_energy: object, forces: list
) -> bool:
    """Return whether the S0/S1 energy and force data are usable for one geometry."""
    return (
        ground_energy is not None
        and first_excited_energy is not None
        and all(force is not None and len(force) == len(coordinates) for force in forces)
    )


def write_azobenzene_xyz(
    input_path: Path, output_path: Path
) -> tuple[int, Counter, Counter]:
    """Stream every MessagePack chunk and write valid Z/E azobenzene geometries."""
    total_geometries = 0
    selected_counts: Counter = Counter()
    written_counts: Counter = Counter()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with input_path.open("rb") as msgpack_file, output_path.open("w") as xyz_file:
        for geometries in msgpack.Unpacker(msgpack_file, strict_map_key=False):
            total_geometries += len(geometries)
            for geometry in geometries.values():
                isomer = INCHIKEY_TO_ISOMER.get(
                    geometry.get("species", {}).get("inchikey")
                )
                if isomer is None:
                    continue

                selected_counts[isomer] += 1
                coordinates = geometry["xyz"]
                properties = geometry["props"]
                ground_energy = properties.get("totalenergy")
                ground_forces = properties.get("forces")
                excited_states = properties.get("excitedstates", [])
                first_excited_state = excited_states[0] if excited_states else {}
                first_excited_energy = first_excited_state.get("energy")
                first_excited_forces = first_excited_state.get("forces")
                forces = [ground_forces, first_excited_forces]

                if not has_complete_state_data(
                    coordinates, ground_energy, first_excited_energy, forces
                ):
                    continue

                energies = [[ground_energy, first_excited_energy]]
                # X-MACE expects state-resolved forces as
                # [atom, state, Cartesian component], not [state, atom, component].
                forces_by_atom = [
                    [ground_force, excited_force]
                    for ground_force, excited_force in zip(
                        ground_forces, first_excited_forces
                    )
                ]
                xyz_file.write(f"{len(coordinates)}\n")
                xyz_file.write(
                    "Properties=species:S:1:pos:R:3 "
                    f"azobenzene_isomer={isomer} "
                    f'REF_energy="_JSON {json.dumps(energies)}" '
                    f'REF_forces="_JSON {json.dumps(forces_by_atom)}"\n'
                )
                for atomic_number, x, y, z in coordinates:
                    symbol = ATOMIC_SYMBOLS[int(atomic_number)]
                    xyz_file.write(f"{symbol:<2} {x: .8f} {y: .8f} {z: .8f}\n")
                written_counts[isomer] += 1

    return total_geometries, selected_counts, written_counts


def write_run_log(
    log_path: Path,
    input_path: Path,
    output_path: Path,
    total_geometries: int,
    selected_counts: Counter,
    written_counts: Counter,
) -> None:
    """Write the export summary beside the extended-XYZ output."""
    skipped_counts = selected_counts - written_counts
    log_path.write_text(
        f"Read {total_geometries} total geometries from {input_path}\n"
        "Selected Z/E azobenzene geometries: "
        f"{sum(selected_counts.values())} "
        f"(Z: {selected_counts['Z']}, E: {selected_counts['E']})\n"
        f"Wrote {sum(written_counts.values())} geometries to "
        f"{output_path} "
        f"(Z: {written_counts['Z']}, E: {written_counts['E']})\n"
        "Skipped selected geometries without complete S0/S1 energy/force data: "
        f"{sum(skipped_counts.values())} "
        f"(Z: {skipped_counts['Z']}, E: {skipped_counts['E']})\n"
    )


def main() -> None:
    """Run the command-line exporter."""
    arguments = parse_arguments()
    input_path = arguments.input_path or Path(DEFAULT_INPUT_FILENAME)
    output_path = arguments.output or input_path.with_name(DEFAULT_OUTPUT_FILENAME)
    total_geometries, selected_counts, written_counts = write_azobenzene_xyz(
        input_path, output_path
    )
    write_run_log(
        output_path.with_suffix(".log"),
        input_path,
        output_path,
        total_geometries,
        selected_counts,
        written_counts,
    )


if __name__ == "__main__":
    main()
