import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from e3nn import o3
from mace import data, modules, tools
from mace.tools import torch_geometric


@dataclass
class ModelTrainingResult:
    model: torch.nn.Module
    history: list[dict[str, float | int | str]]
    loss_fn: torch.nn.Module
    z_table: Any
    atomic_energies: np.ndarray
    atomic_energies_dict: dict[Any, Any]
    train_configs: list[Any]
    valid_configs: list[Any]
    train_loader: Any
    valid_loader: Any
    avg_num_neighbors: float
    n_energies: int
    device: torch.device
    encoder: torch.nn.Module

def add_autoencoder_targets(active_model, batch, output):
    """Add autoencoder targets to output for loss computation."""
    centred_reference_energy = (
        batch["energy"] - output["e0s"] - output["pair_energy"]
    ).unsqueeze(-1)
    
    output["encoded_energy"] = active_model.perm_encoder(centred_reference_energy)
    output["decoded_energy"] = (
        active_model.perm_decoder(output["encoded_energy"])
        + output["e0s"]
        + output["pair_energy"]
    )
    return output

def train_model(
    atoms_list,
    *,
    model=None,
    z_table=None,
    output_dir="../outputs",
    device="cpu",
    seed=123,
    max_epochs=50,
    learning_rate=1.0e-3,
    batch_size=4,
    r_max=5.0,
    valid_fraction=0.20,
    model_kwargs=None,
    loss_kwargs=None,
    print_every=5,
):
    """
    Train the X-MACE model.
    If a base model is provided, it will be used for transfer learning. Otherwise, a new model will be trained from scratch.
    Returns the intermediate objects in a ModelTrainingResult, which can be used for further transfer learning.
    """
    if not atoms_list:
        raise ValueError("atoms_list must contain at least one ASE Atoms object.")

    device = torch.device(device)
    model_kwargs = model_kwargs or {}
    loss_kwargs = loss_kwargs or {}

    np.random.seed(seed)
    torch.manual_seed(seed)

    atoms_configs = data.config_from_atoms_list(
        atoms_list,
        energy_key="REF_energy",
        forces_key="REF_forces",
        config_type_weights={"Default": 1.0},
    )

    train_configs, valid_configs = data.random_train_valid_split(
        atoms_configs,
        valid_fraction=valid_fraction,
        seed=seed,
        work_dir=output_dir+"/random_train_valid_split",
    )

    if z_table is None:
        z_table = tools.get_atomic_number_table_from_zs(
            z for config in atoms_configs for z in config.atomic_numbers
        )

    atomic_energies_dict = data.compute_average_E0s(train_configs, z_table)
    atomic_energies = np.array(
        [atomic_energies_dict[z] for z in z_table.zs],
        dtype=np.float64,
    )
    example_atom = atoms_list[0]
    n_energies = np.asarray(example_atom.info["REF_energy"]).shape[-1]

    train_set = [
        data.AtomicData.from_config(config, z_table=z_table, cutoff=r_max)
        for config in train_configs
    ]
    valid_set = [
        data.AtomicData.from_config(config, z_table=z_table, cutoff=r_max)
        for config in valid_configs
    ]

    train_loader = torch_geometric.dataloader.DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
    )
    valid_loader = torch_geometric.dataloader.DataLoader(
        valid_set,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )

    avg_num_neighbors = float(modules.compute_avg_num_neighbors(train_loader))

    default_model_kwargs = {
        "r_max": r_max,
        "num_bessel": 4,
        "num_polynomial_cutoff": 3,
        "num_permutational_invariant": 4,
        "n_energies": n_energies,
        "max_ell": 2,
        "interaction_cls": modules.interaction_classes[
            "RealAgnosticResidualInteractionBlock"
        ],
        "interaction_cls_first": modules.interaction_classes[
            "RealAgnosticResidualInteractionBlock"
        ],
        "num_interactions": 1,
        "num_elements": len(z_table),
        "hidden_irreps": o3.Irreps("4x0e + 4x1o"),
        "MLP_irreps": o3.Irreps("4x0e"),
        "atomic_energies": atomic_energies,
        "avg_num_neighbors": avg_num_neighbors,
        "atomic_numbers": [int(z) for z in z_table.zs],
        "correlation": 1,
        "gate": modules.gate_dict["silu"],
        "radial_MLP": [32, 32],
        "compute_nacs": False,
        "compute_socs": False,
        "nac_num": 0,
        "soc_num": 0,
    }
    default_model_kwargs.update(model_kwargs)

    if model is None:
        model = modules.AutoencoderExcitedMACE(**default_model_kwargs).to(device)

    default_loss_kwargs = {
        "energy_weight": 1.0,
        "forces_weight": 5.0,
        "dipoles_weight": 0.0,
        "nacs_weight": 0.0,
        "socs_weight": 0.0,
    }
    default_loss_kwargs.update(loss_kwargs)
    loss_fn = modules.InvariantsWeightedEnergyForcesNacsDipoleLoss(
        **default_loss_kwargs
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    history = []

    for epoch in range(max_epochs):
        epoch_start = time.time()
        model.train()
        train_losses = []

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)

            output = model(
                batch.to_dict(),
                training=True,
                compute_force=True,
                compute_virials=False,
                compute_stress=False,
            )
            output = add_autoencoder_targets(model, batch, output)

            loss = loss_fn(ref=batch, pred=output)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))

        model.eval()
        valid_losses = []
        for batch in valid_loader:
            batch = batch.to(device)
            output = model(
                batch.to_dict(),
                training=False,
                compute_force=True,
                compute_virials=False,
                compute_stress=False,
            )
            output = add_autoencoder_targets(model, batch, output)
            loss = loss_fn(ref=batch, pred=output)
            valid_losses.append(float(loss.detach().cpu()))

        row = {
            "stage": "base_model",
            "epoch": epoch + 1,
            "train_loss": float(np.mean(train_losses)),
            "valid_loss": float(np.mean(valid_losses)),
            "seconds": round(time.time() - epoch_start, 2),
        }
        history.append(row)

        if print_every and ((epoch + 1) % print_every == 0 or epoch == 0):
            print(row)

    print(f"\nTraining complete after {max_epochs} epochs")
    print(f"Final train loss: {history[-1]['train_loss']:.6f}")
    print(f"Final valid loss: {history[-1]['valid_loss']:.6f}")

    return ModelTrainingResult(
        model=model,
        history=history,
        loss_fn=loss_fn,
        z_table=z_table,
        atomic_energies=atomic_energies,
        atomic_energies_dict=atomic_energies_dict,
        train_configs=train_configs,
        valid_configs=valid_configs,
        train_loader=train_loader,
        valid_loader=valid_loader,
        avg_num_neighbors=avg_num_neighbors,
        n_energies=n_energies,
        device=device,
        encoder=model.perm_encoder
    )
