"""Save and load full simulation state for backend switching and resumption."""

import numpy as np

from config import MAX_CELLS, MAX_GENOMES, GENOME_SIZE, GRID_WIDTH, GRID_HEIGHT, NUM_DEPOSITS_S, NUM_DEPOSITS_R

from cell.cell_state import (
    cell_alive, cell_x, cell_y, cell_energy, cell_structure, cell_repmat,
    cell_signal, cell_membrane, cell_age, cell_genome_id, cell_facing,
    cell_bonds, cell_bond_strength, cell_bond_signal_out, cell_bond_signal_in,
    cell_last_attacker, grid_cell_id, free_slots, free_slot_count, cell_count,
)
from cell.genome import (
    genome_weights, genome_ref_count, genome_count,
    genome_free_list, genome_free_count, needs_mutation,
)
from world.chemistry import (
    env_S_a, env_S_b, env_R_a, env_R_b, env_G_a, env_G_b,
    deposit_S_x, deposit_S_y, deposit_R_x, deposit_R_y,
)


def save_checkpoint(path: str, tick: int, current_buffer: int, mutation_rng_state):
    """Save all simulation state to a .npz file."""
    np.savez_compressed(
        path,
        # Metadata
        tick=tick,
        current_buffer=current_buffer,
        mutation_rng_state=mutation_rng_state,
        # Cell state
        cell_alive=cell_alive.to_numpy(),
        cell_x=cell_x.to_numpy(),
        cell_y=cell_y.to_numpy(),
        cell_energy=cell_energy.to_numpy(),
        cell_structure=cell_structure.to_numpy(),
        cell_repmat=cell_repmat.to_numpy(),
        cell_signal=cell_signal.to_numpy(),
        cell_membrane=cell_membrane.to_numpy(),
        cell_age=cell_age.to_numpy(),
        cell_genome_id=cell_genome_id.to_numpy(),
        cell_facing=cell_facing.to_numpy(),
        cell_bonds=cell_bonds.to_numpy(),
        cell_bond_strength=cell_bond_strength.to_numpy(),
        cell_bond_signal_out=cell_bond_signal_out.to_numpy(),
        cell_bond_signal_in=cell_bond_signal_in.to_numpy(),
        cell_last_attacker=cell_last_attacker.to_numpy(),
        grid_cell_id=grid_cell_id.to_numpy(),
        free_slots=free_slots.to_numpy(),
        free_slot_count=free_slot_count[None],
        cell_count=cell_count[None],
        # Genome state
        genome_weights=genome_weights.to_numpy(),
        genome_ref_count=genome_ref_count.to_numpy(),
        genome_count=genome_count[None],
        genome_free_list=genome_free_list.to_numpy(),
        genome_free_count=genome_free_count[None],
        needs_mutation=needs_mutation.to_numpy(),
        # Chemistry state
        env_S_a=env_S_a.to_numpy(),
        env_S_b=env_S_b.to_numpy(),
        env_R_a=env_R_a.to_numpy(),
        env_R_b=env_R_b.to_numpy(),
        env_G_a=env_G_a.to_numpy(),
        env_G_b=env_G_b.to_numpy(),
        deposit_S_x=deposit_S_x.to_numpy(),
        deposit_S_y=deposit_S_y.to_numpy(),
        deposit_R_x=deposit_R_x.to_numpy(),
        deposit_R_y=deposit_R_y.to_numpy(),
    )


def load_checkpoint(path: str) -> dict:
    """Load simulation state from a .npz file and restore all fields.

    Returns a dict with metadata: tick, current_buffer, mutation_rng_state.
    """
    data = np.load(path, allow_pickle=True)

    # Cell state
    cell_alive.from_numpy(data["cell_alive"])
    cell_x.from_numpy(data["cell_x"])
    cell_y.from_numpy(data["cell_y"])
    cell_energy.from_numpy(data["cell_energy"])
    cell_structure.from_numpy(data["cell_structure"])
    cell_repmat.from_numpy(data["cell_repmat"])
    cell_signal.from_numpy(data["cell_signal"])
    cell_membrane.from_numpy(data["cell_membrane"])
    cell_age.from_numpy(data["cell_age"])
    cell_genome_id.from_numpy(data["cell_genome_id"])
    cell_facing.from_numpy(data["cell_facing"])
    cell_bonds.from_numpy(data["cell_bonds"])
    if "cell_bond_strength" in data:
        cell_bond_strength.from_numpy(data["cell_bond_strength"])
        cell_bond_signal_out.from_numpy(data["cell_bond_signal_out"])
        cell_bond_signal_in.from_numpy(data["cell_bond_signal_in"])
    if "cell_last_attacker" in data:
        cell_last_attacker.from_numpy(data["cell_last_attacker"])
    grid_cell_id.from_numpy(data["grid_cell_id"])
    free_slots.from_numpy(data["free_slots"])
    free_slot_count[None] = int(data["free_slot_count"])
    cell_count[None] = int(data["cell_count"])

    # Genome state
    genome_weights.from_numpy(data["genome_weights"])
    genome_ref_count.from_numpy(data["genome_ref_count"])
    genome_count[None] = int(data["genome_count"])
    genome_free_list.from_numpy(data["genome_free_list"])
    genome_free_count[None] = int(data["genome_free_count"])
    needs_mutation.from_numpy(data["needs_mutation"])

    # Chemistry state
    env_S_a.from_numpy(data["env_S_a"])
    env_S_b.from_numpy(data["env_S_b"])
    env_R_a.from_numpy(data["env_R_a"])
    env_R_b.from_numpy(data["env_R_b"])
    env_G_a.from_numpy(data["env_G_a"])
    env_G_b.from_numpy(data["env_G_b"])
    deposit_S_x.from_numpy(data["deposit_S_x"])
    deposit_S_y.from_numpy(data["deposit_S_y"])
    deposit_R_x.from_numpy(data["deposit_R_x"])
    deposit_R_y.from_numpy(data["deposit_R_y"])

    return {
        "tick": int(data["tick"]),
        "current_buffer": int(data["current_buffer"]),
        "mutation_rng_state": data["mutation_rng_state"],
    }
