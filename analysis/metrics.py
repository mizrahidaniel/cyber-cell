"""Population stats, genome diversity measures, and birth/death tracking."""

import numpy as np
import taichi as ti

from config import MAX_CELLS, MAX_GENOMES, ACTION_THRESHOLD
from cell.cell_state import (
    cell_alive, cell_energy, cell_age, cell_repmat, cell_x, cell_y,
    cell_membrane, cell_bonds, cell_genome_id, cell_facing,
)
from cell.genome import (
    genome_ref_count, action_outputs, genome_weights,
    genome_parent_id, genome_birth_tick,
)
from config import GENOME_SIZE
from cell.lifecycle import deaths_by_attack, deaths_by_starvation

# Birth/death counters (reset periodically)
births_counter = ti.field(dtype=ti.i32, shape=())
deaths_counter = ti.field(dtype=ti.i32, shape=())


def get_population_stats() -> dict:
    """Compute population statistics for alive cells."""
    alive = cell_alive.to_numpy() == 1
    count = int(alive.sum())

    if count == 0:
        return {
            "population": 0,
            "avg_energy": 0.0,
            "avg_age": 0.0,
            "avg_repmat": 0.0,
            "min_age": 0,
            "max_age": 0,
        }

    energies = cell_energy.to_numpy()[alive]
    ages = cell_age.to_numpy()[alive]
    repmats = cell_repmat.to_numpy()[alive]

    return {
        "population": count,
        "avg_energy": float(energies.mean()),
        "avg_age": float(ages.mean()),
        "avg_repmat": float(repmats.mean()),
        "min_age": int(ages.min()),
        "max_age": int(ages.max()),
    }


def get_movement_stats() -> dict:
    """Compute movement-related stats for chemotaxis detection."""
    alive = cell_alive.to_numpy() == 1
    count = int(alive.sum())

    if count == 0:
        return {"move_fraction": 0.0, "avg_x_position": 0.0}

    # Fraction of alive cells whose move output exceeds threshold
    move_outputs = action_outputs.to_numpy()[:MAX_CELLS, 0]
    movers = (move_outputs[alive] >= ACTION_THRESHOLD).sum()

    # Average x position (rightward shift indicates movement toward dark zone R deposits)
    x_positions = cell_x.to_numpy()[alive]

    return {
        "move_fraction": float(movers / count),
        "avg_x_position": float(x_positions.mean()),
    }


def get_genome_diversity() -> dict:
    """Compute genome diversity metrics."""
    refs = genome_ref_count.to_numpy()
    active = refs[refs > 0]
    num_genomes = len(active)

    if num_genomes == 0:
        return {"num_genomes": 0, "shannon_index": 0.0, "dominant_fraction": 0.0}

    total = active.sum()
    proportions = active / total
    # Shannon diversity index
    shannon = -float(np.sum(proportions * np.log(proportions + 1e-10)))
    # Fraction held by the most common genome
    dominant = float(active.max()) / total

    return {
        "num_genomes": num_genomes,
        "shannon_index": shannon,
        "dominant_fraction": dominant,
    }


def get_spatial_snapshot() -> dict:
    """Capture cell positions, bonds, and genome IDs for spatial analysis."""
    alive_np = cell_alive.to_numpy() == 1
    count = int(alive_np.sum())

    if count == 0:
        return {"positions": np.empty((0, 2), dtype=np.int32),
                "genome_ids": np.empty(0, dtype=np.int32),
                "facings": np.empty(0, dtype=np.int32),
                "bonds": np.empty((0, 2), dtype=np.int32)}

    xs = cell_x.to_numpy()[alive_np]
    ys = cell_y.to_numpy()[alive_np]
    positions = np.column_stack([xs, ys]).astype(np.int32)
    genome_ids = cell_genome_id.to_numpy()[alive_np].astype(np.int32)
    facings = cell_facing.to_numpy()[alive_np].astype(np.int32)

    # Build bond edge list (unique pairs only)
    alive_indices = np.where(alive_np)[0]
    index_map = {int(idx): i for i, idx in enumerate(alive_indices)}
    bonds_all = cell_bonds.to_numpy()[:MAX_CELLS]
    bond_pairs = []
    for local_i, global_i in enumerate(alive_indices):
        for b in range(4):
            partner = int(bonds_all[global_i, b])
            if partner >= 0 and partner in index_map and partner > global_i:
                bond_pairs.append([local_i, index_map[partner]])

    bonds = np.array(bond_pairs, dtype=np.int32) if bond_pairs else np.empty((0, 2), dtype=np.int32)

    return {"positions": positions, "genome_ids": genome_ids,
            "facings": facings, "bonds": bonds}


def get_genome_weight_snapshot() -> dict:
    """Capture weights, lineage, and ref counts for all active genomes."""
    refs = genome_ref_count.to_numpy()
    active_mask = refs > 0
    active_ids = np.where(active_mask)[0].astype(np.int32)

    if len(active_ids) == 0:
        return {"genome_ids": np.empty(0, dtype=np.int32),
                "weights": np.empty((0, GENOME_SIZE), dtype=np.float32),
                "ref_counts": np.empty(0, dtype=np.int32),
                "parent_ids": np.empty(0, dtype=np.int32),
                "birth_ticks": np.empty(0, dtype=np.int32)}

    all_weights = genome_weights.to_numpy()
    weights = all_weights[active_ids]
    ref_counts = refs[active_ids]
    parent_ids = genome_parent_id.to_numpy()[active_ids]
    birth_ticks = genome_birth_tick.to_numpy()[active_ids]

    return {"genome_ids": active_ids, "weights": weights,
            "ref_counts": ref_counts.astype(np.int32),
            "parent_ids": parent_ids.astype(np.int32),
            "birth_ticks": birth_ticks.astype(np.int32)}


def get_burst_spatial_snapshot() -> dict:
    """Capture cell positions, energy, and age for burst frame analysis."""
    alive_np = cell_alive.to_numpy() == 1
    count = int(alive_np.sum())

    if count == 0:
        return {"positions": np.empty((0, 2), dtype=np.int32),
                "genome_ids": np.empty(0, dtype=np.int32),
                "energies": np.empty(0, dtype=np.float32),
                "ages": np.empty(0, dtype=np.int32)}

    xs = cell_x.to_numpy()[alive_np]
    ys = cell_y.to_numpy()[alive_np]
    positions = np.column_stack([xs, ys]).astype(np.int32)
    genome_ids = cell_genome_id.to_numpy()[alive_np].astype(np.int32)
    energies = cell_energy.to_numpy()[alive_np].astype(np.float32)
    ages = cell_age.to_numpy()[alive_np].astype(np.int32)

    return {"positions": positions, "genome_ids": genome_ids,
            "energies": energies, "ages": ages}


def get_predation_stats() -> dict:
    """Compute predation and bonding metrics for Stage 3 detection."""
    alive = cell_alive.to_numpy() == 1
    count = int(alive.sum())

    if count == 0:
        return {
            "attack_fraction": 0.0,
            "avg_membrane": 0.0,
            "bond_fraction": 0.0,
            "deaths_by_attack": 0,
            "deaths_by_starvation": 0,
        }

    # Fraction of cells with attack output above threshold
    attack_out = action_outputs.to_numpy()[:MAX_CELLS, 8]
    attackers = int((attack_out[alive] >= ACTION_THRESHOLD).sum())

    # Average membrane integrity (drops indicate active combat)
    membranes = cell_membrane.to_numpy()[alive]

    # Fraction of cells with any bonds
    bonds = cell_bonds.to_numpy()[:MAX_CELLS]
    bonded_cells = int(((bonds[alive] >= 0).any(axis=1)).sum())

    # Death cause counters (cumulative since last reset)
    d_attack = int(deaths_by_attack[None])
    d_starve = int(deaths_by_starvation[None])

    # Reset counters for next interval
    deaths_by_attack[None] = 0
    deaths_by_starvation[None] = 0

    return {
        "attack_fraction": float(attackers / count),
        "avg_membrane": float(membranes.mean()),
        "bond_fraction": float(bonded_cells / count),
        "deaths_by_attack": d_attack,
        "deaths_by_starvation": d_starve,
    }
