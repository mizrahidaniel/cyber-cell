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
    from config import GENOME_TYPE

    refs = genome_ref_count.to_numpy()
    active_mask = refs > 0
    active_ids = np.where(active_mask)[0].astype(np.int32)

    if GENOME_TYPE == "crn":
        from cell.crn_genome import crn_weights
        from config import CRN_GENOME_SIZE
        gs = CRN_GENOME_SIZE
        all_w = crn_weights.to_numpy()
    else:
        gs = GENOME_SIZE
        all_w = genome_weights.to_numpy()

    if len(active_ids) == 0:
        return {"genome_ids": np.empty(0, dtype=np.int32),
                "weights": np.empty((0, gs), dtype=np.float32),
                "ref_counts": np.empty(0, dtype=np.int32),
                "parent_ids": np.empty(0, dtype=np.int32),
                "birth_ticks": np.empty(0, dtype=np.int32)}

    weights = all_w[active_ids]
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


def get_crn_snapshot() -> dict | None:
    """Extract CRN internal state for diagnostics. Returns None if not CRN."""
    from config import GENOME_TYPE
    if GENOME_TYPE != "crn":
        return None

    from config import (
        NUM_INTERNAL_CHEMICALS, NUM_SENSORY_CHEMICALS, NUM_HIDDEN_CHEMICALS,
        NUM_ACTION_CHEMICALS, MAX_REACTIONS, CRN_PARAMS_PER_REACTION,
    )
    from cell.crn_genome import crn_chemicals, crn_weights

    NC = NUM_INTERNAL_CHEMICALS    # 16
    NS = NUM_SENSORY_CHEMICALS     # 8
    NH = NUM_HIDDEN_CHEMICALS      # 4
    NA = NUM_ACTION_CHEMICALS      # 4
    MR = MAX_REACTIONS             # 16
    PPR = CRN_PARAMS_PER_REACTION  # 7
    REACT_END = MR * PPR           # 112

    alive = cell_alive.to_numpy() == 1
    count = int(alive.sum())
    if count == 0:
        return None

    chems = crn_chemicals.to_numpy()[:MAX_CELLS][alive]
    gids = cell_genome_id.to_numpy()[alive]
    weights = crn_weights.to_numpy()

    # Zone activation means
    sensory_mean = float(chems[:, :NS].mean())
    hidden_mean = float(chems[:, NS:NS + NH].mean())
    action_mean = float(chems[:, NS + NH:].mean())

    # Per-chemical stats
    chem_means = [float(chems[:, c].mean()) for c in range(NC)]
    chem_stds = [float(chems[:, c].std()) for c in range(NC)]

    # Population-weighted genome stats
    unique_gids, gid_counts = np.unique(gids, return_counts=True)
    total = float(gid_counts.sum())
    bias_w = np.zeros(NA)
    decay_w = np.zeros(NH)
    active_w = 0.0
    inv_count, inv_total = 0, 0

    for j, gid in enumerate(unique_gids):
        g, frac = int(gid), gid_counts[j] / total
        for a in range(NA):
            bias_w[a] += weights[g, REACT_END + a] * frac
        for h in range(NH):
            decay_w[h] += abs(weights[g, REACT_END + NA + h]) * frac
        active = sum(1 for r in range(MR) if abs(weights[g, r * PPR + 5]) > 0.001)
        active_w += active * frac
        for r in range(MR):
            for ti in (3, 4):
                inv_total += 1
                if weights[g, r * PPR + ti] < 0:
                    inv_count += 1

    # Dominant genome reaction topology
    dom_idx = int(np.argmax(gid_counts))
    dom_gid = int(unique_gids[dom_idx])
    dw = weights[dom_gid]
    zone_flow = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    dom_reactions = []

    def _zone(c):
        return 0 if c < NS else (1 if c < NS + NH else 2)

    for r in range(MR):
        base = r * PPR
        inp_a = int(abs(dw[base]) * NC) % NC
        inp_b = int(abs(dw[base + 1]) * NC) % NC
        out = int(abs(dw[base + 2]) * NC) % NC
        rate = float(dw[base + 5])
        dom_reactions.append({
            "input_a": inp_a, "input_b": inp_b, "output": out,
            "threshold_a": float(dw[base + 3]),
            "threshold_b": float(dw[base + 4]),
            "rate": rate, "decay": float(dw[base + 6]),
        })
        if abs(rate) > 0.001:
            zone_flow[max(_zone(inp_a), _zone(inp_b))][_zone(out)] += 1

    return {
        "sensory_mean": sensory_mean, "hidden_mean": hidden_mean,
        "action_mean": action_mean,
        "chem_means": chem_means, "chem_stds": chem_stds,
        "bias_mean": bias_w.tolist(), "decay_mean": decay_w.tolist(),
        "active_reactions": float(active_w),
        "inverted_threshold_frac": float(inv_count / max(1, inv_total)),
        "zone_flow": zone_flow,
        "dominant_gid": dom_gid,
        "dominant_count": int(gid_counts[dom_idx]),
        "dominant_reactions": dom_reactions,
        "num_active_genomes": len(unique_gids),
    }
