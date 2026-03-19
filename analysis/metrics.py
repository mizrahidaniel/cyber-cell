"""Population stats, genome diversity measures, and birth/death tracking."""

import numpy as np
import taichi as ti

from config import MAX_CELLS, MAX_GENOMES
from cell.cell_state import cell_alive, cell_energy, cell_age, cell_repmat
from cell.genome import genome_ref_count

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
