"""Initial cell seeding and emergency respawning."""

import numpy as np

from config import (
    INITIAL_CELL_COUNT, INITIAL_ENERGY, INITIAL_STRUCTURE, INITIAL_REPMAT,
    MEMBRANE_INITIAL, LIGHT_ZONE_END, GRID_HEIGHT, RANDOM_SEED,
    RESPAWN_COUNT, GENOME_TYPE,
)
from cell.cell_state import (
    cell_alive, cell_x, cell_y, cell_energy, cell_structure, cell_repmat,
    cell_signal, cell_membrane, cell_age, cell_genome_id, cell_facing,
    grid_cell_id, cell_count, allocate_cell_python,
)


def seed_cells(count: int = INITIAL_CELL_COUNT, seed: int = RANDOM_SEED):
    """Place initial cells randomly in the light zone."""
    rng = np.random.default_rng(seed + 1)  # offset from chemistry seed
    placed = 0

    while placed < count:
        x = int(rng.integers(0, LIGHT_ZONE_END))
        y = int(rng.integers(0, GRID_HEIGHT))

        # Skip if position already occupied
        if grid_cell_id[x, y] >= 0:
            continue

        slot = allocate_cell_python()
        if slot < 0:
            break  # no more slots

        cell_alive[slot] = 1
        cell_x[slot] = x
        cell_y[slot] = y
        cell_energy[slot] = INITIAL_ENERGY
        cell_structure[slot] = INITIAL_STRUCTURE
        cell_repmat[slot] = INITIAL_REPMAT
        cell_signal[slot] = 0.0
        cell_membrane[slot] = MEMBRANE_INITIAL
        cell_age[slot] = 0
        cell_genome_id[slot] = placed  # each cell gets a unique genome for initial diversity
        cell_facing[slot] = int(rng.integers(0, 4))

        grid_cell_id[x, y] = slot
        placed += 1

    cell_count[None] = placed


_respawn_rng = np.random.default_rng(RANDOM_SEED + 200)


def respawn_cells(count: int = RESPAWN_COUNT):
    """Emergency respawn: add fresh cells when population is dangerously low.

    Uses existing genome allocation infrastructure to create new genomes
    with random parameters, mimicking immigration of novel organisms.
    """
    from cell.genome import (
        genome_weights, genome_ref_count, allocate_genome_python,
    )

    placed = 0
    for _ in range(count * 3):  # extra attempts for occupied cells
        if placed >= count:
            break

        x = int(_respawn_rng.integers(0, LIGHT_ZONE_END))
        y = int(_respawn_rng.integers(0, GRID_HEIGHT))

        if grid_cell_id[x, y] >= 0:
            continue

        slot = allocate_cell_python()
        if slot < 0:
            break

        # Allocate a fresh genome
        gid = allocate_genome_python()
        if gid < 0:
            break

        # Initialize genome with random weights
        if GENOME_TYPE == "crn":
            from cell.crn_genome import crn_weights, CRN_GENOME_SIZE
            for w in range(CRN_GENOME_SIZE):
                crn_weights[gid, w] = float(_respawn_rng.uniform(-0.5, 0.5))
        else:
            from config import GENOME_SIZE, SEED_WEIGHT_SIGMA
            for w in range(GENOME_SIZE):
                genome_weights[gid, w] = float(
                    _respawn_rng.normal(0, SEED_WEIGHT_SIGMA))

        genome_ref_count[gid] = 1

        cell_alive[slot] = 1
        cell_x[slot] = x
        cell_y[slot] = y
        cell_energy[slot] = INITIAL_ENERGY
        cell_structure[slot] = INITIAL_STRUCTURE
        cell_repmat[slot] = INITIAL_REPMAT
        cell_signal[slot] = 0.0
        cell_membrane[slot] = MEMBRANE_INITIAL
        cell_age[slot] = 0
        cell_genome_id[slot] = gid
        cell_facing[slot] = int(_respawn_rng.integers(0, 4))

        grid_cell_id[x, y] = slot
        placed += 1

    cell_count[None] = cell_count[None] + placed
    return placed
