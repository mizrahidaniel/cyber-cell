"""Archipelago: soft-wall quadrant partitioning for diversity maintenance.

Divides the 500x500 grid into 4 250x250 quadrants (islands). Boundaries have
reduced chemical diffusion, creating semi-isolated populations. Periodic
migration copies fit cells between quadrants.

Each quadrant gets slight parameter variance (light intensity, deposit density)
to encourage different evolutionary strategies.
"""

import numpy as np
import taichi as ti

from config import (
    GRID_WIDTH, GRID_HEIGHT, MAX_CELLS, NUM_ISLANDS,
    MIGRATION_INTERVAL, MIGRATION_COUNT, ISLAND_ENV_VARIANCE,
    ISLAND_BOUNDARY_DIFFUSION, RANDOM_SEED,
    LIGHT_BRIGHT, LIGHT_DIM, PHOTOSYNTHESIS_RATE,
)
from cell.cell_state import (
    cell_alive, cell_x, cell_y, cell_energy, cell_structure, cell_repmat,
    cell_signal, cell_membrane, cell_age, cell_genome_id, cell_facing,
    cell_bonds, cell_bond_strength, cell_bond_signal_out, cell_bond_signal_in,
    grid_cell_id, cell_count, free_slots, free_slot_count,
)
from cell.genome import (
    genome_weights, genome_ref_count, genome_free_list, genome_free_count,
    genome_count, genome_parent_id, genome_birth_tick, GENOME_SIZE,
)

# Per-island parameter multipliers (set during init)
island_light_mult = ti.field(dtype=ti.f32, shape=(NUM_ISLANDS,))
island_photo_mult = ti.field(dtype=ti.f32, shape=(NUM_ISLANDS,))

# Quadrant boundaries (2x2 grid)
_HALF_W = GRID_WIDTH // 2
_HALF_H = GRID_HEIGHT // 2

_rng = np.random.default_rng(RANDOM_SEED + 300)


@ti.func
def get_island(x: ti.i32, y: ti.i32) -> ti.i32:
    """Return island index (0-3) for a grid position."""
    ix = 0
    if x >= _HALF_W:
        ix = 1
    iy = 0
    if y >= _HALF_H:
        iy = 1
    return iy * 2 + ix


@ti.func
def is_boundary(i: ti.i32, j: ti.i32) -> ti.i32:
    """Return 1 if position is on a quadrant boundary, 0 otherwise."""
    on_boundary = 0
    if i == _HALF_W or i == _HALF_W - 1:
        on_boundary = 1
    if j == _HALF_H or j == _HALF_H - 1:
        on_boundary = 1
    return on_boundary


@ti.kernel
def apply_boundary_diffusion_damping(
    src: ti.template(), dst: ti.template(), normal_rate: ti.f32
):
    """Reduce diffusion at quadrant boundaries by blending toward boundary rate.

    Called after normal diffusion. At boundary cells, we blend the diffused
    value back toward the pre-diffusion value to simulate reduced flow.
    """
    boundary_rate = ISLAND_BOUNDARY_DIFFUSION
    for i, j in dst:
        if is_boundary(i, j) == 1:
            # The ratio of how much diffusion to suppress
            ratio = boundary_rate / ti.max(1e-6, normal_rate)
            # Blend: dst = src + ratio * (dst - src)
            # This effectively reduces the diffusion that already happened
            dst[i, j] = src[i, j] + ratio * (dst[i, j] - src[i, j])


def init_archipelago():
    """Initialize per-island parameter variations."""
    variance = ISLAND_ENV_VARIANCE
    for island in range(NUM_ISLANDS):
        light_mult = 1.0 + _rng.uniform(-variance, variance)
        photo_mult = 1.0 + _rng.uniform(-variance, variance)
        island_light_mult[island] = float(light_mult)
        island_photo_mult[island] = float(photo_mult)


def migrate_cells():
    """Copy MIGRATION_COUNT fit cells from each island to a random other island.

    Migrants are selected proportional to energy (fitness). The migrant is
    teleported to a random empty position in the destination island.
    """
    # Gather live cells per island
    alive_np = cell_alive.to_numpy()
    x_np = cell_x.to_numpy()
    y_np = cell_y.to_numpy()
    energy_np = cell_energy.to_numpy()

    islands = {q: [] for q in range(NUM_ISLANDS)}
    for i in range(MAX_CELLS):
        if alive_np[i] == 1:
            ix = 0 if x_np[i] < _HALF_W else 1
            iy = 0 if y_np[i] < _HALF_H else 1
            q = iy * 2 + ix
            islands[q].append(i)

    for src_island in range(NUM_ISLANDS):
        src_cells = islands[src_island]
        if len(src_cells) < MIGRATION_COUNT:
            continue

        # Pick destination island (not self)
        possible_dests = [q for q in range(NUM_ISLANDS) if q != src_island]
        dst_island = _rng.choice(possible_dests)

        # Select migrants randomly (uniform — avoids accelerating selective sweeps)
        migrant_indices = _rng.choice(len(src_cells), size=MIGRATION_COUNT,
                                       replace=False)

        # Destination quadrant bounds
        dst_ix = dst_island % 2
        dst_iy = dst_island // 2
        dst_x_min = dst_ix * _HALF_W
        dst_x_max = dst_x_min + _HALF_W
        dst_y_min = dst_iy * _HALF_H
        dst_y_max = dst_y_min + _HALF_H

        for mi in migrant_indices:
            cell_idx = src_cells[mi]
            # Find random empty position in destination
            for _ in range(50):  # max attempts
                nx = _rng.integers(dst_x_min, dst_x_max)
                ny = _rng.integers(dst_y_min, dst_y_max)
                if grid_cell_id[int(nx), int(ny)] == -1:
                    # Teleport cell
                    old_x = cell_x[cell_idx]
                    old_y = cell_y[cell_idx]
                    grid_cell_id[old_x, old_y] = -1
                    cell_x[cell_idx] = int(nx)
                    cell_y[cell_idx] = int(ny)
                    grid_cell_id[int(nx), int(ny)] = cell_idx
                    break
