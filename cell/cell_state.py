"""Cell state fields — all persistent cell data as separate Taichi scalar fields."""

import numpy as np
import taichi as ti

from config import MAX_CELLS, GRID_WIDTH, GRID_HEIGHT, BOND_SIGNAL_CHANNELS

# Per-cell scalar fields
cell_alive = ti.field(dtype=ti.i32, shape=(MAX_CELLS,))
cell_x = ti.field(dtype=ti.i32, shape=(MAX_CELLS,))
cell_y = ti.field(dtype=ti.i32, shape=(MAX_CELLS,))
cell_energy = ti.field(dtype=ti.f32, shape=(MAX_CELLS,))
cell_structure = ti.field(dtype=ti.f32, shape=(MAX_CELLS,))
cell_repmat = ti.field(dtype=ti.f32, shape=(MAX_CELLS,))
cell_signal = ti.field(dtype=ti.f32, shape=(MAX_CELLS,))
cell_membrane = ti.field(dtype=ti.f32, shape=(MAX_CELLS,))
cell_age = ti.field(dtype=ti.i32, shape=(MAX_CELLS,))
cell_genome_id = ti.field(dtype=ti.i32, shape=(MAX_CELLS,))
cell_facing = ti.field(dtype=ti.i32, shape=(MAX_CELLS,))

# Bond connections (-1 = no bond)
cell_bonds = ti.field(dtype=ti.i32, shape=(MAX_CELLS, 4))

# Bond strength per slot (0.0-1.0, decays without reinforcement)
cell_bond_strength = ti.field(dtype=ti.f32, shape=(MAX_CELLS, 4))

# Bond signal channels: outgoing signals this cell emits, incoming signals from partners
cell_bond_signal_out = ti.field(dtype=ti.f32, shape=(MAX_CELLS, 4, BOND_SIGNAL_CHANNELS))
cell_bond_signal_in = ti.field(dtype=ti.f32, shape=(MAX_CELLS, 4, BOND_SIGNAL_CHANNELS))

# Spatial lookup: which cell occupies each grid position (-1 = empty)
grid_cell_id = ti.field(dtype=ti.i32, shape=(GRID_WIDTH, GRID_HEIGHT))

# Free-slot stack for cell allocation
free_slots = ti.field(dtype=ti.i32, shape=(MAX_CELLS,))
free_slot_count = ti.field(dtype=ti.i32, shape=())

# Live cell count
cell_count = ti.field(dtype=ti.i32, shape=())


@ti.kernel
def _init_fields():
    for i in range(MAX_CELLS):
        cell_alive[i] = 0
        free_slots[i] = i
        for b in range(4):
            cell_bonds[i, b] = -1
            cell_bond_strength[i, b] = 0.0
            for ch in range(BOND_SIGNAL_CHANNELS):
                cell_bond_signal_out[i, b, ch] = 0.0
                cell_bond_signal_in[i, b, ch] = 0.0
    for i, j in grid_cell_id:
        grid_cell_id[i, j] = -1


def init_cell_state():
    """Initialize all cell fields to defaults."""
    _init_fields()
    free_slot_count[None] = MAX_CELLS
    cell_count[None] = 0


def allocate_cell_python() -> int:
    """Allocate a cell slot from the free stack (Python-side, for spawning)."""
    count = free_slot_count[None]
    if count <= 0:
        return -1
    count -= 1
    free_slot_count[None] = count
    slot = free_slots[count]
    return int(slot)


def deallocate_cell_python(idx: int):
    """Return a cell slot to the free stack (Python-side)."""
    count = free_slot_count[None]
    free_slots[count] = idx
    free_slot_count[None] = count + 1
