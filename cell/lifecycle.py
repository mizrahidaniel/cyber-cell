"""Cell lifecycle: photosynthesis, metabolism, death, and chemical spillage."""

import taichi as ti

from config import (
    MAX_CELLS, GRID_WIDTH, GRID_HEIGHT,
    PHOTOSYNTHESIS_RATE, BASAL_METABOLISM, E_DECAY_FLAT, NETWORK_COST,
    ENERGY_ZERO_MEMBRANE_DAMAGE, AGE_MEMBRANE_DECAY, MAX_CELL_AGE,
    EAT_ABSORB_CAP, EAT_COST, S_ENERGY_VALUE, R_ENERGY_VALUE,
    PASSIVE_EAT_CAP,
)
from cell.cell_state import (
    cell_alive, cell_x, cell_y, cell_energy, cell_structure, cell_repmat,
    cell_signal, cell_membrane, cell_age, grid_cell_id,
    cell_count, free_slots, free_slot_count,
)
from world.grid import light_field


@ti.kernel
def photosynthesis(env_S: ti.template(), env_R: ti.template()):
    """Passive energy gain from light. Always active — not gated by neural net."""
    for i in range(MAX_CELLS):
        if cell_alive[i] == 1:
            light = light_field[cell_x[i], cell_y[i]]
            cell_energy[i] += light * PHOTOSYNTHESIS_RATE


@ti.kernel
def eat_passive(env_S: ti.template(), env_R: ti.template()):
    """Passive chemical absorption — always active like photosynthesis.

    Cells absorb a small amount of environmental S and R at their position.
    This ensures cells can accumulate replication material for division.
    The neural net 'eat' action can provide additional absorption.
    """
    for i in range(MAX_CELLS):
        if cell_alive[i] == 1:
            x = cell_x[i]
            y = cell_y[i]

            passive_cap = PASSIVE_EAT_CAP

            # Absorb S
            avail_s = env_S[x, y]
            take_s = ti.min(avail_s, passive_cap)
            if take_s > 0.0:
                env_S[x, y] -= take_s
                cell_structure[i] += take_s
                cell_energy[i] += take_s * S_ENERGY_VALUE

            # Absorb R
            avail_r = env_R[x, y]
            take_r = ti.min(avail_r, passive_cap)
            if take_r > 0.0:
                env_R[x, y] -= take_r
                cell_repmat[i] += take_r
                cell_energy[i] += take_r * R_ENERGY_VALUE


@ti.kernel
def apply_metabolism():
    """Deduct basal costs, apply membrane decay, increment age."""
    for i in range(MAX_CELLS):
        if cell_alive[i] == 1:
            # Passive energy drains
            cell_energy[i] -= BASAL_METABOLISM + E_DECAY_FLAT + NETWORK_COST
            cell_energy[i] = ti.max(0.0, cell_energy[i])

            # Membrane damage when out of energy
            if cell_energy[i] <= 0.0:
                cell_membrane[i] -= ENERGY_ZERO_MEMBRANE_DAMAGE

            # Age-based membrane decay
            if cell_age[i] > MAX_CELL_AGE:
                cell_membrane[i] -= AGE_MEMBRANE_DECAY

            cell_age[i] += 1


@ti.kernel
def check_death(env_S: ti.template(), env_R: ti.template(),
                env_G: ti.template()):
    """Kill cells with membrane <= 0, spill internals to environment."""
    for i in range(MAX_CELLS):
        if cell_alive[i] == 1 and cell_membrane[i] <= 0.0:
            x = cell_x[i]
            y = cell_y[i]

            # Spill internal chemicals to environment
            env_S[x, y] += cell_structure[i] + cell_energy[i] * 0.5
            env_R[x, y] += cell_repmat[i]
            env_G[x, y] += cell_signal[i]

            # Clear cell
            cell_alive[i] = 0
            cell_energy[i] = 0.0
            cell_structure[i] = 0.0
            cell_repmat[i] = 0.0
            cell_signal[i] = 0.0
            cell_membrane[i] = 0.0
            cell_age[i] = 0

            # Free grid cell
            grid_cell_id[x, y] = -1

            # Return slot to free stack
            slot = ti.atomic_add(free_slot_count[None], 1)
            free_slots[slot] = i

            # Decrement population
            ti.atomic_sub(cell_count[None], 1)
