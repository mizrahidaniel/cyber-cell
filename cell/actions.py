"""Cell action execution with two-phase conflict resolution."""

import taichi as ti

from config import (
    MAX_CELLS, GRID_WIDTH, GRID_HEIGHT, ACTION_THRESHOLD,
    MOVE_COST, TURN_COST, EAT_COST, SIGNAL_COST, ATTACK_COST,
    REPAIR_COST, REPAIR_S_COST, REPAIR_MEMBRANE_GAIN,
    EAT_ABSORB_CAP, S_ENERGY_VALUE, R_ENERGY_VALUE,
    ATTACK_MEMBRANE_DAMAGE, DIVIDE_COST, DIVIDE_R_COST,
    PARENT_RESOURCE_SHARE, DAUGHTER_RESOURCE_SHARE, MEMBRANE_INITIAL,
    BOND_SIGNAL_CHANNELS,
)
from cell.cell_state import (
    cell_alive, cell_x, cell_y, cell_energy, cell_structure, cell_repmat,
    cell_signal, cell_membrane, cell_age, cell_genome_id, cell_facing,
    cell_bonds, cell_bond_signal_out, cell_last_attacker, grid_cell_id,
    cell_count, free_slots, free_slot_count,
)
from cell.genome import action_outputs, needs_mutation
from cell.sensing import facing_offset

# Movement intention fields
intent_move_active = ti.field(dtype=ti.i32, shape=(MAX_CELLS,))
intent_move_x = ti.field(dtype=ti.i32, shape=(MAX_CELLS,))
intent_move_y = ti.field(dtype=ti.i32, shape=(MAX_CELLS,))
grid_move_claim = ti.field(dtype=ti.i32, shape=(GRID_WIDTH, GRID_HEIGHT))

# Division intention fields
intent_divide_active = ti.field(dtype=ti.i32, shape=(MAX_CELLS,))
intent_divide_x = ti.field(dtype=ti.i32, shape=(MAX_CELLS,))
intent_divide_y = ti.field(dtype=ti.i32, shape=(MAX_CELLS,))
grid_divide_claim = ti.field(dtype=ti.i32, shape=(GRID_WIDTH, GRID_HEIGHT))


@ti.kernel
def clear_intentions():
    """Reset all intention fields at the start of each tick."""
    for i in range(MAX_CELLS):
        intent_move_active[i] = 0
        intent_divide_active[i] = 0
    for i, j in grid_move_claim:
        grid_move_claim[i, j] = MAX_CELLS
        grid_divide_claim[i, j] = MAX_CELLS


@ti.kernel
def process_turns():
    """Handle turning actions. If both fire, pick higher activation."""
    for i in range(MAX_CELLS):
        if cell_alive[i] == 1:
            turn_l = action_outputs[i, 1]
            turn_r = action_outputs[i, 2]

            if turn_l > ACTION_THRESHOLD or turn_r > ACTION_THRESHOLD:
                if turn_l > turn_r:
                    cell_facing[i] = (cell_facing[i] - 1 + 4) % 4
                else:
                    cell_facing[i] = (cell_facing[i] + 1) % 4
                cell_energy[i] -= TURN_COST
                cell_energy[i] = ti.max(0.0, cell_energy[i])


@ti.kernel
def process_movement_phase1():
    """Phase 1: each cell declares its movement intention and claims the target."""
    for i in range(MAX_CELLS):
        if cell_alive[i] == 1 and action_outputs[i, 0] > ACTION_THRESHOLD:
            # Bonded cells cannot move independently
            bonded = 0
            for b in range(4):
                if cell_bonds[i, b] >= 0:
                    bonded = 1
            if bonded == 0 and cell_energy[i] >= MOVE_COST:
                offset = facing_offset(cell_facing[i])
                tx = (cell_x[i] + offset[0] + GRID_WIDTH) % GRID_WIDTH
                ty = (cell_y[i] + offset[1] + GRID_HEIGHT) % GRID_HEIGHT

                intent_move_active[i] = 1
                intent_move_x[i] = tx
                intent_move_y[i] = ty

                # Lowest cell index wins the claim
                ti.atomic_min(grid_move_claim[tx, ty], i)


@ti.kernel
def process_movement_phase2():
    """Phase 2: resolve movement conflicts — only the winner moves."""
    for i in range(MAX_CELLS):
        if intent_move_active[i] == 1:
            tx = intent_move_x[i]
            ty = intent_move_y[i]

            # Did we win the claim AND is the target still empty?
            if grid_move_claim[tx, ty] == i and grid_cell_id[tx, ty] == -1:
                ox = cell_x[i]
                oy = cell_y[i]

                # Clear old position
                grid_cell_id[ox, oy] = -1
                # Occupy new position
                cell_x[i] = tx
                cell_y[i] = ty
                grid_cell_id[tx, ty] = i

                cell_energy[i] -= MOVE_COST
                cell_energy[i] = ti.max(0.0, cell_energy[i])


@ti.kernel
def process_eat(env_S: ti.template(), env_R: ti.template()):
    """Absorb environmental S and R at current position."""
    for i in range(MAX_CELLS):
        if cell_alive[i] == 1 and action_outputs[i, 3] > ACTION_THRESHOLD:
            x = cell_x[i]
            y = cell_y[i]

            # Absorb S
            avail_s = env_S[x, y]
            take_s = ti.min(avail_s, EAT_ABSORB_CAP)
            if take_s > 0.0:
                ti.atomic_sub(env_S[x, y], take_s)
                cell_structure[i] += take_s
                cell_energy[i] += take_s * S_ENERGY_VALUE

            # Absorb R
            avail_r = env_R[x, y]
            take_r = ti.min(avail_r, EAT_ABSORB_CAP)
            if take_r > 0.0:
                ti.atomic_sub(env_R[x, y], take_r)
                cell_repmat[i] += take_r
                cell_energy[i] += take_r * R_ENERGY_VALUE

            cell_energy[i] -= EAT_COST
            cell_energy[i] = ti.max(0.0, cell_energy[i])


@ti.kernel
def process_emit_signal(env_G: ti.template()):
    """Emit signal chemical G into the environment."""
    for i in range(MAX_CELLS):
        if cell_alive[i] == 1 and action_outputs[i, 4] > ACTION_THRESHOLD:
            x = cell_x[i]
            y = cell_y[i]
            emit_amount = cell_signal[i] * 0.5
            if emit_amount > 0.0:
                ti.atomic_add(env_G[x, y], emit_amount)
                cell_signal[i] -= emit_amount
            cell_energy[i] -= SIGNAL_COST
            cell_energy[i] = ti.max(0.0, cell_energy[i])


@ti.kernel
def process_repair():
    """Spend structure to repair membrane."""
    for i in range(MAX_CELLS):
        if cell_alive[i] == 1 and action_outputs[i, 9] > ACTION_THRESHOLD:
            if cell_structure[i] >= REPAIR_S_COST:
                cell_membrane[i] = ti.min(100.0, cell_membrane[i] + REPAIR_MEMBRANE_GAIN)
                cell_structure[i] -= REPAIR_S_COST
                cell_energy[i] -= REPAIR_COST
                cell_energy[i] = ti.max(0.0, cell_energy[i])


@ti.kernel
def process_bond_signal_output():
    """Write bond signal outputs [10..13] to all active bond slots. No energy cost."""
    for i in range(MAX_CELLS):
        if cell_alive[i] == 1:
            for ch in range(BOND_SIGNAL_CHANNELS):
                # action_outputs[i, 10 + ch] is the sigmoid output for this channel
                val = action_outputs[i, 10 + ch]
                for b in range(4):
                    if cell_bonds[i, b] >= 0:
                        cell_bond_signal_out[i, b, ch] = val
                    else:
                        cell_bond_signal_out[i, b, ch] = 0.0


@ti.kernel
def process_attack():
    """Attack the cell directly ahead, damaging its membrane."""
    for i in range(MAX_CELLS):
        if cell_alive[i] == 1 and action_outputs[i, 8] > ACTION_THRESHOLD:
            if cell_energy[i] >= ATTACK_COST:
                offset = facing_offset(cell_facing[i])
                ax = (cell_x[i] + offset[0] + GRID_WIDTH) % GRID_WIDTH
                ay = (cell_y[i] + offset[1] + GRID_HEIGHT) % GRID_HEIGHT

                target = grid_cell_id[ax, ay]
                if target >= 0 and cell_alive[target] == 1:
                    ti.atomic_sub(cell_membrane[target], ATTACK_MEMBRANE_DAMAGE)
                    cell_last_attacker[target] = i

                cell_energy[i] -= ATTACK_COST
                cell_energy[i] = ti.max(0.0, cell_energy[i])


# =============================================================================
# Division (Phase 4)
# =============================================================================

@ti.kernel
def process_divide_phase1():
    """Phase 1: cells declare intent to divide and claim an adjacent empty cell."""
    for i in range(MAX_CELLS):
        if cell_alive[i] == 1 and action_outputs[i, 5] > ACTION_THRESHOLD:
            if cell_energy[i] >= DIVIDE_COST and cell_repmat[i] >= DIVIDE_R_COST:
                x = cell_x[i]
                y = cell_y[i]

                # Check 4 neighbors in deterministic order: up, right, down, left
                found = 0
                tx = 0
                ty = 0
                for d in range(4):
                    if found == 0:
                        off = facing_offset(d)
                        nx = (x + off[0] + GRID_WIDTH) % GRID_WIDTH
                        ny = (y + off[1] + GRID_HEIGHT) % GRID_HEIGHT
                        if grid_cell_id[nx, ny] == -1:
                            tx = nx
                            ty = ny
                            found = 1

                if found == 1:
                    intent_divide_active[i] = 1
                    intent_divide_x[i] = tx
                    intent_divide_y[i] = ty
                    ti.atomic_min(grid_divide_claim[tx, ty], i)


@ti.kernel
def process_divide_phase2():
    """Phase 2: winners allocate daughter cells."""
    for i in range(MAX_CELLS):
        if intent_divide_active[i] == 1:
            tx = intent_divide_x[i]
            ty = intent_divide_y[i]

            if grid_divide_claim[tx, ty] == i and grid_cell_id[tx, ty] == -1:
                # Allocate a cell slot atomically
                slot_idx = ti.atomic_sub(free_slot_count[None], 1) - 1
                if slot_idx >= 0:
                    daughter = free_slots[slot_idx]

                    # Deduct division costs from parent
                    cell_energy[i] -= DIVIDE_COST
                    cell_repmat[i] -= DIVIDE_R_COST

                    # Split remaining resources
                    daughter_e = cell_energy[i] * DAUGHTER_RESOURCE_SHARE
                    daughter_s = cell_structure[i] * DAUGHTER_RESOURCE_SHARE
                    daughter_r = cell_repmat[i] * DAUGHTER_RESOURCE_SHARE
                    daughter_g = cell_signal[i] * DAUGHTER_RESOURCE_SHARE

                    cell_energy[i] *= PARENT_RESOURCE_SHARE
                    cell_structure[i] *= PARENT_RESOURCE_SHARE
                    cell_repmat[i] *= PARENT_RESOURCE_SHARE
                    cell_signal[i] *= PARENT_RESOURCE_SHARE

                    # Set daughter state
                    cell_alive[daughter] = 1
                    cell_x[daughter] = tx
                    cell_y[daughter] = ty
                    cell_energy[daughter] = daughter_e
                    cell_structure[daughter] = daughter_s
                    cell_repmat[daughter] = daughter_r
                    cell_signal[daughter] = daughter_g
                    cell_membrane[daughter] = MEMBRANE_INITIAL
                    cell_age[daughter] = 0
                    cell_genome_id[daughter] = cell_genome_id[i]
                    # Random facing via Taichi random
                    cell_facing[daughter] = ti.cast(ti.random() * 4.0, ti.i32) % 4

                    grid_cell_id[tx, ty] = daughter
                    needs_mutation[daughter] = 1

                    ti.atomic_add(cell_count[None], 1)
                else:
                    # Allocation failed, restore counter
                    ti.atomic_add(free_slot_count[None], 1)
