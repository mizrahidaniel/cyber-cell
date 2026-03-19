"""Bonding system: mutual bond formation, unilateral unbonding, and chemical sharing."""

import taichi as ti

from config import (
    MAX_CELLS, GRID_WIDTH, GRID_HEIGHT, ACTION_THRESHOLD,
    BOND_COST, BOND_SHARE_RATE, MOVE_COST,
)
from cell.cell_state import (
    cell_alive, cell_x, cell_y, cell_energy, cell_structure, cell_repmat,
    cell_signal, cell_facing, cell_bonds, grid_cell_id,
)
from cell.genome import action_outputs
from cell.sensing import facing_offset

# Two-phase claim field for bond formation
grid_bond_claim = ti.field(dtype=ti.i32, shape=(GRID_WIDTH, GRID_HEIGHT))


@ti.kernel
def clear_bond_claims():
    """Reset bond claim grid before each tick."""
    for i, j in grid_bond_claim:
        grid_bond_claim[i, j] = MAX_CELLS


@ti.kernel
def process_bond_phase1():
    """Phase 1: cells declare bond intent by claiming the cell ahead."""
    for i in range(MAX_CELLS):
        if cell_alive[i] == 1 and action_outputs[i, 6] > ACTION_THRESHOLD:
            if cell_energy[i] >= BOND_COST:
                offset = facing_offset(cell_facing[i])
                bx = (cell_x[i] + offset[0] + GRID_WIDTH) % GRID_WIDTH
                by = (cell_y[i] + offset[1] + GRID_HEIGHT) % GRID_HEIGHT
                target = grid_cell_id[bx, by]
                if target >= 0 and cell_alive[target] == 1:
                    # Claim: write our index at target's position
                    ti.atomic_min(grid_bond_claim[bx, by], i)


@ti.kernel
def process_bond_phase2():
    """Phase 2: mutual bond formation — both cells must want to bond."""
    for i in range(MAX_CELLS):
        if cell_alive[i] == 1 and action_outputs[i, 6] > ACTION_THRESHOLD:
            if cell_energy[i] >= BOND_COST:
                # Check if someone claimed our position
                claimer = grid_bond_claim[cell_x[i], cell_y[i]]
                if claimer < MAX_CELLS and claimer != i and cell_alive[claimer] == 1:
                    # Verify claimer is adjacent to us
                    dx = cell_x[claimer] - cell_x[i]
                    dy = cell_y[claimer] - cell_y[i]
                    # Handle toroidal wrapping
                    if dx > GRID_WIDTH // 2:
                        dx -= GRID_WIDTH
                    if dx < -(GRID_WIDTH // 2):
                        dx += GRID_WIDTH
                    if dy > GRID_HEIGHT // 2:
                        dy -= GRID_HEIGHT
                    if dy < -(GRID_HEIGHT // 2):
                        dy += GRID_HEIGHT
                    dist = ti.abs(dx) + ti.abs(dy)
                    if dist == 1:
                        # Check not already bonded to this cell
                        already_bonded = 0
                        for b in range(4):
                            if cell_bonds[i, b] == claimer:
                                already_bonded = 1
                        if already_bonded == 0:
                            # Find empty bond slots for both
                            my_slot = -1
                            for b in range(4):
                                if cell_bonds[i, b] == -1 and my_slot == -1:
                                    my_slot = b
                            their_slot = -1
                            for b in range(4):
                                if cell_bonds[claimer, b] == -1 and their_slot == -1:
                                    their_slot = b
                            if my_slot >= 0 and their_slot >= 0:
                                cell_bonds[i, my_slot] = claimer
                                cell_bonds[claimer, their_slot] = i
                                cell_energy[i] -= BOND_COST
                                cell_energy[i] = ti.max(0.0, cell_energy[i])


@ti.kernel
def process_unbond():
    """Release all bonds unilaterally when unbond output fires."""
    for i in range(MAX_CELLS):
        if cell_alive[i] == 1 and action_outputs[i, 7] > ACTION_THRESHOLD:
            for b in range(4):
                partner = cell_bonds[i, b]
                if partner >= 0:
                    # Remove us from partner's bond list
                    for pb in range(4):
                        if cell_bonds[partner, pb] == i:
                            cell_bonds[partner, pb] = -1
                    cell_bonds[i, b] = -1


@ti.kernel
def process_bond_sharing():
    """Bonded cells share chemicals: flow from higher to lower concentration."""
    for i in range(MAX_CELLS):
        if cell_alive[i] == 1:
            for b in range(4):
                partner = cell_bonds[i, b]
                if partner >= 0 and cell_alive[partner] == 1:
                    # Only process when i < partner to avoid double-counting
                    if i < partner:
                        rate = BOND_SHARE_RATE

                        # Energy sharing
                        diff_e = (cell_energy[i] - cell_energy[partner]) * rate
                        if diff_e > 0.0:
                            cell_energy[i] -= diff_e
                            cell_energy[partner] += diff_e
                        elif diff_e < 0.0:
                            cell_energy[partner] += diff_e
                            cell_energy[i] -= diff_e

                        # Structure sharing
                        diff_s = (cell_structure[i] - cell_structure[partner]) * rate
                        if diff_s > 0.0:
                            cell_structure[i] -= diff_s
                            cell_structure[partner] += diff_s
                        elif diff_s < 0.0:
                            cell_structure[partner] += diff_s
                            cell_structure[i] -= diff_s

                        # Replication material sharing
                        diff_r = (cell_repmat[i] - cell_repmat[partner]) * rate
                        if diff_r > 0.0:
                            cell_repmat[i] -= diff_r
                            cell_repmat[partner] += diff_r
                        elif diff_r < 0.0:
                            cell_repmat[partner] += diff_r
                            cell_repmat[i] -= diff_r

                    # Bond maintenance cost (each cell pays)
                    cell_energy[i] -= BOND_COST
                    cell_energy[i] = ti.max(0.0, cell_energy[i])


# =============================================================================
# Bonded group movement — cells move as a unit
# =============================================================================

bonded_move_active = ti.field(dtype=ti.i32, shape=(MAX_CELLS,))
bonded_move_dx = ti.field(dtype=ti.i32, shape=(MAX_CELLS,))
bonded_move_dy = ti.field(dtype=ti.i32, shape=(MAX_CELLS,))
grid_bonded_move_claim = ti.field(dtype=ti.i32, shape=(GRID_WIDTH, GRID_HEIGHT))


@ti.kernel
def clear_bonded_move_intents():
    for i in range(MAX_CELLS):
        bonded_move_active[i] = 0
    for i, j in grid_bonded_move_claim:
        grid_bonded_move_claim[i, j] = MAX_CELLS


@ti.kernel
def process_bonded_movement_phase1():
    """Bonded cells propose group moves. Lowest-index mover in a group leads."""
    for i in range(MAX_CELLS):
        if cell_alive[i] == 1 and action_outputs[i, 0] > ACTION_THRESHOLD:
            bonded = 0
            for b in range(4):
                if cell_bonds[i, b] >= 0:
                    bonded = 1
            if bonded == 1 and cell_energy[i] >= MOVE_COST:
                # Am I the lowest-index mover among my direct bond partners?
                is_leader = 1
                for b in range(4):
                    p = cell_bonds[i, b]
                    if p >= 0 and p < i and cell_alive[p] == 1:
                        if action_outputs[p, 0] > ACTION_THRESHOLD and cell_energy[p] >= MOVE_COST:
                            is_leader = 0

                if is_leader == 1:
                    offset = facing_offset(cell_facing[i])
                    dx = offset[0]
                    dy = offset[1]

                    tx = (cell_x[i] + dx + GRID_WIDTH) % GRID_WIDTH
                    ty = (cell_y[i] + dy + GRID_HEIGHT) % GRID_HEIGHT

                    can_move = 1

                    # My target must be empty or a bond partner
                    target_id = grid_cell_id[tx, ty]
                    if target_id != -1:
                        is_partner = 0
                        for b in range(4):
                            if cell_bonds[i, b] == target_id:
                                is_partner = 1
                        if is_partner == 0:
                            can_move = 0

                    # Each partner's shifted position must be empty, me, or another partner
                    if can_move == 1:
                        for b in range(4):
                            p = cell_bonds[i, b]
                            if p >= 0 and cell_alive[p] == 1:
                                px = (cell_x[p] + dx + GRID_WIDTH) % GRID_WIDTH
                                py = (cell_y[p] + dy + GRID_HEIGHT) % GRID_HEIGHT
                                p_target = grid_cell_id[px, py]
                                if p_target != -1 and p_target != i:
                                    is_group = 0
                                    for b2 in range(4):
                                        if cell_bonds[i, b2] == p_target:
                                            is_group = 1
                                    if is_group == 0:
                                        can_move = 0

                    if can_move == 1:
                        bonded_move_active[i] = 1
                        bonded_move_dx[i] = dx
                        bonded_move_dy[i] = dy

                        ti.atomic_min(grid_bonded_move_claim[tx, ty], i)
                        for b in range(4):
                            p = cell_bonds[i, b]
                            if p >= 0 and cell_alive[p] == 1:
                                px = (cell_x[p] + dx + GRID_WIDTH) % GRID_WIDTH
                                py = (cell_y[p] + dy + GRID_HEIGHT) % GRID_HEIGHT
                                ti.atomic_min(grid_bonded_move_claim[px, py], i)


@ti.kernel
def process_bonded_movement_phase2():
    """Resolve conflicts and execute winning group moves."""
    for i in range(MAX_CELLS):
        if bonded_move_active[i] == 1:
            dx = bonded_move_dx[i]
            dy = bonded_move_dy[i]

            tx = (cell_x[i] + dx + GRID_WIDTH) % GRID_WIDTH
            ty = (cell_y[i] + dy + GRID_HEIGHT) % GRID_HEIGHT

            # Check if we won ALL target claims
            all_won = 1
            if grid_bonded_move_claim[tx, ty] != i:
                all_won = 0

            if all_won == 1:
                for b in range(4):
                    p = cell_bonds[i, b]
                    if p >= 0 and cell_alive[p] == 1:
                        px = (cell_x[p] + dx + GRID_WIDTH) % GRID_WIDTH
                        py = (cell_y[p] + dy + GRID_HEIGHT) % GRID_HEIGHT
                        if grid_bonded_move_claim[px, py] != i:
                            all_won = 0

            if all_won == 1:
                # Clear ALL old positions first
                grid_cell_id[cell_x[i], cell_y[i]] = -1
                for b in range(4):
                    p = cell_bonds[i, b]
                    if p >= 0 and cell_alive[p] == 1:
                        grid_cell_id[cell_x[p], cell_y[p]] = -1

                # Move leader
                cell_x[i] = tx
                cell_y[i] = ty
                grid_cell_id[tx, ty] = i

                # Move partners
                for b in range(4):
                    p = cell_bonds[i, b]
                    if p >= 0 and cell_alive[p] == 1:
                        px = (cell_x[p] + dx + GRID_WIDTH) % GRID_WIDTH
                        py = (cell_y[p] + dy + GRID_HEIGHT) % GRID_HEIGHT
                        cell_x[p] = px
                        cell_y[p] = py
                        grid_cell_id[px, py] = p

                # Only the leader pays movement cost
                cell_energy[i] -= MOVE_COST
                cell_energy[i] = ti.max(0.0, cell_energy[i])


def process_bonded_movement():
    """Execute full bonded group movement: clear, propose, resolve."""
    clear_bonded_move_intents()
    process_bonded_movement_phase1()
    process_bonded_movement_phase2()


def process_bond():
    """Execute full two-phase bonding."""
    clear_bond_claims()
    process_bond_phase1()
    process_bond_phase2()
