"""Bonding system: mutual bond formation, unilateral unbonding, and chemical sharing."""

import taichi as ti

from config import (
    MAX_CELLS, GRID_WIDTH, GRID_HEIGHT, ACTION_THRESHOLD,
    BOND_COST, BOND_SHARE_RATE,
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
        grid_bond_claim[i, j] = -1


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
                if claimer >= 0 and claimer != i and cell_alive[claimer] == 1:
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


def process_bond():
    """Execute full two-phase bonding."""
    clear_bond_claims()
    process_bond_phase1()
    process_bond_phase2()
