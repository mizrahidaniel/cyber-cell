"""Tests for cell division, reproduction, and genome management."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import taichi as ti
import numpy as np


def test_division_requires_resources():
    """Division should fail without sufficient energy and replication material."""
    from cell.cell_state import (
        init_cell_state, cell_alive, cell_x, cell_y, cell_energy,
        cell_repmat, cell_membrane, cell_genome_id, cell_facing,
        grid_cell_id, cell_count, free_slot_count,
    )
    from cell.genome import init_genome_table, action_outputs, needs_mutation
    from cell.actions import process_divide_phase1, process_divide_phase2, clear_intentions
    from config import DIVIDE_COST, DIVIDE_R_COST

    init_cell_state()
    init_genome_table(count=1)

    # Place a cell with insufficient energy
    slot = 0
    free_slot_count[None] -= 1
    cell_alive[slot] = 1
    cell_x[slot] = 100
    cell_y[slot] = 100
    cell_energy[slot] = DIVIDE_COST - 1  # not enough
    cell_repmat[slot] = DIVIDE_R_COST + 10
    cell_membrane[slot] = 100.0
    cell_genome_id[slot] = 0
    cell_facing[slot] = 0
    grid_cell_id[100, 100] = slot
    cell_count[None] = 1

    # Force divide output above threshold
    action_outputs[slot, 5] = 0.9

    clear_intentions()
    process_divide_phase1()
    process_divide_phase2()

    assert cell_count[None] == 1, "Division should have failed (insufficient energy)"


def test_division_splits_resources():
    """Successful division should correctly split resources."""
    from cell.cell_state import (
        init_cell_state, cell_alive, cell_x, cell_y, cell_energy,
        cell_structure, cell_repmat, cell_membrane, cell_genome_id,
        cell_facing, grid_cell_id, cell_count, free_slot_count,
    )
    from cell.genome import init_genome_table, action_outputs, needs_mutation
    from cell.actions import process_divide_phase1, process_divide_phase2, clear_intentions
    from config import DIVIDE_COST, DIVIDE_R_COST, PARENT_RESOURCE_SHARE, DAUGHTER_RESOURCE_SHARE

    init_cell_state()
    init_genome_table(count=1)

    slot = 0
    free_slot_count[None] -= 1
    cell_alive[slot] = 1
    cell_x[slot] = 100
    cell_y[slot] = 100
    cell_energy[slot] = 50.0
    cell_structure[slot] = 30.0
    cell_repmat[slot] = 15.0
    cell_membrane[slot] = 100.0
    cell_genome_id[slot] = 0
    cell_facing[slot] = 0
    grid_cell_id[100, 100] = slot
    cell_count[None] = 1

    action_outputs[slot, 5] = 0.9

    clear_intentions()
    process_divide_phase1()
    process_divide_phase2()

    assert cell_count[None] == 2, f"Expected 2 cells, got {cell_count[None]}"

    # Parent energy: 50 - DIVIDE_COST = 30, then * PARENT_RESOURCE_SHARE = 18
    expected_parent_e = (50.0 - DIVIDE_COST) * PARENT_RESOURCE_SHARE
    assert abs(cell_energy[slot] - expected_parent_e) < 0.1, \
        f"Parent energy: expected {expected_parent_e}, got {cell_energy[slot]}"


def test_division_fails_when_surrounded():
    """Division should fail if all 4 neighbors are occupied."""
    from cell.cell_state import (
        init_cell_state, cell_alive, cell_x, cell_y, cell_energy,
        cell_repmat, cell_membrane, cell_genome_id, cell_facing,
        grid_cell_id, cell_count, free_slot_count,
    )
    from cell.genome import init_genome_table, action_outputs
    from cell.actions import process_divide_phase1, process_divide_phase2, clear_intentions

    init_cell_state()
    init_genome_table(count=5)

    # Place center cell and all 4 neighbors
    positions = [(100, 100), (100, 101), (101, 100), (100, 99), (99, 100)]
    for idx, (x, y) in enumerate(positions):
        free_slot_count[None] -= 1
        cell_alive[idx] = 1
        cell_x[idx] = x
        cell_y[idx] = y
        cell_energy[idx] = 50.0
        cell_repmat[idx] = 15.0
        cell_membrane[idx] = 100.0
        cell_genome_id[idx] = idx
        cell_facing[idx] = 0
        grid_cell_id[x, y] = idx
    cell_count[None] = 5

    action_outputs[0, 5] = 0.9  # center cell tries to divide

    clear_intentions()
    process_divide_phase1()
    process_divide_phase2()

    assert cell_count[None] == 5, "Division should fail when surrounded"


if __name__ == "__main__":
    ti.init(arch=ti.cpu, random_seed=42)
    test_division_requires_resources()
    print("PASS: division_requires_resources")
    test_division_splits_resources()
    print("PASS: division_splits_resources")
    test_division_fails_when_surrounded()
    print("PASS: division_fails_when_surrounded")
    print("\nAll lifecycle tests passed!")
