"""Tests for predation, bonding, and new sensory inputs."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import taichi as ti
import numpy as np


def _place_cell(slot, x, y, energy=50.0, membrane=100.0, genome_id=0, facing=0):
    """Helper to place a cell at a given position."""
    from cell.cell_state import (
        cell_alive, cell_x, cell_y, cell_energy, cell_structure, cell_repmat,
        cell_membrane, cell_genome_id, cell_facing, grid_cell_id,
        free_slot_count, cell_count,
    )
    free_slot_count[None] -= 1
    cell_alive[slot] = 1
    cell_x[slot] = x
    cell_y[slot] = y
    cell_energy[slot] = energy
    cell_structure[slot] = 25.0
    cell_repmat[slot] = 5.0
    cell_membrane[slot] = membrane
    cell_genome_id[slot] = genome_id
    cell_facing[slot] = facing
    grid_cell_id[x, y] = slot
    cell_count[None] += 1


def test_attack_damages_membrane():
    """Attack should reduce target's membrane by ATTACK_MEMBRANE_DAMAGE."""
    from cell.cell_state import init_cell_state, cell_membrane, cell_energy
    from cell.genome import init_genome_table, action_outputs
    from cell.actions import process_attack
    from config import ATTACK_MEMBRANE_DAMAGE, ATTACK_COST

    init_cell_state()
    init_genome_table(count=2)

    # Attacker at (100,100) facing right, target at (101,100)
    _place_cell(0, 100, 100, energy=50.0, facing=1, genome_id=0)
    _place_cell(1, 101, 100, energy=30.0, facing=0, genome_id=1)

    action_outputs[0, 8] = 0.9  # attacker fires attack

    initial_membrane = float(cell_membrane[1])
    initial_energy = float(cell_energy[0])
    process_attack()

    assert abs(cell_membrane[1] - (initial_membrane - ATTACK_MEMBRANE_DAMAGE)) < 0.01, \
        f"Target membrane should drop by {ATTACK_MEMBRANE_DAMAGE}"
    assert abs(cell_energy[0] - (initial_energy - ATTACK_COST)) < 0.01, \
        f"Attacker should pay {ATTACK_COST} energy"


def test_death_spills_chemicals():
    """Dying cell should spill internal chemicals to environment."""
    from cell.cell_state import init_cell_state, cell_membrane, cell_energy, cell_structure
    from cell.genome import init_genome_table
    from world.chemistry import env_S_a, env_R_a, init_chemistry
    from cell.lifecycle import check_death

    init_cell_state()
    init_genome_table(count=1)
    init_chemistry()

    _place_cell(0, 200, 200, energy=40.0, membrane=0.0)  # membrane=0 -> dies immediately
    cell_structure[0] = 20.0

    # Record environment before death
    env_s_before = float(env_S_a[200, 200])

    check_death(env_S_a, env_R_a, env_S_a)  # env_G doesn't matter for this test

    env_s_after = float(env_S_a[200, 200])
    # Should have spilled: structure(20) + energy*0.5(20) = 40
    spilled = env_s_after - env_s_before
    assert spilled > 30.0, f"Expected significant chemical spillage, got {spilled}"


def test_bond_formation_mutual():
    """Two cells facing each other with bond output should form a bond."""
    from cell.cell_state import init_cell_state, cell_bonds
    from cell.genome import init_genome_table, action_outputs
    from cell.bonding import process_bond

    init_cell_state()
    init_genome_table(count=2)

    # Cell 0 at (100,100) facing right, cell 1 at (101,100) facing left
    _place_cell(0, 100, 100, facing=1, genome_id=0)
    _place_cell(1, 101, 100, facing=3, genome_id=1)

    # Both fire bond
    action_outputs[0, 6] = 0.9
    action_outputs[1, 6] = 0.9

    process_bond()

    # Check bonds formed
    bonded_0 = False
    bonded_1 = False
    for b in range(4):
        if cell_bonds[0, b] == 1:
            bonded_0 = True
        if cell_bonds[1, b] == 0:
            bonded_1 = True

    assert bonded_0 and bonded_1, "Mutual bond should form between both cells"


def test_unbond_unilateral():
    """One cell should be able to break all bonds unilaterally."""
    from cell.cell_state import init_cell_state, cell_bonds
    from cell.genome import init_genome_table, action_outputs
    from cell.bonding import process_bond, process_unbond

    init_cell_state()
    init_genome_table(count=2)

    _place_cell(0, 100, 100, facing=1, genome_id=0)
    _place_cell(1, 101, 100, facing=3, genome_id=1)

    # Form bond first
    action_outputs[0, 6] = 0.9
    action_outputs[1, 6] = 0.9
    process_bond()

    # Cell 0 unbonds
    action_outputs[0, 6] = 0.0  # stop bonding
    action_outputs[0, 7] = 0.9  # fire unbond
    action_outputs[1, 7] = 0.0  # cell 1 does NOT unbond

    process_unbond()

    # Both should have no bonds
    for b in range(4):
        assert cell_bonds[0, b] == -1, "Cell 0 should have no bonds after unbond"
        assert cell_bonds[1, b] == -1, "Cell 1's bond to cell 0 should also be cleared"


def test_bond_sharing_equalizes():
    """Bonded cells should share chemicals toward equilibrium."""
    from cell.cell_state import init_cell_state, cell_bonds, cell_energy
    from cell.genome import init_genome_table, action_outputs
    from cell.bonding import process_bond, process_bond_sharing

    init_cell_state()
    init_genome_table(count=2)

    _place_cell(0, 100, 100, energy=100.0, facing=1, genome_id=0)
    _place_cell(1, 101, 100, energy=20.0, facing=3, genome_id=1)

    # Form bond
    action_outputs[0, 6] = 0.9
    action_outputs[1, 6] = 0.9
    process_bond()

    e0_before = float(cell_energy[0])
    e1_before = float(cell_energy[1])

    process_bond_sharing()

    e0_after = float(cell_energy[0])
    e1_after = float(cell_energy[1])

    # Cell 0 should have lost energy, cell 1 gained (minus bond cost)
    assert e0_after < e0_before, "Rich cell should lose energy through sharing"
    assert e1_after > e1_before - 1.0, "Poor cell should gain energy through sharing (minus bond cost)"


def test_bonded_cells_move_together():
    """Bonded pair should move together as a unit when leader fires move."""
    from cell.cell_state import init_cell_state, cell_bonds, cell_x, cell_y, grid_cell_id
    from cell.genome import init_genome_table, action_outputs
    from cell.bonding import process_bond, process_bonded_movement
    from cell.actions import clear_intentions, process_movement_phase1, process_movement_phase2

    init_cell_state()
    init_genome_table(count=2)

    # Cell 0 at (100,100) facing right, cell 1 at (101,100)
    _place_cell(0, 100, 100, facing=1, genome_id=0)
    _place_cell(1, 101, 100, facing=3, genome_id=1)

    # Form bond
    action_outputs[0, 6] = 0.9
    action_outputs[1, 6] = 0.9
    process_bond()

    # Cell 0 tries to move forward (right)
    action_outputs[0, 0] = 0.9
    action_outputs[0, 6] = 0.0
    action_outputs[1, 6] = 0.0

    clear_intentions()
    process_movement_phase1()
    process_movement_phase2()
    process_bonded_movement()

    assert cell_x[0] == 101, f"Leader should move right: expected 101, got {cell_x[0]}"
    assert cell_x[1] == 102, f"Partner should move right: expected 102, got {cell_x[1]}"
    assert grid_cell_id[100, 100] == -1, "Old leader position should be cleared"
    assert grid_cell_id[101, 100] == 0, "Leader should be at new position"
    assert grid_cell_id[102, 100] == 1, "Partner should be at new position"


def test_bonded_group_blocked_when_cant_fit():
    """Bonded pair should NOT move when partner's target is blocked."""
    from cell.cell_state import init_cell_state, cell_bonds, cell_x
    from cell.genome import init_genome_table, action_outputs
    from cell.bonding import process_bond, process_bonded_movement
    from cell.actions import clear_intentions, process_movement_phase1, process_movement_phase2

    init_cell_state()
    init_genome_table(count=3)

    # Cell 0 at (100,100) facing right, cell 1 at (101,100)
    _place_cell(0, 100, 100, facing=1, genome_id=0)
    _place_cell(1, 101, 100, facing=3, genome_id=1)
    # Blocker at (102,100) — cell 1's target if group moves right
    _place_cell(2, 102, 100, facing=0, genome_id=2)

    # Form bond between 0 and 1
    action_outputs[0, 6] = 0.9
    action_outputs[1, 6] = 0.9
    process_bond()

    # Cell 0 tries to move right
    action_outputs[0, 0] = 0.9
    action_outputs[0, 6] = 0.0
    action_outputs[1, 6] = 0.0

    clear_intentions()
    process_movement_phase1()
    process_movement_phase2()
    process_bonded_movement()

    assert cell_x[0] == 100, f"Leader should NOT move: expected 100, got {cell_x[0]}"
    assert cell_x[1] == 101, f"Partner should NOT move: expected 101, got {cell_x[1]}"


def test_death_cleans_bonds():
    """When a cell dies, its bonds should be cleaned up."""
    from cell.cell_state import init_cell_state, cell_bonds, cell_membrane
    from cell.genome import init_genome_table, action_outputs
    from cell.bonding import process_bond
    from cell.lifecycle import check_death
    from world.chemistry import env_S_a, env_R_a, env_G_a, init_chemistry

    init_cell_state()
    init_genome_table(count=2)
    init_chemistry()

    _place_cell(0, 100, 100, facing=1, genome_id=0)
    _place_cell(1, 101, 100, facing=3, genome_id=1)

    # Form bond
    action_outputs[0, 6] = 0.9
    action_outputs[1, 6] = 0.9
    process_bond()

    # Kill cell 0
    cell_membrane[0] = 0.0
    check_death(env_S_a, env_R_a, env_G_a)

    # Cell 1's bonds should be cleared
    for b in range(4):
        assert cell_bonds[1, b] == -1, "Surviving cell's bond to dead cell should be cleared"


def test_new_sensory_inputs():
    """Inputs 16-17 should report energy and membrane of cell ahead."""
    from cell.cell_state import init_cell_state
    from cell.genome import init_genome_table, sensory_inputs
    from cell.sensing import compute_sensory_inputs
    from world.chemistry import env_S_a, env_R_a, env_G_a, env_W_a, init_chemistry
    from world.grid import init_grid, compute_light

    init_cell_state()
    init_genome_table(count=2)
    init_chemistry()
    init_grid()
    compute_light(0)

    # Cell 0 at (100,100) facing right, cell 1 at (101,100) with known energy/membrane
    _place_cell(0, 100, 100, energy=50.0, facing=1, genome_id=0)
    _place_cell(1, 101, 100, energy=75.0, membrane=80.0, facing=0, genome_id=1)

    compute_sensory_inputs(env_S_a, env_R_a, env_G_a, env_W_a)

    # Input 11: cell ahead should be 1.0
    assert abs(sensory_inputs[0, 11] - 1.0) < 0.01, "Should detect cell ahead"

    # Input 16: energy of cell ahead = min(1.0, 75/100) = 0.75
    assert abs(sensory_inputs[0, 16] - 0.75) < 0.01, \
        f"Prey energy should be 0.75, got {sensory_inputs[0, 16]}"

    # Input 17: membrane of cell ahead = 80/100 = 0.8
    assert abs(sensory_inputs[0, 17] - 0.8) < 0.01, \
        f"Prey membrane should be 0.8, got {sensory_inputs[0, 17]}"

    # Cell 1 faces up (no cell there), inputs 16-17 should be 0
    assert abs(sensory_inputs[1, 16]) < 0.01, "No cell ahead -> prey energy should be 0"
    assert abs(sensory_inputs[1, 17]) < 0.01, "No cell ahead -> prey membrane should be 0"


if __name__ == "__main__":
    ti.init(arch=ti.cpu, random_seed=42)

    tests = [
        ("attack_damages_membrane", test_attack_damages_membrane),
        ("death_spills_chemicals", test_death_spills_chemicals),
        ("bond_formation_mutual", test_bond_formation_mutual),
        ("unbond_unilateral", test_unbond_unilateral),
        ("bond_sharing_equalizes", test_bond_sharing_equalizes),
        ("bonded_cells_move_together", test_bonded_cells_move_together),
        ("bonded_group_blocked_when_cant_fit", test_bonded_group_blocked_when_cant_fit),
        ("death_cleans_bonds", test_death_cleans_bonds),
        ("new_sensory_inputs", test_new_sensory_inputs),
    ]

    for name, test_fn in tests:
        test_fn()
        print(f"PASS: {name}")

    print(f"\nAll {len(tests)} predation tests passed!")
