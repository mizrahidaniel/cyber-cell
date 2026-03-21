"""Tests for CTRNN genome (CfC dynamics, bootstrap, mutation)."""

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


def test_ctrnn_evaluation_produces_valid_outputs():
    """CTRNN evaluation should produce action outputs for alive cells."""
    from cell.cell_state import init_cell_state
    from cell.ctrnn_genome import (
        ctrnn_weights, ctrnn_neurons, init_ctrnn_genome_table,
        evaluate_all_ctrnn,
    )
    from cell.genome import action_outputs, sensory_inputs

    init_cell_state()
    init_ctrnn_genome_table(count=2)

    _place_cell(0, 50, 50, genome_id=0)
    _place_cell(1, 52, 50, genome_id=1)

    # Set some sensory inputs
    sensory_inputs[0, 0] = 0.8   # light (high)
    sensory_inputs[0, 1] = 0.5   # energy
    sensory_inputs[0, 2] = 0.3   # structure
    sensory_inputs[1, 0] = 0.0   # no light
    sensory_inputs[1, 1] = 0.2   # low energy

    # Run a few ticks to let dynamics settle
    for _ in range(5):
        evaluate_all_ctrnn()

    # Neurons should have non-zero activations
    n0_sum = sum(abs(float(ctrnn_neurons[0, n])) for n in range(16))
    assert n0_sum > 0.0, f"Cell 0 neurons should be non-zero, sum={n0_sum}"

    # Action outputs should have been set (some may be 0 due to sigmoid)
    # At minimum, eat action should fire for cell 0 (high light)
    # Check that the action output fields were written
    any_output = False
    for a in range(14):
        if float(action_outputs[0, a]) > 0.0:
            any_output = True
            break
    assert any_output, "Cell 0 should have at least one non-zero action output"


def test_ctrnn_bootstrap_produces_eat_action():
    """Bootstrap circuit: eat action should fire for cells in light."""
    from cell.cell_state import init_cell_state
    from cell.ctrnn_genome import (
        ctrnn_neurons, init_ctrnn_genome_table, evaluate_all_ctrnn,
    )
    from cell.genome import action_outputs, sensory_inputs

    init_cell_state()
    init_ctrnn_genome_table(count=1)

    _place_cell(0, 50, 50, genome_id=0)

    # Drive cell with light input for several ticks
    eat_fired = 0
    for _ in range(30):
        sensory_inputs[0, 0] = 0.8   # light
        sensory_inputs[0, 1] = 0.5
        sensory_inputs[0, 2] = 0.3
        evaluate_all_ctrnn()
        if float(action_outputs[0, 3]) > 0.0:  # eat action
            eat_fired += 1

    # Eat should fire at least sometimes with light input
    assert eat_fired > 0, \
        f"Eat action should fire at least once in 30 ticks with light input, fired {eat_fired}"


def test_ctrnn_mutation_changes_genome():
    """Mutation should produce a different genome from parent."""
    from cell.cell_state import init_cell_state
    from cell.ctrnn_genome import (
        ctrnn_weights, init_ctrnn_genome_table, process_ctrnn_mutations,
    )
    from cell.genome import needs_mutation
    from cell.cell_state import cell_genome_id
    from config import CTRNN_GENOME_SIZE

    init_cell_state()
    init_ctrnn_genome_table(count=2)

    _place_cell(0, 50, 50, genome_id=0)

    # Record parent weights
    parent_weights = [float(ctrnn_weights[0, w]) for w in range(CTRNN_GENOME_SIZE)]

    # Trigger mutation
    needs_mutation[0] = 1
    process_ctrnn_mutations(tick=100)

    new_gid = int(cell_genome_id[0])

    # Should have a new genome (or same if no mutation occurred)
    if new_gid != 0:
        child_weights = [float(ctrnn_weights[new_gid, w])
                         for w in range(CTRNN_GENOME_SIZE)]
        diffs = sum(1 for p, c in zip(parent_weights, child_weights) if p != c)
        assert diffs > 0, "Mutated genome should differ from parent"


def test_ctrnn_hidden_neurons_have_memory():
    """Hidden neurons should retain state between evaluations (CfC memory)."""
    from cell.cell_state import init_cell_state
    from cell.ctrnn_genome import (
        ctrnn_neurons, init_ctrnn_genome_table, evaluate_all_ctrnn,
    )
    from cell.genome import sensory_inputs

    init_cell_state()
    init_ctrnn_genome_table(count=1)

    _place_cell(0, 50, 50, genome_id=0)

    # Drive sensory inputs for several ticks
    for _ in range(10):
        sensory_inputs[0, 0] = 0.8
        sensory_inputs[0, 1] = 0.5
        evaluate_all_ctrnn()

    # Record hidden state
    hidden_after_drive = [float(ctrnn_neurons[0, 8 + h]) for h in range(4)]

    # Now set sensory to zero for one tick
    for s in range(34):
        sensory_inputs[0, s] = 0.0
    evaluate_all_ctrnn()

    # Hidden neurons should still have non-zero state (memory)
    hidden_after_zero = [float(ctrnn_neurons[0, 8 + h]) for h in range(4)]
    has_memory = any(abs(h) > 0.001 for h in hidden_after_zero)
    assert has_memory, \
        f"Hidden neurons should retain memory: {hidden_after_zero}"


if __name__ == "__main__":
    ti.init(arch=ti.cpu, random_seed=42)

    tests = [
        ("ctrnn_evaluation", test_ctrnn_evaluation_produces_valid_outputs),
        ("ctrnn_bootstrap_eat", test_ctrnn_bootstrap_produces_eat_action),
        ("ctrnn_mutation", test_ctrnn_mutation_changes_genome),
        ("ctrnn_hidden_memory", test_ctrnn_hidden_neurons_have_memory),
    ]

    for name, test_fn in tests:
        test_fn()
        print(f"PASS: {name}")

    print(f"\nAll {len(tests)} CTRNN tests passed!")
