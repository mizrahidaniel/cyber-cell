"""Tests for genome table and neural network evaluation."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import taichi as ti
import numpy as np


def test_network_outputs_in_range():
    """Network outputs should be in [0, 1] (sigmoid)."""
    from cell.cell_state import init_cell_state, cell_alive, cell_genome_id
    from cell.genome import (
        init_genome_table, evaluate_all_networks, sensory_inputs,
        action_outputs, genome_weights, GENOME_SIZE,
    )
    from config import NUM_INPUTS, NUM_OUTPUTS, MAX_CELLS

    init_cell_state()
    init_genome_table(count=1)

    # Set up one cell
    cell_alive[0] = 1
    cell_genome_id[0] = 0

    # Random sensory inputs
    rng = np.random.default_rng(42)
    for s in range(NUM_INPUTS):
        sensory_inputs[0, s] = float(rng.uniform(-1, 1))

    evaluate_all_networks()

    for o in range(NUM_OUTPUTS):
        val = action_outputs[0, o]
        assert 0.0 <= val <= 1.0, f"Output {o} out of range: {val}"


def test_zero_weights_produce_half():
    """With all-zero weights, sigmoid(0) = 0.5 for all outputs."""
    from cell.cell_state import init_cell_state, cell_alive, cell_genome_id
    from cell.genome import (
        evaluate_all_networks, sensory_inputs, action_outputs,
        genome_weights, GENOME_SIZE,
    )
    from config import NUM_INPUTS, NUM_OUTPUTS, MAX_GENOMES

    init_cell_state()

    # Zero out genome 0
    zeros = np.zeros((MAX_GENOMES, GENOME_SIZE), dtype=np.float32)
    genome_weights.from_numpy(zeros)

    cell_alive[0] = 1
    cell_genome_id[0] = 0

    # Arbitrary inputs
    for s in range(NUM_INPUTS):
        sensory_inputs[0, s] = 0.5

    evaluate_all_networks()

    for o in range(NUM_OUTPUTS):
        val = action_outputs[0, o]
        assert abs(val - 0.5) < 0.01, f"Output {o} should be ~0.5, got {val}"


def test_near_zero_weights_cluster():
    """Near-zero weights (sigma=0.01) should produce outputs near 0.5."""
    from cell.cell_state import init_cell_state, cell_alive, cell_genome_id
    from cell.genome import (
        init_genome_table, evaluate_all_networks, sensory_inputs,
        action_outputs,
    )
    from config import NUM_INPUTS, NUM_OUTPUTS

    init_cell_state()
    init_genome_table(count=1)

    cell_alive[0] = 1
    cell_genome_id[0] = 0

    for s in range(NUM_INPUTS):
        sensory_inputs[0, s] = 0.5

    evaluate_all_networks()

    # Expected approximate outputs with biased initial genomes:
    # move(0): bias=0.0→~0.50, divide(5): bias=0.5→~0.62, attack(8): bias=-0.3→~0.43
    # Others: bias=0→~0.5
    expected_approx = {0: 0.50, 5: 0.62, 8: 0.43}
    for o in range(NUM_OUTPUTS):
        val = action_outputs[0, o]
        target = expected_approx.get(o, 0.5)
        assert abs(val - target) < 0.15, \
            f"Output {o} unexpected: expected ~{target}, got {val}"


def test_mutation_produces_valid_genome():
    """Mutation should produce a genome that evaluates without error."""
    from cell.cell_state import init_cell_state, cell_alive, cell_genome_id
    from cell.genome import (
        init_genome_table, process_mutations, needs_mutation,
        genome_weights, GENOME_SIZE,
    )
    from config import MAX_GENOMES

    init_cell_state()
    init_genome_table(count=2)

    # Set up cell 1 as a newly-born cell needing mutation
    cell_alive[1] = 1
    cell_genome_id[1] = 0
    needs_mutation[1] = 1

    process_mutations()

    # Verify the mutated genome has finite weights
    weights = genome_weights.to_numpy()
    gid = cell_genome_id[1]
    assert np.all(np.isfinite(weights[gid])), "Mutation produced non-finite weights"


if __name__ == "__main__":
    ti.init(arch=ti.cpu, random_seed=42)
    test_network_outputs_in_range()
    print("PASS: network_outputs_in_range")
    test_zero_weights_produce_half()
    print("PASS: zero_weights_produce_half")
    test_near_zero_weights_cluster()
    print("PASS: near_zero_weights_cluster")
    test_mutation_produces_valid_genome()
    print("PASS: mutation_produces_valid_genome")
    print("\nAll genome tests passed!")
