"""Tests for v7.2 bond-waste mechanics."""

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


def test_waste_equalization_between_bonded_pair():
    """Waste exposure should converge between bonded cells."""
    from cell.cell_state import (
        init_cell_state, cell_bonds, cell_bond_strength, cell_waste_exposure,
    )
    from cell.genome import init_genome_table, action_outputs
    from cell.bonding import process_bond, process_bond_waste_equalization

    init_cell_state()
    init_genome_table(count=2)

    _place_cell(0, 100, 100, facing=1, genome_id=0)
    _place_cell(1, 101, 100, facing=3, genome_id=1)

    # Form bond
    action_outputs[0, 6] = 0.9
    action_outputs[1, 6] = 0.9
    process_bond()

    # Set different waste exposure
    cell_waste_exposure[0] = 0.8
    cell_waste_exposure[1] = 0.2

    # Run equalization several times
    for _ in range(10):
        process_bond_waste_equalization()

    w0 = float(cell_waste_exposure[0])
    w1 = float(cell_waste_exposure[1])
    assert abs(w0 - w1) < 0.1, f"Waste should converge: {w0:.3f} vs {w1:.3f}"
    assert 0.3 < w0 < 0.7, f"Average should be near 0.5: {w0:.3f}"


def test_metabolic_efficiency_reduces_waste():
    """Bonded cells should produce less waste during photosynthesis."""
    from cell.cell_state import (
        init_cell_state, cell_bonds, cell_bond_strength,
    )
    from cell.genome import init_genome_table
    from world.chemistry import init_chemistry, get_env_S, get_env_R, _get_dst_W
    from world.grid import init_grid, compute_light
    from cell.lifecycle import photosynthesis

    init_cell_state()
    init_genome_table(count=3)
    init_chemistry()
    init_grid()
    compute_light(250)  # tick 250 = peak daylight (sin(pi/2)=1.0)

    # Solo cell
    _place_cell(0, 50, 200, genome_id=0)
    # Bonded pair
    _place_cell(1, 50, 202, facing=1, genome_id=1)
    _place_cell(2, 51, 202, facing=3, genome_id=2)

    # Manual bond between cells 1 and 2
    cell_bonds[1, 0] = 2
    cell_bonds[2, 0] = 1
    cell_bond_strength[1, 0] = 0.5
    cell_bond_strength[2, 0] = 0.5

    env_S = get_env_S()
    env_R = get_env_R()
    env_W = _get_dst_W()

    # Run photosynthesis
    photosynthesis(env_S, env_R, env_W)

    solo_waste = float(env_W[50, 200])
    bonded_waste = float(env_W[50, 202])

    # Bonded cell should produce less waste
    assert bonded_waste < solo_waste, (
        f"Bonded cell waste ({bonded_waste:.4f}) should be less than "
        f"solo cell waste ({solo_waste:.4f})")


def test_syntrophy_converts_waste_to_energy():
    """Syntrophy should convert waste above threshold to energy."""
    from cell.cell_state import init_cell_state, cell_energy
    from cell.genome import init_genome_table
    from world.chemistry import init_chemistry, _get_dst_W
    from cell.lifecycle import syntrophy

    init_cell_state()
    init_genome_table(count=1)
    init_chemistry()

    _place_cell(0, 100, 100, energy=10.0, genome_id=0)

    env_W = _get_dst_W()
    env_W[100, 100] = 0.6  # Above threshold

    e_before = float(cell_energy[0])
    syntrophy(env_W)
    e_after = float(cell_energy[0])

    assert e_after > e_before, f"Energy should increase: {e_before:.3f} -> {e_after:.3f}"
    assert float(env_W[100, 100]) < 0.6, "Waste should decrease after syntrophy"


def test_death_cause_classification():
    """Death causes should be correctly classified into 4 categories."""
    from cell.cell_state import (
        init_cell_state, cell_membrane, cell_energy, cell_age, cell_last_attacker,
    )
    from cell.genome import init_genome_table
    from world.chemistry import init_chemistry, env_S_a, env_R_a, env_G_a
    from cell.lifecycle import (
        check_death, deaths_by_starvation, deaths_by_age,
        deaths_by_waste, deaths_by_predation, reset_death_counters,
    )
    from config import MAX_CELL_AGE

    init_cell_state()
    init_genome_table(count=5)
    init_chemistry()
    reset_death_counters()

    # Cell 0: starvation (no energy, membrane=0)
    _place_cell(0, 100, 100, energy=0.0, membrane=0.0, genome_id=0)
    # Cell 1: age (past max age, has energy, membrane=0)
    _place_cell(1, 102, 100, energy=20.0, membrane=0.0, genome_id=1)
    cell_age[1] = MAX_CELL_AGE + 100
    # Cell 2: waste (has energy, not old, no attacker, membrane=0)
    _place_cell(2, 104, 100, energy=20.0, membrane=0.0, genome_id=2)
    # Cell 3: predation (has valid living attacker, membrane=0)
    _place_cell(3, 106, 100, energy=20.0, membrane=0.0, genome_id=3)
    _place_cell(4, 108, 100, energy=50.0, membrane=100.0, genome_id=0)  # attacker
    cell_last_attacker[3] = 4

    check_death(env_S_a, env_R_a, env_G_a)

    assert deaths_by_starvation[None] == 1, \
        f"Expected 1 starvation, got {deaths_by_starvation[None]}"
    assert deaths_by_age[None] == 1, \
        f"Expected 1 age death, got {deaths_by_age[None]}"
    assert deaths_by_waste[None] == 1, \
        f"Expected 1 waste death, got {deaths_by_waste[None]}"
    assert deaths_by_predation[None] == 1, \
        f"Expected 1 predation death, got {deaths_by_predation[None]}"


def test_environmental_predation_respects_bonds():
    """Environmental predation kills solo cells but spares clustered cells.

    Uses default config (SOLO_KILL_PROB=0.02, IMMUNE_BONDS=2).
    Run predation 500 times: solo cell dies with >99.99% probability,
    cluster cell (2+ bonds) is deterministically immune.
    """
    from cell.cell_state import (
        init_cell_state, cell_bonds, cell_bond_strength, cell_membrane,
    )
    from cell.genome import init_genome_table
    from cell.lifecycle import apply_environmental_predation

    init_cell_state()
    init_genome_table(count=4)

    # Solo cell (0 bonds) — should eventually be killed
    _place_cell(0, 100, 100, membrane=100.0, genome_id=0)

    # Cluster cell (2+ bonds) — should be immune
    _place_cell(1, 120, 100, membrane=100.0, genome_id=1)
    _place_cell(2, 121, 100, membrane=100.0, genome_id=2)
    _place_cell(3, 122, 100, membrane=100.0, genome_id=3)
    cell_bonds[1, 0] = 2
    cell_bonds[1, 1] = 3
    cell_bond_strength[1, 0] = 0.5
    cell_bond_strength[1, 1] = 0.5
    cell_bonds[2, 0] = 1
    cell_bond_strength[2, 0] = 0.5
    cell_bonds[3, 0] = 1
    cell_bond_strength[3, 0] = 0.5

    # Run predation many times
    for _ in range(500):
        apply_environmental_predation()

    assert float(cell_membrane[0]) == 0.0, \
        f"Solo cell should be killed by predation, membrane={float(cell_membrane[0])}"
    assert float(cell_membrane[1]) == 100.0, \
        f"Cluster cell (2+ bonds) should be immune, membrane={float(cell_membrane[1])}"


def test_env_predation_death_classified_as_predation():
    """Cell killed by environmental predation (sentinel -2) should be classified as predation."""
    from cell.cell_state import (
        init_cell_state, cell_membrane, cell_energy, cell_last_attacker,
    )
    from cell.genome import init_genome_table
    from world.chemistry import init_chemistry, env_S_a, env_R_a, env_G_a
    from cell.lifecycle import (
        check_death, deaths_by_predation, reset_death_counters,
    )

    init_cell_state()
    init_genome_table(count=1)
    init_chemistry()
    reset_death_counters()

    # Cell with membrane=0 and sentinel -2 (environmental predation)
    _place_cell(0, 100, 100, energy=20.0, membrane=0.0, genome_id=0)
    cell_last_attacker[0] = -2

    check_death(env_S_a, env_R_a, env_G_a)

    assert deaths_by_predation[None] == 1, \
        f"Expected 1 predation death, got {deaths_by_predation[None]}"


def test_bond_signal_emission_from_hidden_chemicals():
    """CRN hidden chemicals 10-11 above threshold should emit bond signals to partner."""
    from cell.cell_state import (
        init_cell_state, cell_bonds, cell_bond_strength,
        cell_bond_signal_in,
    )
    from cell.genome import action_outputs, sensory_inputs
    from cell.crn_genome import (
        crn_weights, crn_chemicals, init_crn_genome_table, evaluate_all_crns,
    )
    from cell.actions import process_bond_signal_output
    from cell.bonding import process_bond_signal_relay
    from config import MAX_REACTIONS, CRN_PARAMS_PER_REACTION

    init_cell_state()
    init_crn_genome_table(count=2)

    _place_cell(0, 100, 100, facing=1, genome_id=0)
    _place_cell(1, 101, 100, facing=3, genome_id=1)

    # Form bond manually
    cell_bonds[0, 0] = 1
    cell_bonds[1, 0] = 0
    cell_bond_strength[0, 0] = 0.5
    cell_bond_strength[1, 0] = 0.5

    # Zero all reaction rates so hidden chemicals aren't perturbed
    for gid in range(2):
        for r in range(MAX_REACTIONS):
            crn_weights[gid, r * CRN_PARAMS_PER_REACTION + 5] = 0.0

    # Set hidden chemicals 10-11 high for cell 0, low for cell 1
    crn_chemicals[0, 10] = 2.0
    crn_chemicals[0, 11] = 1.5
    crn_chemicals[1, 10] = 0.1
    crn_chemicals[1, 11] = 0.1

    # Clear sensory inputs
    for s in range(34):
        sensory_inputs[0, s] = 0.0
        sensory_inputs[1, s] = 0.0

    evaluate_all_crns()

    # Cell 0: hidden chems > 0.5 → bond signal outputs should be non-zero
    assert float(action_outputs[0, 10]) > 0.0, \
        f"Cell 0 bond signal ch0 should be non-zero: {float(action_outputs[0, 10])}"
    assert float(action_outputs[0, 12]) > 0.0, \
        f"Cell 0 bond signal ch2 should be non-zero: {float(action_outputs[0, 12])}"

    # Cell 1: hidden chems < 0.5 → bond signal outputs should be zero
    assert float(action_outputs[1, 10]) == 0.0, \
        f"Cell 1 bond signal ch0 should be zero: {float(action_outputs[1, 10])}"
    assert float(action_outputs[1, 12]) == 0.0, \
        f"Cell 1 bond signal ch2 should be zero: {float(action_outputs[1, 12])}"

    # Run bond signal pipeline: output → relay
    process_bond_signal_output()
    process_bond_signal_relay()

    # Cell 1 should receive cell 0's bond signals through bond slot 0
    sig_ch0 = float(cell_bond_signal_in[1, 0, 0])
    sig_ch2 = float(cell_bond_signal_in[1, 0, 2])
    assert sig_ch0 > 0.0, f"Cell 1 should receive signal ch0 from cell 0: {sig_ch0}"
    assert sig_ch2 > 0.0, f"Cell 1 should receive signal ch2 from cell 0: {sig_ch2}"

    # -- Bond signal RECEPTION: verify hidden chemicals increase on next eval --
    # Record cell 1's hidden chemicals before re-evaluation
    h8_before = float(crn_chemicals[1, 8])
    h9_before = float(crn_chemicals[1, 9])

    # Set up sensory_inputs for cell 1 with the received bond signals
    # (compute_sensory_inputs would do this, but we set manually for unit test)
    sensory_inputs[1, 18] = sig_ch0  # bond 0, channel 0 -> hidden 8
    sensory_inputs[1, 19] = float(cell_bond_signal_in[1, 0, 1])  # ch 1 -> hidden 9
    sensory_inputs[1, 20] = sig_ch2  # bond 0, channel 2 -> hidden 10
    sensory_inputs[1, 21] = float(cell_bond_signal_in[1, 0, 3])  # ch 3 -> hidden 11

    evaluate_all_crns()

    h8_after = float(crn_chemicals[1, 8])
    h9_after = float(crn_chemicals[1, 9])
    assert h8_after > h8_before, \
        f"Cell 1 hidden[8] should increase from bond signal reception: {h8_before:.4f} -> {h8_after:.4f}"


if __name__ == "__main__":
    ti.init(arch=ti.cpu, random_seed=42)

    tests = [
        ("waste_equalization", test_waste_equalization_between_bonded_pair),
        ("metabolic_efficiency", test_metabolic_efficiency_reduces_waste),
        ("syntrophy", test_syntrophy_converts_waste_to_energy),
        ("death_cause_classification", test_death_cause_classification),
        ("env_predation_respects_bonds", test_environmental_predation_respects_bonds),
        ("env_predation_death_classification", test_env_predation_death_classified_as_predation),
        ("bond_signal_emission", test_bond_signal_emission_from_hidden_chemicals),
    ]

    for name, test_fn in tests:
        test_fn()
        print(f"PASS: {name}")

    print(f"\nAll {len(tests)} bond-waste tests passed!")
