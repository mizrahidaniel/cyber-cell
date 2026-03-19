"""Tests for energy model: photosynthesis, metabolism, death, and spillage."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import taichi as ti
import numpy as np


def setup_module():
    ti.init(arch=ti.cpu, random_seed=42)


def _setup_single_cell(x=50, y=50, energy=50.0, structure=25.0, repmat=5.0,
                        membrane=100.0):
    """Helper: initialize state and place a single cell."""
    from cell.cell_state import (
        init_cell_state, cell_alive, cell_x, cell_y, cell_energy,
        cell_structure, cell_repmat, cell_membrane, cell_age,
        cell_genome_id, cell_facing, grid_cell_id, cell_count,
        free_slot_count,
    )
    from world.grid import init_grid
    from world.chemistry import init_chemistry

    init_cell_state()
    init_grid()
    init_chemistry()

    # Manually place one cell at slot 0
    slot = 0
    free_slot_count[None] = free_slot_count[None] - 1
    cell_alive[slot] = 1
    cell_x[slot] = x
    cell_y[slot] = y
    cell_energy[slot] = energy
    cell_structure[slot] = structure
    cell_repmat[slot] = repmat
    cell_membrane[slot] = membrane
    cell_age[slot] = 0
    cell_genome_id[slot] = 0
    cell_facing[slot] = 0
    grid_cell_id[x, y] = slot
    cell_count[None] = 1

    return slot


def test_photosynthesis_in_light():
    """Cell in bright zone at peak daylight gains energy."""
    from cell.cell_state import cell_energy
    from cell.lifecycle import photosynthesis
    from world.grid import compute_light
    from world.chemistry import get_env_S, get_env_R
    from config import DAY_LENGTH, PHOTOSYNTHESIS_RATE

    slot = _setup_single_cell(x=50, y=50, energy=10.0)

    # Peak daylight
    compute_light(DAY_LENGTH // 4)

    initial_energy = cell_energy[slot]
    photosynthesis(get_env_S(), get_env_R())
    gained = cell_energy[slot] - initial_energy

    # Should gain approximately PHOTOSYNTHESIS_RATE * 1.0 (bright zone, peak day)
    assert abs(gained - PHOTOSYNTHESIS_RATE) < 0.01, \
        f"Expected ~{PHOTOSYNTHESIS_RATE} E gain, got {gained}"


def test_photosynthesis_in_dark():
    """Cell in dark zone gains no energy."""
    from cell.cell_state import cell_energy
    from cell.lifecycle import photosynthesis
    from world.grid import compute_light
    from world.chemistry import get_env_S, get_env_R
    from config import DAY_LENGTH, DIM_ZONE_END

    slot = _setup_single_cell(x=DIM_ZONE_END + 50, y=50, energy=10.0)

    compute_light(DAY_LENGTH // 4)

    initial_energy = cell_energy[slot]
    photosynthesis(get_env_S(), get_env_R())
    gained = cell_energy[slot] - initial_energy

    assert abs(gained) < 0.01, f"Dark zone cell gained energy: {gained}"


def test_metabolism_drains_energy():
    """Metabolism should drain at the expected rate."""
    from cell.cell_state import cell_energy
    from cell.lifecycle import apply_metabolism
    from config import BASAL_METABOLISM, E_DECAY_FLAT, NETWORK_COST

    slot = _setup_single_cell(energy=50.0)

    initial = cell_energy[slot]
    apply_metabolism()
    drained = initial - cell_energy[slot]

    expected = BASAL_METABOLISM + E_DECAY_FLAT + NETWORK_COST
    assert abs(drained - expected) < 0.001, \
        f"Expected drain {expected}, got {drained}"


def test_energy_never_negative():
    """Energy should be clamped to 0."""
    from cell.cell_state import cell_energy
    from cell.lifecycle import apply_metabolism

    slot = _setup_single_cell(energy=0.01)
    apply_metabolism()

    assert cell_energy[slot] >= 0.0, f"Energy went negative: {cell_energy[slot]}"


def test_death_at_zero_membrane():
    """Cell dies when membrane reaches 0, chemicals spill."""
    from cell.cell_state import cell_alive, cell_energy, cell_structure, grid_cell_id, cell_count
    from cell.lifecycle import check_death
    from world.chemistry import get_env_S, get_env_R, get_env_G

    x, y = 80, 80
    slot = _setup_single_cell(x=x, y=y, energy=10.0, structure=20.0, membrane=0.0)

    env_S = get_env_S()
    initial_env_S = float(env_S[x, y])

    check_death(get_env_S(), get_env_R(), get_env_G())

    assert cell_alive[slot] == 0, "Cell should be dead"
    assert grid_cell_id[x, y] == -1, "Grid should be clear"
    assert cell_count[None] == 0, "Population count should be 0"

    # Check spillage: structure + 0.5*energy should have been added to env_S
    expected_spill = 20.0 + 10.0 * 0.5
    actual_spill = float(get_env_S()[x, y]) - initial_env_S
    assert abs(actual_spill - expected_spill) < 0.1, \
        f"Expected ~{expected_spill} S spillage, got {actual_spill}"


def test_membrane_damage_at_zero_energy():
    """Cell with 0 energy takes membrane damage each tick."""
    from cell.cell_state import cell_membrane
    from cell.lifecycle import apply_metabolism
    from config import ENERGY_ZERO_MEMBRANE_DAMAGE

    slot = _setup_single_cell(energy=0.0, membrane=100.0)

    apply_metabolism()

    expected = 100.0 - ENERGY_ZERO_MEMBRANE_DAMAGE
    assert abs(cell_membrane[slot] - expected) < 0.1, \
        f"Expected membrane {expected}, got {cell_membrane[slot]}"


if __name__ == "__main__":
    setup_module()
    test_photosynthesis_in_light()
    print("PASS: photosynthesis_in_light")
    test_photosynthesis_in_dark()
    print("PASS: photosynthesis_in_dark")
    test_metabolism_drains_energy()
    print("PASS: metabolism_drains_energy")
    test_energy_never_negative()
    print("PASS: energy_never_negative")
    test_death_at_zero_membrane()
    print("PASS: death_at_zero_membrane")
    test_membrane_damage_at_zero_energy()
    print("PASS: membrane_damage_at_zero_energy")
    print("\nAll energy tests passed!")
