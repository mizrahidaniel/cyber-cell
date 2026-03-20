"""Tests for chemistry diffusion, decay, and boundary conditions."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import taichi as ti
import numpy as np


def _total(field):
    """Sum all values in a Taichi field."""
    return float(field.to_numpy().sum())


def test_diffusion_conserves_mass_without_decay():
    """Diffusion with zero decay should conserve total mass."""
    from world.chemistry import _diffuse_and_decay, env_S_a, env_S_b
    from config import GRID_WIDTH, GRID_HEIGHT

    # Place a blob of chemical at the center
    arr = np.zeros((GRID_WIDTH, GRID_HEIGHT), dtype=np.float32)
    arr[250, 250] = 100.0
    env_S_a.from_numpy(arr)
    env_S_b.from_numpy(np.zeros_like(arr))

    initial_total = _total(env_S_a)

    # Diffuse with zero decay
    _diffuse_and_decay(env_S_a, env_S_b, 0.01, 0.0)
    after_total = _total(env_S_b)

    assert abs(after_total - initial_total) < 1e-3, \
        f"Mass not conserved: {initial_total} -> {after_total}"


def test_decay_reduces_total():
    """Decay should reduce total chemical mass."""
    from world.chemistry import _diffuse_and_decay, env_S_a, env_S_b
    from config import GRID_WIDTH, GRID_HEIGHT

    arr = np.zeros((GRID_WIDTH, GRID_HEIGHT), dtype=np.float32)
    arr[100:200, 100:200] = 1.0
    env_S_a.from_numpy(arr)
    env_S_b.from_numpy(np.zeros_like(arr))

    initial_total = _total(env_S_a)
    decay_rate = 0.05

    _diffuse_and_decay(env_S_a, env_S_b, 0.01, decay_rate)
    after_total = _total(env_S_b)

    # Total should decrease by approximately the decay rate
    expected = initial_total * (1.0 - decay_rate)
    assert after_total < initial_total, "Decay did not reduce total"
    assert abs(after_total - expected) / expected < 0.01, \
        f"Decay amount unexpected: expected ~{expected}, got {after_total}"


def test_toroidal_wrapping():
    """Chemical at edge (0,0) should diffuse to the opposite edge."""
    from world.chemistry import _diffuse_and_decay, env_S_a, env_S_b
    from config import GRID_WIDTH, GRID_HEIGHT

    arr = np.zeros((GRID_WIDTH, GRID_HEIGHT), dtype=np.float32)
    arr[0, 0] = 100.0
    env_S_a.from_numpy(arr)
    env_S_b.from_numpy(np.zeros_like(arr))

    _diffuse_and_decay(env_S_a, env_S_b, 0.2, 0.0)
    result = env_S_b.to_numpy()

    # Chemical should have spread to the wrapped neighbors
    assert result[GRID_WIDTH - 1, 0] > 0, "No wrapping on x-axis (left)"
    assert result[0, GRID_HEIGHT - 1] > 0, "No wrapping on y-axis (top)"
    assert result[1, 0] > 0, "No spread to right neighbor"
    assert result[0, 1] > 0, "No spread to bottom neighbor"


def test_no_negative_values():
    """Chemical concentrations should never go negative."""
    from world.chemistry import _diffuse_and_decay, env_S_a, env_S_b
    from config import GRID_WIDTH, GRID_HEIGHT

    # Very small values with high decay
    arr = np.full((GRID_WIDTH, GRID_HEIGHT), 1e-8, dtype=np.float32)
    env_S_a.from_numpy(arr)
    env_S_b.from_numpy(np.zeros_like(arr))

    for _ in range(100):
        _diffuse_and_decay(env_S_a, env_S_b, 0.3, 0.1)
        _diffuse_and_decay(env_S_b, env_S_a, 0.3, 0.1)

    result = env_S_a.to_numpy()
    assert np.all(result >= 0), f"Negative values found: min = {result.min()}"


def test_light_zones():
    """Light field should have correct zone intensities during day."""
    from world.grid import compute_light, light_field
    from config import LIGHT_ZONE_END, DIM_ZONE_END, DAY_LENGTH

    # Tick at peak daylight (quarter cycle = sin peak)
    compute_light(DAY_LENGTH // 4)
    light = light_field.to_numpy()

    # Bright zone should have light
    assert light[0, 0] > 0.9, f"Bright zone too dim: {light[0, 0]}"
    # Dim zone should have some light
    assert 0.1 < light[LIGHT_ZONE_END + 1, 0] < 0.5, \
        f"Dim zone intensity wrong: {light[LIGHT_ZONE_END + 1, 0]}"
    # Dark zone should be dark
    assert light[DIM_ZONE_END + 1, 0] < 0.01, \
        f"Dark zone not dark: {light[DIM_ZONE_END + 1, 0]}"


def test_night_is_dark():
    """At night (tick 750 of a 1000-tick cycle), all light should be zero."""
    from world.grid import compute_light, light_field
    from config import DAY_LENGTH

    # sin(2*pi*750/1000) = sin(3*pi/2) = -1, clamped to 0
    compute_light(DAY_LENGTH * 3 // 4)
    light = light_field.to_numpy()

    assert light.max() < 1e-6, f"Light during night: max = {light.max()}"


if __name__ == "__main__":
    ti.init(arch=ti.cpu, random_seed=42)
    test_diffusion_conserves_mass_without_decay()
    print("PASS: diffusion_conserves_mass_without_decay")
    test_decay_reduces_total()
    print("PASS: decay_reduces_total")
    test_toroidal_wrapping()
    print("PASS: toroidal_wrapping")
    test_no_negative_values()
    print("PASS: no_negative_values")
    test_light_zones()
    print("PASS: light_zones")
    test_night_is_dark()
    print("PASS: night_is_dark")
    print("\nAll chemistry tests passed!")
