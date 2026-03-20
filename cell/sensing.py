"""Compute all 34 sensory inputs for every alive cell in parallel.

Inputs [0..17]: original 18 sensory channels (light, internal state, gradients, neighbors).
  Input [15]: waste concentration at cell position (if WASTE_ENABLED), else age.
Inputs [18..33]: bond signal channels (4 bonds × 4 channels each).
"""

import taichi as ti

from config import (
    MAX_CELLS, GRID_WIDTH, GRID_HEIGHT, MAX_CELL_AGE, NUM_INPUTS,
    GRADIENT_SCALE_S, GRADIENT_SCALE_R, BOND_SIGNAL_CHANNELS,
    GRADIENT_NOISE_SIGMA, WASTE_ENABLED,
)
from cell.cell_state import (
    cell_alive, cell_x, cell_y, cell_energy, cell_structure, cell_repmat,
    cell_membrane, cell_age, cell_facing, cell_bonds, cell_bond_signal_in,
    grid_cell_id,
)
from cell.genome import sensory_inputs
from world.grid import light_field


@ti.func
def _box_muller_normal() -> ti.f32:
    """Generate a standard normal random number on GPU using Box-Muller transform."""
    u1 = ti.max(1e-7, ti.random(ti.f32))
    u2 = ti.random(ti.f32)
    return ti.sqrt(-2.0 * ti.log(u1)) * ti.cos(2.0 * 3.14159265358979 * u2)


@ti.func
def facing_offset(facing: ti.i32) -> ti.math.ivec2:
    """Return (dx, dy) for a facing direction. 0=up, 1=right, 2=down, 3=left."""
    dx = 0
    dy = 0
    if facing == 0:
        dy = 1
    elif facing == 1:
        dx = 1
    elif facing == 2:
        dy = -1
    else:
        dx = -1
    return ti.math.ivec2(dx, dy)


@ti.func
def left_of(facing: ti.i32) -> ti.i32:
    return (facing - 1 + 4) % 4


@ti.func
def right_of(facing: ti.i32) -> ti.i32:
    return (facing + 1) % 4


@ti.func
def cell_at(x: ti.i32, y: ti.i32) -> ti.i32:
    """Check if a cell exists at (x, y). Returns 1 if occupied, 0 if empty."""
    wx = x % GRID_WIDTH
    wy = y % GRID_HEIGHT
    result = 0
    if grid_cell_id[wx, wy] >= 0:
        result = 1
    return result


@ti.kernel
def compute_sensory_inputs(env_S: ti.template(), env_R: ti.template(),
                           env_G: ti.template(), env_W: ti.template()):
    """Fill sensory_inputs[i, 0..17] for all alive cells."""
    for i in range(MAX_CELLS):
        if cell_alive[i] == 1:
            x = cell_x[i]
            y = cell_y[i]
            f = cell_facing[i]

            # [0] light at current position
            sensory_inputs[i, 0] = light_field[x, y]

            # [1] energy level (normalized, cap at 1.0)
            sensory_inputs[i, 1] = ti.min(1.0, cell_energy[i] / 100.0)

            # [2] structure level
            sensory_inputs[i, 2] = ti.min(1.0, cell_structure[i] / 50.0)

            # [3] replication material level
            sensory_inputs[i, 3] = ti.min(1.0, cell_repmat[i] / 20.0)

            # [4] membrane integrity
            sensory_inputs[i, 4] = cell_membrane[i] / 100.0

            # [5]-[6] S gradient with Gaussian noise
            xp = (x + 1) % GRID_WIDTH
            xm = (x - 1 + GRID_WIDTH) % GRID_WIDTH
            yp = (y + 1) % GRID_HEIGHT
            ym = (y - 1 + GRID_HEIGHT) % GRID_HEIGHT

            noise_sigma = GRADIENT_NOISE_SIGMA

            sensory_inputs[i, 5] = ti.min(1.0, ti.max(-1.0,
                (env_S[xp, y] - env_S[xm, y]) * GRADIENT_SCALE_S
                + _box_muller_normal() * noise_sigma))
            sensory_inputs[i, 6] = ti.min(1.0, ti.max(-1.0,
                (env_S[x, yp] - env_S[x, ym]) * GRADIENT_SCALE_S
                + _box_muller_normal() * noise_sigma))

            # [7]-[8] R gradient with Gaussian noise
            sensory_inputs[i, 7] = ti.min(1.0, ti.max(-1.0,
                (env_R[xp, y] - env_R[xm, y]) * GRADIENT_SCALE_R
                + _box_muller_normal() * noise_sigma))
            sensory_inputs[i, 8] = ti.min(1.0, ti.max(-1.0,
                (env_R[x, yp] - env_R[x, ym]) * GRADIENT_SCALE_R
                + _box_muller_normal() * noise_sigma))

            # [9]-[10] G (signal) gradient with Gaussian noise
            sensory_inputs[i, 9] = ti.min(1.0, ti.max(-1.0,
                (env_G[xp, y] - env_G[xm, y]) * 0.5
                + _box_muller_normal() * noise_sigma))
            sensory_inputs[i, 10] = ti.min(1.0, ti.max(-1.0,
                (env_G[x, yp] - env_G[x, ym]) * 0.5
                + _box_muller_normal() * noise_sigma))

            # [11] cell ahead + [16]-[17] prey energy/membrane
            ahead = facing_offset(f)
            ax = (x + ahead[0] + GRID_WIDTH) % GRID_WIDTH
            ay = (y + ahead[1] + GRID_HEIGHT) % GRID_HEIGHT
            ahead_id = grid_cell_id[ax, ay]
            if ahead_id >= 0 and cell_alive[ahead_id] == 1:
                sensory_inputs[i, 11] = 1.0
                sensory_inputs[i, 16] = ti.min(1.0, cell_energy[ahead_id] / 100.0)
                sensory_inputs[i, 17] = cell_membrane[ahead_id] / 100.0
            else:
                sensory_inputs[i, 11] = 0.0
                sensory_inputs[i, 16] = 0.0
                sensory_inputs[i, 17] = 0.0

            # [12] cell to the left
            left_dir = facing_offset(left_of(f))
            sensory_inputs[i, 12] = ti.cast(
                cell_at(x + left_dir[0], y + left_dir[1]), ti.f32)

            # [13] cell to the right
            right_dir = facing_offset(right_of(f))
            sensory_inputs[i, 13] = ti.cast(
                cell_at(x + right_dir[0], y + right_dir[1]), ti.f32)

            # [14] bond count (normalized 0-1, max 4 bonds)
            bonds = 0
            for b in range(4):
                if cell_bonds[i, b] >= 0:
                    bonds += 1
            sensory_inputs[i, 14] = ti.cast(bonds, ti.f32) / 4.0

            # [15] waste concentration at cell position (or age if waste disabled)
            if ti.static(WASTE_ENABLED):
                sensory_inputs[i, 15] = ti.min(1.0, env_W[cell_x[i], cell_y[i]])
            else:
                sensory_inputs[i, 15] = ti.cast(cell_age[i], ti.f32) / ti.cast(
                    MAX_CELL_AGE, ti.f32)

            # [18..33] bond signal inputs: 4 bonds × 4 channels
            for b in range(4):
                for ch in range(BOND_SIGNAL_CHANNELS):
                    idx = 18 + b * BOND_SIGNAL_CHANNELS + ch
                    if cell_bonds[i, b] >= 0:
                        sensory_inputs[i, idx] = cell_bond_signal_in[i, b, ch]
                    else:
                        sensory_inputs[i, idx] = 0.0
