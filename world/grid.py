"""World grid, terrain zones, and light model with day/night cycle."""

import math
import taichi as ti

from config import (
    GRID_WIDTH, GRID_HEIGHT,
    LIGHT_ZONE_END, DIM_ZONE_END,
    LIGHT_BRIGHT, LIGHT_DIM, LIGHT_DARK,
    DAY_LENGTH, MAX_CELLS,
    LIGHT_ATTENUATION_ENABLED, LIGHT_ATTENUATION_K, LIGHT_ATTENUATION_RADIUS,
)
from cell.cell_state import cell_alive, cell_x, cell_y

light_field = ti.field(dtype=ti.f32, shape=(GRID_WIDTH, GRID_HEIGHT))
local_density = ti.field(dtype=ti.i32, shape=(GRID_WIDTH, GRID_HEIGHT))


@ti.func
def get_light(x: ti.i32, y: ti.i32) -> ti.f32:
    return light_field[x, y]


@ti.func
def wrap_x(x: ti.i32) -> ti.i32:
    return x % GRID_WIDTH


@ti.func
def wrap_y(y: ti.i32) -> ti.i32:
    return y % GRID_HEIGHT


@ti.kernel
def compute_light(tick: ti.i32):
    """Recompute light intensity at every grid cell for the current tick."""
    day_night = ti.max(0.0, ti.sin(2.0 * math.pi * ti.cast(tick, ti.f32)
                                    / ti.cast(DAY_LENGTH, ti.f32)))
    for i, j in light_field:
        base = LIGHT_DARK
        if i < LIGHT_ZONE_END:
            base = LIGHT_BRIGHT
        elif i < DIM_ZONE_END:
            base = LIGHT_DIM
        light_field[i, j] = base * day_night


@ti.kernel
def compute_local_density():
    """Count alive cells within LIGHT_ATTENUATION_RADIUS of each grid cell."""
    for i, j in local_density:
        local_density[i, j] = 0
    for idx in range(MAX_CELLS):
        if cell_alive[idx] == 1:
            cx = cell_x[idx]
            cy = cell_y[idx]
            for di in ti.static(range(-LIGHT_ATTENUATION_RADIUS,
                                      LIGHT_ATTENUATION_RADIUS + 1)):
                for dj in ti.static(range(-LIGHT_ATTENUATION_RADIUS,
                                          LIGHT_ATTENUATION_RADIUS + 1)):
                    if di != 0 or dj != 0:
                        ni = (cx + di) % GRID_WIDTH
                        nj = (cy + dj) % GRID_HEIGHT
                        ti.atomic_add(local_density[ni, nj], 1)


@ti.kernel
def apply_light_attenuation():
    """Beer-Lambert: multiply light by exp(-k * local_density)."""
    for i, j in light_field:
        d = local_density[i, j]
        if d > 0:
            light_field[i, j] *= ti.exp(-LIGHT_ATTENUATION_K
                                        * ti.cast(d, ti.f32))


def init_grid():
    """One-time grid initialization."""
    compute_light(0)
