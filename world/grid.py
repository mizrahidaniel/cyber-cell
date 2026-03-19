"""World grid, terrain zones, and light model with day/night cycle."""

import math
import taichi as ti

from config import (
    GRID_WIDTH, GRID_HEIGHT,
    LIGHT_ZONE_END, DIM_ZONE_END,
    LIGHT_BRIGHT, LIGHT_DIM, LIGHT_DARK,
    DAY_LENGTH,
)

light_field = ti.field(dtype=ti.f32, shape=(GRID_WIDTH, GRID_HEIGHT))


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


def init_grid():
    """One-time grid initialization."""
    compute_light(0)
