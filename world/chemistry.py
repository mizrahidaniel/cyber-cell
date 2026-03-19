"""Chemical fields, diffusion kernel, decay, and deposit placement."""

import numpy as np
import taichi as ti

from config import (
    GRID_WIDTH, GRID_HEIGHT,
    DIFFUSION_RATE_S, DIFFUSION_RATE_R, DIFFUSION_RATE_G,
    DECAY_RATE_S, DECAY_RATE_R, DECAY_RATE_G,
    DEPOSIT_REPLENISH_RATE,
    NUM_DEPOSITS_S, NUM_DEPOSITS_R,
    DEPOSIT_CLUSTER_RADIUS, DEPOSIT_CLUSTER_AMOUNT,
    LIGHT_ZONE_END, DIM_ZONE_END,
    RANDOM_SEED, R_LIGHT_ZONE_FRACTION,
)

# Double-buffered chemical fields (environment only — E is internal)
env_S_a = ti.field(dtype=ti.f32, shape=(GRID_WIDTH, GRID_HEIGHT))
env_S_b = ti.field(dtype=ti.f32, shape=(GRID_WIDTH, GRID_HEIGHT))
env_R_a = ti.field(dtype=ti.f32, shape=(GRID_WIDTH, GRID_HEIGHT))
env_R_b = ti.field(dtype=ti.f32, shape=(GRID_WIDTH, GRID_HEIGHT))
env_G_a = ti.field(dtype=ti.f32, shape=(GRID_WIDTH, GRID_HEIGHT))
env_G_b = ti.field(dtype=ti.f32, shape=(GRID_WIDTH, GRID_HEIGHT))

# Deposit source positions
deposit_S_x = ti.field(dtype=ti.i32, shape=(NUM_DEPOSITS_S,))
deposit_S_y = ti.field(dtype=ti.i32, shape=(NUM_DEPOSITS_S,))
deposit_R_x = ti.field(dtype=ti.i32, shape=(NUM_DEPOSITS_R,))
deposit_R_y = ti.field(dtype=ti.i32, shape=(NUM_DEPOSITS_R,))

# Buffer toggle: 0 means read from _a write to _b, 1 means opposite
_current_buffer = 0


def get_env_S():
    """Return the current read buffer for S."""
    return env_S_a if _current_buffer == 0 else env_S_b


def get_env_R():
    return env_R_a if _current_buffer == 0 else env_R_b


def get_env_G():
    return env_G_a if _current_buffer == 0 else env_G_b


def _get_dst_S():
    return env_S_b if _current_buffer == 0 else env_S_a


def _get_dst_R():
    return env_R_b if _current_buffer == 0 else env_R_a


def _get_dst_G():
    return env_G_b if _current_buffer == 0 else env_G_a


@ti.kernel
def _diffuse_and_decay(src: ti.template(), dst: ti.template(),
                       diff_rate: ti.f32, decay_rate: ti.f32):
    """Diffuse chemicals to 4 cardinal neighbors and apply decay.

    Each cell reads its own value and its 4 neighbors from src,
    computes retained + received, applies decay, writes to dst.
    No atomics needed — each cell writes to exactly one location.
    """
    for i, j in src:
        # Toroidal neighbor coordinates
        ip = (i + 1) % GRID_WIDTH
        im = (i - 1 + GRID_WIDTH) % GRID_WIDTH
        jp = (j + 1) % GRID_HEIGHT
        jm = (j - 1 + GRID_HEIGHT) % GRID_HEIGHT

        # What this cell retains after spreading
        retained = src[i, j] * (1.0 - diff_rate)

        # What this cell receives from its 4 neighbors
        spread_frac = diff_rate * 0.25
        received = (src[ip, j] + src[im, j] + src[i, jp] + src[i, jm]) * spread_frac

        # Apply decay
        dst[i, j] = ti.max(0.0, (retained + received) * (1.0 - decay_rate))


@ti.kernel
def _replenish_deposits_S():
    for d in range(NUM_DEPOSITS_S):
        x = deposit_S_x[d]
        y = deposit_S_y[d]
        env_S_a[x, y] += DEPOSIT_REPLENISH_RATE
        env_S_b[x, y] += DEPOSIT_REPLENISH_RATE


@ti.kernel
def _replenish_deposits_R():
    for d in range(NUM_DEPOSITS_R):
        x = deposit_R_x[d]
        y = deposit_R_y[d]
        env_R_a[x, y] += DEPOSIT_REPLENISH_RATE
        env_R_b[x, y] += DEPOSIT_REPLENISH_RATE


def diffuse_all():
    """Run one step of diffusion and decay for all environmental chemicals."""
    if _current_buffer == 0:
        _diffuse_and_decay(env_S_a, env_S_b, DIFFUSION_RATE_S, DECAY_RATE_S)
        _diffuse_and_decay(env_R_a, env_R_b, DIFFUSION_RATE_R, DECAY_RATE_R)
        _diffuse_and_decay(env_G_a, env_G_b, DIFFUSION_RATE_G, DECAY_RATE_G)
    else:
        _diffuse_and_decay(env_S_b, env_S_a, DIFFUSION_RATE_S, DECAY_RATE_S)
        _diffuse_and_decay(env_R_b, env_R_a, DIFFUSION_RATE_R, DECAY_RATE_R)
        _diffuse_and_decay(env_G_b, env_G_a, DIFFUSION_RATE_G, DECAY_RATE_G)


def replenish_deposits():
    """Add chemicals at deposit source locations (both buffers for consistency)."""
    _replenish_deposits_S()
    _replenish_deposits_R()


def swap_buffers():
    """Toggle which buffer is the current read buffer."""
    global _current_buffer
    _current_buffer = 1 - _current_buffer


def get_current_buffer() -> int:
    return _current_buffer


def set_current_buffer(val: int):
    global _current_buffer
    _current_buffer = val


def _place_clustered_deposits(rng, num_deposits, x_field, y_field,
                              x_min, x_max, num_clusters=None):
    """Place deposits in clusters within the given x-range."""
    if num_clusters is None:
        num_clusters = max(1, num_deposits // 5)

    # Pick cluster centers
    centers_x = rng.integers(x_min, x_max, size=num_clusters)
    centers_y = rng.integers(0, GRID_HEIGHT, size=num_clusters)

    positions_x = np.zeros(num_deposits, dtype=np.int32)
    positions_y = np.zeros(num_deposits, dtype=np.int32)

    for d in range(num_deposits):
        # Pick a random cluster center
        c = rng.integers(0, num_clusters)
        # Offset from center within cluster radius
        ox = rng.integers(-DEPOSIT_CLUSTER_RADIUS, DEPOSIT_CLUSTER_RADIUS + 1)
        oy = rng.integers(-DEPOSIT_CLUSTER_RADIUS, DEPOSIT_CLUSTER_RADIUS + 1)
        positions_x[d] = (centers_x[c] + ox) % GRID_WIDTH
        positions_y[d] = (centers_y[c] + oy) % GRID_HEIGHT

    x_field.from_numpy(positions_x)
    y_field.from_numpy(positions_y)

    return positions_x, positions_y


def init_chemistry(seed: int = RANDOM_SEED):
    """Place initial chemical deposits and seed the environment."""
    rng = np.random.default_rng(seed)

    # S deposits: dim + dark zones (stepping stones for cells leaving light)
    _place_clustered_deposits(rng, NUM_DEPOSITS_S, deposit_S_x, deposit_S_y,
                              x_min=LIGHT_ZONE_END, x_max=GRID_WIDTH)

    # R deposits: mostly dim + dark zones, small fraction in light zone
    n_r_light = int(NUM_DEPOSITS_R * R_LIGHT_ZONE_FRACTION)
    n_r_outer = NUM_DEPOSITS_R - n_r_light
    # Place outer deposits into temp arrays, then light zone deposits
    r_x_all = np.zeros(NUM_DEPOSITS_R, dtype=np.int32)
    r_y_all = np.zeros(NUM_DEPOSITS_R, dtype=np.int32)
    # Outer (dim + dark)
    _tmp_x = ti.field(dtype=ti.i32, shape=(n_r_outer,))
    _tmp_y = ti.field(dtype=ti.i32, shape=(n_r_outer,))
    _place_clustered_deposits(rng, n_r_outer, _tmp_x, _tmp_y,
                              x_min=LIGHT_ZONE_END, x_max=GRID_WIDTH)
    r_x_all[:n_r_outer] = _tmp_x.to_numpy()
    r_y_all[:n_r_outer] = _tmp_y.to_numpy()
    # Light zone
    if n_r_light > 0:
        _tmp_x2 = ti.field(dtype=ti.i32, shape=(n_r_light,))
        _tmp_y2 = ti.field(dtype=ti.i32, shape=(n_r_light,))
        _place_clustered_deposits(rng, n_r_light, _tmp_x2, _tmp_y2,
                                  x_min=0, x_max=LIGHT_ZONE_END)
        r_x_all[n_r_outer:] = _tmp_x2.to_numpy()
        r_y_all[n_r_outer:] = _tmp_y2.to_numpy()
    deposit_R_x.from_numpy(r_x_all)
    deposit_R_y.from_numpy(r_y_all)

    # Seed initial chemical concentrations at deposit locations
    s_x = deposit_S_x.to_numpy()
    s_y = deposit_S_y.to_numpy()
    r_x = deposit_R_x.to_numpy()
    r_y = deposit_R_y.to_numpy()

    s_field = np.zeros((GRID_WIDTH, GRID_HEIGHT), dtype=np.float32)
    r_field = np.zeros((GRID_WIDTH, GRID_HEIGHT), dtype=np.float32)

    for i in range(NUM_DEPOSITS_S):
        s_field[s_x[i], s_y[i]] += DEPOSIT_CLUSTER_AMOUNT

    for i in range(NUM_DEPOSITS_R):
        r_field[r_x[i], r_y[i]] += DEPOSIT_CLUSTER_AMOUNT

    env_S_a.from_numpy(s_field)
    env_S_b.from_numpy(s_field)
    env_R_a.from_numpy(r_field)
    env_R_b.from_numpy(r_field)
