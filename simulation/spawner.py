"""Initial cell seeding and emergency respawning."""

import numpy as np

from config import (
    INITIAL_CELL_COUNT, INITIAL_ENERGY, INITIAL_STRUCTURE, INITIAL_REPMAT,
    MEMBRANE_INITIAL, LIGHT_ZONE_END, GRID_HEIGHT, RANDOM_SEED,
    RESPAWN_COUNT, GENOME_TYPE, RESPAWN_ENERGY,
)
from cell.cell_state import (
    cell_alive, cell_x, cell_y, cell_energy, cell_structure, cell_repmat,
    cell_signal, cell_membrane, cell_age, cell_genome_id, cell_facing,
    grid_cell_id, cell_count, allocate_cell_python,
)


def seed_cells(count: int = INITIAL_CELL_COUNT, seed: int = RANDOM_SEED):
    """Place initial cells randomly in the light zone."""
    rng = np.random.default_rng(seed + 1)  # offset from chemistry seed
    placed = 0

    while placed < count:
        x = int(rng.integers(0, LIGHT_ZONE_END))
        y = int(rng.integers(0, GRID_HEIGHT))

        # Skip if position already occupied
        if grid_cell_id[x, y] >= 0:
            continue

        slot = allocate_cell_python()
        if slot < 0:
            break  # no more slots

        cell_alive[slot] = 1
        cell_x[slot] = x
        cell_y[slot] = y
        cell_energy[slot] = INITIAL_ENERGY
        cell_structure[slot] = INITIAL_STRUCTURE
        cell_repmat[slot] = INITIAL_REPMAT
        cell_signal[slot] = 0.0
        cell_membrane[slot] = MEMBRANE_INITIAL
        cell_age[slot] = 0
        cell_genome_id[slot] = placed  # each cell gets a unique genome for initial diversity
        cell_facing[slot] = int(rng.integers(0, 4))

        grid_cell_id[x, y] = slot
        placed += 1

    cell_count[None] = placed


_respawn_rng = np.random.default_rng(RANDOM_SEED + 200)


def respawn_cells(count: int = RESPAWN_COUNT):
    """Emergency respawn: add fresh cells when population is dangerously low.

    Uses existing genome allocation infrastructure to create new genomes
    with random parameters, mimicking immigration of novel organisms.
    """
    from cell.genome import (
        genome_weights, genome_ref_count, allocate_genome_python,
    )

    placed = 0
    for _ in range(count * 3):  # extra attempts for occupied cells
        if placed >= count:
            break

        x = int(_respawn_rng.integers(0, LIGHT_ZONE_END))
        y = int(_respawn_rng.integers(0, GRID_HEIGHT))

        if grid_cell_id[x, y] >= 0:
            continue

        slot = allocate_cell_python()
        if slot < 0:
            break

        # Allocate a fresh genome
        gid = allocate_genome_python()
        if gid < 0:
            break

        # Initialize genome with random weights
        if GENOME_TYPE == "crn":
            from cell.crn_genome import crn_weights, crn_chemicals
            from config import (CRN_GENOME_SIZE, CRN_EXTRA_PARAMS,
                                NUM_INTERNAL_CHEMICALS, NUM_ACTION_CHEMICALS,
                                MAX_REACTIONS, CRN_PARAMS_PER_REACTION,
                                CRN_HIDDEN_DECAY)
            react_end = CRN_GENOME_SIZE - CRN_EXTRA_PARAMS
            n = 0.02  # noise scale (matches init_crn_genome_table)
            # Bootstrap reactions 0-3 (same as init_crn_genome_table)
            bootstraps = [
                # (inp_a, inp_b, out, th_a, th_b, rate, decay)
                (0.03, 0.03, 0.78, 0.2, 0.2, 0.3, 0.1),    # R0: light→eat
                (0.09, 0.09, 0.90, 0.3, 0.3, 0.4, 0.2),    # R1: energy→divide
                (0.15, 0.15, 0.84, 0.1, 0.1, 0.2, 0.15),   # R2: structure→move
                (0.47, 0.47, 0.84, 0.3, 0.3, 0.25, 0.1),   # R3: waste→move
            ]
            for r, (ia, ib, out, ta, tb, rate, decay) in enumerate(bootstraps):
                base = r * CRN_PARAMS_PER_REACTION
                crn_weights[gid, base + 0] = float(ia + _respawn_rng.normal(0, n))
                crn_weights[gid, base + 1] = float(ib + _respawn_rng.normal(0, n))
                crn_weights[gid, base + 2] = float(out + _respawn_rng.normal(0, n))
                crn_weights[gid, base + 3] = float(ta + _respawn_rng.normal(0, n))
                crn_weights[gid, base + 4] = float(tb + _respawn_rng.normal(0, n))
                crn_weights[gid, base + 5] = float(rate + _respawn_rng.normal(0, n * 2))
                crn_weights[gid, base + 6] = float(decay + _respawn_rng.normal(0, n))
            # Reactions 4-15: random with zone-biased targeting
            for r in range(4, min(16, MAX_REACTIONS)):
                base = r * CRN_PARAMS_PER_REACTION
                crn_weights[gid, base + 0] = float(_respawn_rng.uniform(0.0, 0.5))
                crn_weights[gid, base + 1] = float(_respawn_rng.uniform(0.0, 0.5))
                if _respawn_rng.random() < 0.6:
                    crn_weights[gid, base + 2] = float(_respawn_rng.uniform(0.75, 1.0))
                else:
                    crn_weights[gid, base + 2] = float(_respawn_rng.uniform(0.5, 0.75))
                crn_weights[gid, base + 3] = float(_respawn_rng.uniform(0.1, 0.5))
                crn_weights[gid, base + 4] = float(_respawn_rng.uniform(0.1, 0.5))
                crn_weights[gid, base + 5] = float(_respawn_rng.uniform(-0.3, 0.3))
                crn_weights[gid, base + 6] = float(_respawn_rng.uniform(0.01, 0.3))
            # Reactions 16+: silent slots (wired but rate=0)
            for r in range(16, MAX_REACTIONS):
                base = r * CRN_PARAMS_PER_REACTION
                crn_weights[gid, base + 0] = float(_respawn_rng.uniform(0.0, 0.5))
                crn_weights[gid, base + 1] = float(_respawn_rng.uniform(0.0, 0.5))
                if _respawn_rng.random() < 0.6:
                    crn_weights[gid, base + 2] = float(_respawn_rng.uniform(0.75, 1.0))
                else:
                    crn_weights[gid, base + 2] = float(_respawn_rng.uniform(0.5, 0.75))
                crn_weights[gid, base + 3] = float(_respawn_rng.uniform(0.1, 0.5))
                crn_weights[gid, base + 4] = float(_respawn_rng.uniform(0.1, 0.5))
                crn_weights[gid, base + 5] = 0.0  # silent: rate=0
                crn_weights[gid, base + 6] = float(_respawn_rng.uniform(0.01, 0.3))
            # Action biases: eat=0.3, move=0.3, divide=0.2, attack=-0.3
            for a, bias in enumerate([0.3, 0.3, 0.2, -0.3]):
                crn_weights[gid, react_end + a] = float(
                    bias + _respawn_rng.normal(0, 0.05))
            # Hidden decay rates
            for h in range(CRN_EXTRA_PARAMS - NUM_ACTION_CHEMICALS):
                crn_weights[gid, react_end + NUM_ACTION_CHEMICALS + h] = float(
                    CRN_HIDDEN_DECAY + abs(_respawn_rng.normal(0, 0.01)))
            for c in range(NUM_INTERNAL_CHEMICALS):
                crn_chemicals[slot, c] = 0.0
            for a in range(NUM_ACTION_CHEMICALS):
                crn_chemicals[slot, 12 + a] = crn_weights[gid, react_end + a]
        elif GENOME_TYPE == "ctrnn":
            from cell.ctrnn_genome import ctrnn_weights, ctrnn_neurons
            from config import (CTRNN_GENOME_SIZE, CTRNN_NUM_NEURONS,
                                CTRNN_NUM_SENSORY, CTRNN_NUM_HIDDEN,
                                CTRNN_NUM_ACTION, CTRNN_PARAMS_PER_NEURON,
                                CTRNN_TAU_MIN, CTRNN_TAU_MAX,
                                CTRNN_AMPLITUDE_INIT, CTRNN_RECURRENT_K)
            NN = CTRNN_NUM_NEURONS
            NS = CTRNN_NUM_SENSORY
            NH = CTRNN_NUM_HIDDEN
            NA = CTRNN_NUM_ACTION
            PPR = CTRNN_PARAMS_PER_NEURON
            NEURON_END = NN * PPR
            nn = 0.02
            for n_idx in range(NN):
                base = n_idx * PPR
                # Zone-dependent tau
                if n_idx < NS:
                    tau = 0.3 + _respawn_rng.normal(0, 0.05)
                elif n_idx < NS + NH:
                    tau = 1.5 + _respawn_rng.normal(0, 0.2)
                else:
                    tau = 0.7 + _respawn_rng.normal(0, 0.1)
                ctrnn_weights[gid, base + 0] = float(
                    max(CTRNN_TAU_MIN, min(CTRNN_TAU_MAX, tau)))
                # Bias: small random for sensory/hidden, zero for action readouts
                if n_idx >= NS + NH:
                    ctrnn_weights[gid, base + 1] = 0.0
                else:
                    ctrnn_weights[gid, base + 1] = float(
                        _respawn_rng.normal(0, 0.1))
                # Amplitude (unused for action neurons since feedforward)
                ctrnn_weights[gid, base + 2] = float(
                    CTRNN_AMPLITUDE_INIT + _respawn_rng.normal(0, 0.1))
                # Weights and targets: random for sensory/hidden, zero for action
                if n_idx >= NS + NH:
                    for k in range(CTRNN_RECURRENT_K):
                        ctrnn_weights[gid, base + 3 + k] = 0.0
                        ctrnn_weights[gid, base + 7 + k] = 0.0
                else:
                    for k in range(CTRNN_RECURRENT_K):
                        ctrnn_weights[gid, base + 3 + k] = float(
                            _respawn_rng.normal(0, 0.2))
                        ctrnn_weights[gid, base + 7 + k] = float(
                            _respawn_rng.uniform(0.0, 1.0))
            # Bootstrap: feedforward readout connections
            b12 = 12 * PPR
            ctrnn_weights[gid, b12 + 3] = float(1.0 + _respawn_rng.normal(0, nn))
            ctrnn_weights[gid, b12 + 7] = float(0.03 + _respawn_rng.normal(0, nn))
            b14 = 14 * PPR
            ctrnn_weights[gid, b14 + 3] = float(2.5 + _respawn_rng.normal(0, nn))
            ctrnn_weights[gid, b14 + 7] = float(0.09 + _respawn_rng.normal(0, nn))
            b13 = 13 * PPR
            ctrnn_weights[gid, b13 + 3] = float(0.35 + _respawn_rng.normal(0, nn))
            ctrnn_weights[gid, b13 + 7] = float(0.15 + _respawn_rng.normal(0, nn))
            ctrnn_weights[gid, b13 + 4] = float(0.35 + _respawn_rng.normal(0, nn))
            ctrnn_weights[gid, b13 + 8] = float(0.47 + _respawn_rng.normal(0, nn))
            # Input gains
            for s in range(NS):
                ctrnn_weights[gid, NEURON_END + s] = float(
                    1.0 + _respawn_rng.normal(0, 0.05))
            # Action biases (feedforward readout, divide=-0.35 for E=35 threshold)
            for a, bias in enumerate([0.3, 0.3, -0.35, -0.3]):
                ctrnn_weights[gid, NEURON_END + NS + a] = float(
                    bias + _respawn_rng.normal(0, 0.05))
            # Reset neurons
            for n_idx in range(CTRNN_NUM_NEURONS):
                ctrnn_neurons[slot, n_idx] = 0.0
        else:
            from config import GENOME_SIZE, SEED_WEIGHT_SIGMA
            for w in range(GENOME_SIZE):
                genome_weights[gid, w] = float(
                    _respawn_rng.normal(0, SEED_WEIGHT_SIGMA))

        genome_ref_count[gid] = 1

        cell_alive[slot] = 1
        cell_x[slot] = x
        cell_y[slot] = y
        cell_energy[slot] = RESPAWN_ENERGY
        cell_structure[slot] = INITIAL_STRUCTURE
        cell_repmat[slot] = INITIAL_REPMAT
        cell_signal[slot] = 0.0
        cell_membrane[slot] = MEMBRANE_INITIAL
        cell_age[slot] = 0
        cell_genome_id[slot] = gid
        cell_facing[slot] = int(_respawn_rng.integers(0, 4))

        grid_cell_id[x, y] = slot
        placed += 1

    cell_count[None] = cell_count[None] + placed
    return placed
