"""Chemical Reaction Network (CRN) genome with separated chemical spaces.

16 internal chemicals in three zones:
  - Sensory (0-7):  Written by environment, read by reactions
  - Hidden (8-11):  Purely internal — memory, oscillators, gates
  - Action (12-15): Reset each tick to evolved biases, drive actions

This separation forces reactions to build sensory->action circuits rather
than having sensory inputs directly trigger actions (the CRN equivalent
of a hidden layer). Negative thresholds enable NOT-gate logic.
"""

import numpy as np
import taichi as ti

from config import (
    MAX_CELLS, MAX_GENOMES, NUM_INTERNAL_CHEMICALS, MAX_REACTIONS,
    CRN_PARAMS_PER_REACTION, CRN_GENOME_SIZE,
    CRN_MUTATION_RATE_PERTURB, CRN_MUTATION_SIGMA,
    CRN_MUTATION_RATE_DUPLICATE, CRN_MUTATION_RATE_DELETE,
    CRN_MUTATION_RATE_REWIRE, NUM_OUTPUTS, INITIAL_CELL_COUNT,
    RANDOM_SEED, NUM_SENSORY_CHEMICALS, NUM_ACTION_CHEMICALS,
    NUM_HIDDEN_CHEMICALS, CRN_SENSORY_BLEND, CRN_EXTRA_PARAMS,
    CRN_GRADIENT_TURN_MIN, ACTION_THRESHOLD, CRN_HIDDEN_DECAY,
    CRN_ACTION_GAIN, CRN_ACTION_CENTER, CRN_HIDDEN_BASAL,
)
from cell.cell_state import cell_alive, cell_genome_id, cell_facing
from cell.genome import sensory_inputs, action_outputs, needs_mutation

# CRN genome storage: reaction params (112) + action biases (4) + hidden decay (4)
crn_weights = ti.field(dtype=ti.f32, shape=(MAX_GENOMES, CRN_GENOME_SIZE))

# Per-cell internal chemical concentrations (persistent between ticks)
crn_chemicals = ti.field(dtype=ti.f32, shape=(MAX_CELLS, NUM_INTERNAL_CHEMICALS))

from cell.genome import (
    genome_ref_count, genome_count, genome_free_list, genome_free_count,
    genome_parent_id, genome_birth_tick,
    _mutation_event_count, _mutation_events, _pending_mutations, _pending_count,
)

NC = NUM_INTERNAL_CHEMICALS       # 16
NS = NUM_SENSORY_CHEMICALS        # 8
NH = NUM_HIDDEN_CHEMICALS         # 4
NA = NUM_ACTION_CHEMICALS         # 4
MR = MAX_REACTIONS                # 16
PPR = CRN_PARAMS_PER_REACTION    # 7
REACT_END = MR * PPR             # 112

# Chem 0-7 <- sensory_inputs[idx]: light, energy, structure, S_gx, S_gy, cell_ahead, bonds, waste
_SENSORY_MAP = ti.field(dtype=ti.i32, shape=(NS,))
# Chem 12-15 -> action_outputs[idx]: eat, move_forward, divide, attack
_ACTION_MAP = ti.field(dtype=ti.i32, shape=(NA,))


def _init_maps():
    """Initialize sensory and action mapping tables."""
    _SENSORY_MAP.from_numpy(np.array([0, 1, 2, 5, 6, 11, 14, 15], dtype=np.int32))
    _ACTION_MAP.from_numpy(np.array([3, 0, 5, 8], dtype=np.int32))


@ti.kernel
def evaluate_all_crns():
    """CRN evaluation with separated chemical spaces."""
    for i in range(MAX_CELLS):
        if cell_alive[i] == 1:
            gid = cell_genome_id[i]

            # 1. Reset action chemicals (12-15) to evolved biases
            for a in range(NA):
                crn_chemicals[i, 12 + a] = crn_weights[gid, REACT_END + a]

            # 2. Blend sensory inputs into chemicals 0-7
            for s in range(NS):
                sens_idx = _SENSORY_MAP[s]
                crn_chemicals[i, s] = (crn_chemicals[i, s] * (1.0 - CRN_SENSORY_BLEND) +
                                       sensory_inputs[i, sens_idx] * CRN_SENSORY_BLEND)

            # 2b. Basal production for hidden chemicals
            for h in range(NH):
                crn_chemicals[i, NS + h] += CRN_HIDDEN_BASAL

            # 3. Evaluate reactions (read/write any of 16 chemicals)
            for r in range(MR):
                base = r * PPR
                input_a_raw = crn_weights[gid, base + 0]
                input_b_raw = crn_weights[gid, base + 1]
                output_raw = crn_weights[gid, base + 2]
                threshold_a = crn_weights[gid, base + 3]
                threshold_b = crn_weights[gid, base + 4]
                rate = crn_weights[gid, base + 5]

                input_a = ti.cast(ti.abs(input_a_raw) * NC, ti.i32) % NC
                input_b = ti.cast(ti.abs(input_b_raw) * NC, ti.i32) % NC
                output = ti.cast(ti.abs(output_raw) * NC, ti.i32) % NC

                val_a = crn_chemicals[i, input_a]
                val_b = crn_chemicals[i, input_b]

                # Inverted threshold: negative threshold -> fire when input < |threshold|
                cond_a = 0
                if threshold_a >= 0.0:
                    if val_a > threshold_a:
                        cond_a = 1
                else:
                    if val_a < -threshold_a:
                        cond_a = 1
                cond_b = 0
                if threshold_b >= 0.0:
                    if val_b > threshold_b:
                        cond_b = 1
                else:
                    if val_b < -threshold_b:
                        cond_b = 1

                if cond_a == 1 and cond_b == 1:
                    crn_chemicals[i, output] += rate

            # 4. Apply per-reaction decay to sensory chemicals only (0-7)
            #    Action (12-15) reset each tick; hidden (8-11) has step 5.
            for r in range(MR):
                base = r * PPR
                output_raw = crn_weights[gid, base + 2]
                output = ti.cast(ti.abs(output_raw) * NC, ti.i32) % NC
                if output < NS:
                    decay = ti.abs(crn_weights[gid, base + 6])
                    decay = ti.min(1.0, decay)
                    crn_chemicals[i, output] *= (1.0 - decay * 0.1)

            # 5. Decay hidden chemicals (8-11) using per-genome evolved rates
            for h in range(NH):
                decay = ti.abs(crn_weights[gid, REACT_END + NA + h])
                decay = ti.min(0.5, decay)
                crn_chemicals[i, 8 + h] *= (1.0 - decay)

            # 6. Clamp: sensory/hidden [0, 5], action [-1, 5]
            for c in range(NC):
                lo = -1.0 if c >= NS + NH else 0.0
                crn_chemicals[i, c] = ti.max(lo, ti.min(5.0, crn_chemicals[i, c]))

            # 7. Read action chemicals -> probabilistic firing (sigmoid)
            for a in range(NA):
                action_idx = _ACTION_MAP[a]
                val = crn_chemicals[i, 12 + a]
                # Sigmoid probability: smooth gradient, no hard cliff
                p = 1.0 / (1.0 + ti.exp(-CRN_ACTION_GAIN * (val - CRN_ACTION_CENTER)))
                if ti.random(ti.f32) < p:
                    # Guarantee downstream action check passes
                    action_outputs[i, action_idx] = ti.max(val, ACTION_THRESHOLD + 0.1)
                else:
                    action_outputs[i, action_idx] = 0.0

            # 8. Turns: facing-aware gradient steering
            grad_x = crn_chemicals[i, 3]
            grad_y = crn_chemicals[i, 4]
            facing = cell_facing[i]
            # Perpendicular gradient component (positive = right of facing)
            perp = 0.0
            if facing == 0:
                perp = grad_x
            elif facing == 1:
                perp = -grad_y
            elif facing == 2:
                perp = -grad_x
            else:
                perp = grad_y
            action_outputs[i, 1] = 0.0  # turn_left
            action_outputs[i, 2] = 0.0  # turn_right
            if ti.abs(perp) > CRN_GRADIENT_TURN_MIN:
                if perp < 0.0:
                    action_outputs[i, 1] = ACTION_THRESHOLD + 0.1
                else:
                    action_outputs[i, 2] = ACTION_THRESHOLD + 0.1

            # 9. Auxiliary actions from hidden chemicals
            if crn_chemicals[i, 8] > 0.5:
                action_outputs[i, 4] = crn_chemicals[i, 8]  # emit_signal
            else:
                action_outputs[i, 4] = 0.0
            if crn_chemicals[i, 9] > 0.5:
                action_outputs[i, 6] = crn_chemicals[i, 9]  # bond
            else:
                action_outputs[i, 6] = 0.0

            # 10. Zero unmapped outputs
            action_outputs[i, 7] = 0.0   # unbond
            action_outputs[i, 9] = 0.0   # repair
            for ao in range(10, NUM_OUTPUTS):
                action_outputs[i, ao] = 0.0


@ti.func
def _crn_box_muller() -> ti.f32:
    u1 = ti.max(1e-7, ti.random(ti.f32))
    u2 = ti.random(ti.f32)
    return ti.sqrt(-2.0 * ti.log(u1)) * ti.cos(2.0 * 3.14159265358979 * u2)


@ti.kernel
def _collect_pending_crn_mutations():
    _pending_count[None] = 0
    for i in range(MAX_CELLS):
        if needs_mutation[i] == 1:
            idx = ti.atomic_add(_pending_count[None], 1)
            _pending_mutations[idx] = i


@ti.kernel
def _apply_crn_mutations_gpu(tick: ti.i32):
    """Apply CRN mutation operators on GPU."""
    for idx in range(_pending_count[None]):
        cell_idx = _pending_mutations[idx]
        parent_gid = cell_genome_id[cell_idx]

        slot_idx = ti.atomic_sub(genome_free_count[None], 1) - 1
        if slot_idx < 0:
            ti.atomic_add(genome_free_count[None], 1)
            ti.atomic_add(genome_ref_count[parent_gid], 1)
            needs_mutation[cell_idx] = 0
            continue

        new_gid = genome_free_list[slot_idx]
        ti.atomic_add(genome_count[None], 1)
        changed = 0

        # Copy and mutate all 120 params
        for w in range(CRN_GENOME_SIZE):
            val = crn_weights[parent_gid, w]
            # Bias/decay params (112-119) get half mutation rate
            mut_rate = CRN_MUTATION_RATE_PERTURB
            if w >= REACT_END:
                mut_rate = CRN_MUTATION_RATE_PERTURB * 0.5
            if ti.random(ti.f32) < mut_rate:
                val += _crn_box_muller() * CRN_MUTATION_SIGMA
                changed = 1
            crn_weights[new_gid, w] = val

        # Reaction rewiring with zone-biased targeting
        for r in range(MR):
            base = r * PPR
            if ti.random(ti.f32) < CRN_MUTATION_RATE_REWIRE:
                which = ti.cast(ti.random(ti.f32) * 3.0, ti.i32) % 3
                zone_roll = ti.random(ti.f32)
                if zone_roll < 0.5:
                    crn_weights[new_gid, base + which] = ti.random(ti.f32) * 0.5
                elif zone_roll < 0.75:
                    crn_weights[new_gid, base + which] = 0.5 + ti.random(ti.f32) * 0.25
                else:
                    crn_weights[new_gid, base + which] = 0.75 + ti.random(ti.f32) * 0.25
                changed = 1

        # Reaction deletion
        for r in range(MR):
            if ti.random(ti.f32) < CRN_MUTATION_RATE_DELETE:
                crn_weights[new_gid, r * PPR + 5] = 0.0
                changed = 1

        # Reaction duplication
        if ti.random(ti.f32) < CRN_MUTATION_RATE_DUPLICATE:
            src_r = ti.cast(ti.random(ti.f32) * MR, ti.i32) % MR
            dst_r = ti.cast(ti.random(ti.f32) * MR, ti.i32) % MR
            if src_r != dst_r:
                for p in range(PPR):
                    crn_weights[new_gid, dst_r * PPR + p] = (
                        crn_weights[new_gid, src_r * PPR + p] +
                        _crn_box_muller() * CRN_MUTATION_SIGMA * 0.5
                    )
                changed = 1

        if changed == 0:
            ti.atomic_add(genome_free_count[None], 1)
            genome_free_list[slot_idx] = new_gid
            ti.atomic_sub(genome_count[None], 1)
            ti.atomic_add(genome_ref_count[parent_gid], 1)
        else:
            genome_ref_count[new_gid] = 1
            ti.atomic_sub(genome_ref_count[parent_gid], 1)
            cell_genome_id[cell_idx] = new_gid
            genome_parent_id[new_gid] = parent_gid
            genome_birth_tick[new_gid] = tick
            evt_idx = ti.atomic_add(_mutation_event_count[None], 1)
            if evt_idx < MAX_CELLS:
                _mutation_events[evt_idx, 0] = parent_gid
                _mutation_events[evt_idx, 1] = new_gid
                _mutation_events[evt_idx, 2] = tick
            # Reset daughter's chemicals to clean state
            for c in range(NC):
                crn_chemicals[cell_idx, c] = 0.0
            for a in range(NA):
                crn_chemicals[cell_idx, 12 + a] = crn_weights[new_gid, REACT_END + a]

        needs_mutation[cell_idx] = 0


def process_crn_mutations(tick=0):
    """Process CRN mutations for newly born cells."""
    _mutation_event_count[None] = 0
    _collect_pending_crn_mutations()
    if _pending_count[None] > 0:
        _apply_crn_mutations_gpu(tick)


_DEFAULT_ACTION_BIASES = np.array([0.3, 0.3, 0.2, -0.3], dtype=np.float32)


def _set_reaction(weights, g, r, inp_a, inp_b, out, th_a, th_b, rate, decay, rng):
    """Set a bootstrap reaction with small per-genome noise."""
    base = r * PPR
    n = 0.02
    weights[g, base + 0] = inp_a + rng.normal(0, n)
    weights[g, base + 1] = inp_b + rng.normal(0, n)
    weights[g, base + 2] = out + rng.normal(0, n)
    weights[g, base + 3] = th_a + rng.normal(0, n)
    weights[g, base + 4] = th_b + rng.normal(0, n)
    weights[g, base + 5] = rate + rng.normal(0, n * 2)
    weights[g, base + 6] = decay + rng.normal(0, n)


def init_crn_genome_table(count: int = INITIAL_CELL_COUNT, seed: int = RANDOM_SEED):
    """Initialize CRN genomes with separated chemical spaces."""
    _init_maps()
    rng = np.random.default_rng(seed + 10)
    weights = np.zeros((MAX_GENOMES, CRN_GENOME_SIZE), dtype=np.float32)

    for g in range(count):
        # Bootstrap reactions 0-2: viable sensory->action circuits
        # Reaction 0: light > 0.2 -> eat (chem 0 -> chem 12)
        _set_reaction(weights, g, 0, inp_a=0.03, inp_b=0.03, out=0.78,
                      th_a=0.2, th_b=0.2, rate=0.3, decay=0.1, rng=rng)
        # Reaction 1: energy > 0.3 -> divide (chem 1 -> chem 14)
        _set_reaction(weights, g, 1, inp_a=0.09, inp_b=0.09, out=0.90,
                      th_a=0.3, th_b=0.3, rate=0.4, decay=0.2, rng=rng)
        # Reaction 2: structure > 0.1 -> move (chem 2 -> chem 13)
        _set_reaction(weights, g, 2, inp_a=0.15, inp_b=0.15, out=0.84,
                      th_a=0.1, th_b=0.1, rate=0.2, decay=0.15, rng=rng)
        # Reactions 3-15: random with biased zone targeting
        for r in range(3, MAX_REACTIONS):
            base = r * PPR
            weights[g, base + 0] = rng.uniform(0.0, 0.5)   # input -> sensory
            weights[g, base + 1] = rng.uniform(0.0, 0.5)   # input -> sensory
            if rng.random() < 0.6:
                weights[g, base + 2] = rng.uniform(0.75, 1.0)  # output -> action
            else:
                weights[g, base + 2] = rng.uniform(0.5, 0.75)  # output -> hidden
            weights[g, base + 3] = rng.uniform(0.1, 0.5)
            weights[g, base + 4] = rng.uniform(0.1, 0.5)
            weights[g, base + 5] = rng.uniform(-0.3, 0.3)
            weights[g, base + 6] = rng.uniform(0.01, 0.3)

        # Action biases (112-115) and hidden decay rates (116-119)
        weights[g, REACT_END:REACT_END + NA] = (
            _DEFAULT_ACTION_BIASES + rng.normal(0, 0.02, NA).astype(np.float32))
        weights[g, REACT_END + NA:REACT_END + NA + NH] = (
            CRN_HIDDEN_DECAY + np.abs(rng.normal(0, 0.01, NH)).astype(np.float32))

    crn_weights.from_numpy(weights)

    ref_counts = np.zeros(MAX_GENOMES, dtype=np.int32)
    ref_counts[:count] = 1
    genome_ref_count.from_numpy(ref_counts)

    free_list = np.arange(count, MAX_GENOMES, dtype=np.int32)
    full_free = np.zeros(MAX_GENOMES, dtype=np.int32)
    full_free[:MAX_GENOMES - count] = free_list
    genome_free_list.from_numpy(full_free)
    genome_free_count[None] = MAX_GENOMES - count
    genome_count[None] = count

    genome_parent_id.from_numpy(np.full(MAX_GENOMES, -1, dtype=np.int32))
    genome_birth_tick.fill(0)

    # Initialize chemicals: sensory at 0.2, hidden at 0.0, action at biases
    init_chems = np.zeros((MAX_CELLS, NC), dtype=np.float32)
    init_chems[:, :NS] = 0.2
    for a in range(NA):
        init_chems[:, 12 + a] = _DEFAULT_ACTION_BIASES[a]
    crn_chemicals.from_numpy(init_chems)
