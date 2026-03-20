"""Chemical Reaction Network (CRN) genome: biologically-inspired computational substrate.

Each cell has NUM_INTERNAL_CHEMICALS (8) internal chemical species with persistent
concentrations. MAX_REACTIONS (16) reaction rules define how chemicals interact.

Advantages over neural networks:
- Memory is free (chemical concentrations persist between ticks)
- Differentiation is natural (same genome, different environments → different states)
- Much smaller genome (112 floats vs 2638 for neural network)
- Richer dynamics: oscillations, cascades, thresholds
"""

import numpy as np
import taichi as ti

from config import (
    MAX_CELLS, MAX_GENOMES, NUM_INTERNAL_CHEMICALS, MAX_REACTIONS,
    CRN_PARAMS_PER_REACTION, CRN_GENOME_SIZE, CRN_ACTION_THRESHOLD,
    CRN_MUTATION_RATE_PERTURB, CRN_MUTATION_SIGMA,
    CRN_MUTATION_RATE_DUPLICATE, CRN_MUTATION_RATE_DELETE,
    CRN_MUTATION_RATE_REWIRE, NUM_OUTPUTS, INITIAL_CELL_COUNT,
    RANDOM_SEED, ATTACK_BIAS,
)
from cell.cell_state import cell_alive, cell_genome_id
from cell.genome import sensory_inputs, action_outputs, needs_mutation

# CRN genome storage: flattened reaction parameters
# Layout per reaction (7 floats): input_a, input_b, output, thresh_a, thresh_b, rate, decay
crn_weights = ti.field(dtype=ti.f32, shape=(MAX_GENOMES, CRN_GENOME_SIZE))

# Per-cell internal chemical concentrations (persistent between ticks)
crn_chemicals = ti.field(dtype=ti.f32, shape=(MAX_CELLS, NUM_INTERNAL_CHEMICALS))

# Reuse genome infrastructure from genome.py for ref counting / allocation
from cell.genome import (
    genome_ref_count, genome_count, genome_free_list, genome_free_count,
    genome_parent_id, genome_birth_tick,
    _mutation_event_count, _mutation_events, _pending_mutations, _pending_count,
)

NC = NUM_INTERNAL_CHEMICALS
MR = MAX_REACTIONS
PPR = CRN_PARAMS_PER_REACTION

# Sensory-to-chemical mapping: designed so that direct sensory-motor loops
# are beneficial (eat, divide, move) and destructive actions (attack) require
# evolved reaction chains or aging.
#
# Chemical 0 ← light_here [sensory 0]        → eat (light=photosynthesis)
# Chemical 1 ← energy_level [sensory 1]      → divide (energy→ready to divide)
# Chemical 2 ← structure [sensory 2]         → move_forward (fed cells move)
# Chemical 3 ← S_gradient_x [sensory 5]      → turn_left (gradients steer)
# Chemical 4 ← S_gradient_y [sensory 6]      → turn_right (gradients steer)
# Chemical 5 ← cell_ahead [sensory 11]       → bond (cell ahead→try to bond)
# Chemical 6 ← bond_count [sensory 14]       → emit_signal (bonded→signal)
# Chemical 7 ← age [sensory 15]              → attack (age-gated, must evolve)
_SENSORY_MAP = ti.field(dtype=ti.i32, shape=(NC,))

# Action-to-chemical mapping: matched to sensory mapping for direct loops.
# Chemical 0 → eat [action 3]           (light → eat)
# Chemical 1 → divide [action 5]        (energy → divide)
# Chemical 2 → move_forward [action 0]  (S gradient → move)
# Chemical 3 → turn_left [action 1]
# Chemical 4 → turn_right [action 2]
# Chemical 5 → bond [action 6]          (cell_ahead → bond, harmless)
# Chemical 6 → emit_signal [action 4]
# Chemical 7 → attack [action 8]        (age-gated: only fires when old enough)
_ACTION_MAP = ti.field(dtype=ti.i32, shape=(NC,))


def _init_maps():
    """Initialize sensory and action mapping tables."""
    #                       light, energy, structure, S_gx, S_gy, cell_ahead, bonds, age
    sensory_map = np.array([0,     1,      2,         5,    6,    11,         14,    15], dtype=np.int32)
    #                       eat, divide, move, turn_l, turn_r, bond, signal, attack
    action_map = np.array([3,    5,      0,    1,      2,      6,     4,     8], dtype=np.int32)
    _SENSORY_MAP.from_numpy(sensory_map)
    _ACTION_MAP.from_numpy(action_map)


@ti.kernel
def evaluate_all_crns():
    """CRN evaluation: set sensory chemicals → run reactions → decay → read actions."""
    for i in range(MAX_CELLS):
        if cell_alive[i] == 1:
            gid = cell_genome_id[i]

            # 1. Set sensory chemicals from environment
            for s in range(NC):
                sens_idx = _SENSORY_MAP[s]
                # Blend: sensory input dominates (0.7) while preserving some
                # memory (0.3) for temporal dynamics
                crn_chemicals[i, s] = (crn_chemicals[i, s] * 0.3 +
                                       sensory_inputs[i, sens_idx] * 0.7)

            # 2. Evaluate reactions
            for r in range(MR):
                base = r * PPR
                # Read reaction parameters
                input_a_raw = crn_weights[gid, base + 0]
                input_b_raw = crn_weights[gid, base + 1]
                output_raw = crn_weights[gid, base + 2]
                threshold_a = crn_weights[gid, base + 3]
                threshold_b = crn_weights[gid, base + 4]
                rate = crn_weights[gid, base + 5]
                # decay stored at base+6, applied below

                # Convert float indices to integer chemical indices
                input_a = ti.cast(ti.abs(input_a_raw) * NC, ti.i32) % NC
                input_b = ti.cast(ti.abs(input_b_raw) * NC, ti.i32) % NC
                output = ti.cast(ti.abs(output_raw) * NC, ti.i32) % NC

                # Fire reaction if both inputs exceed thresholds
                if (crn_chemicals[i, input_a] > ti.abs(threshold_a) and
                        crn_chemicals[i, input_b] > ti.abs(threshold_b)):
                    crn_chemicals[i, output] += rate

            # 3. Apply per-reaction decay to output chemicals
            for r in range(MR):
                base = r * PPR
                output_raw = crn_weights[gid, base + 2]
                output = ti.cast(ti.abs(output_raw) * NC, ti.i32) % NC
                decay = ti.abs(crn_weights[gid, base + 6])
                decay = ti.min(1.0, decay)  # clamp decay rate
                crn_chemicals[i, output] *= (1.0 - decay * 0.1)

            # Clamp all chemicals to [0, 5] to prevent runaway
            for c in range(NC):
                crn_chemicals[i, c] = ti.max(0.0, ti.min(5.0, crn_chemicals[i, c]))

            # 4. Read action chemicals and set action outputs
            for c in range(NC):
                action_idx = _ACTION_MAP[c]
                if crn_chemicals[i, c] > CRN_ACTION_THRESHOLD:
                    action_outputs[i, action_idx] = crn_chemicals[i, c]
                else:
                    action_outputs[i, action_idx] = 0.0

            # Zero out unmapped action outputs (unbond=7, repair=9, bond_signals=10-13)
            action_outputs[i, 7] = 0.0
            action_outputs[i, 9] = 0.0
            for ao in range(10, NUM_OUTPUTS):
                action_outputs[i, ao] = 0.0


@ti.func
def _crn_box_muller() -> ti.f32:
    u1 = ti.max(1e-7, ti.random(ti.f32))
    u2 = ti.random(ti.f32)
    return ti.sqrt(-2.0 * ti.log(u1)) * ti.cos(2.0 * 3.14159265358979 * u2)


@ti.kernel
def _collect_pending_crn_mutations():
    """Collect indices of cells needing CRN mutation."""
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

        # Allocate new genome slot
        slot_idx = ti.atomic_sub(genome_free_count[None], 1) - 1
        if slot_idx < 0:
            ti.atomic_add(genome_free_count[None], 1)
            ti.atomic_add(genome_ref_count[parent_gid], 1)
            needs_mutation[cell_idx] = 0
            continue

        new_gid = genome_free_list[slot_idx]
        ti.atomic_add(genome_count[None], 1)

        changed = 0

        # Copy parent weights and apply mutations
        for w in range(CRN_GENOME_SIZE):
            val = crn_weights[parent_gid, w]

            # Parameter perturbation
            if ti.random(ti.f32) < CRN_MUTATION_RATE_PERTURB:
                val += _crn_box_muller() * CRN_MUTATION_SIGMA
                changed = 1

            crn_weights[new_gid, w] = val

        # Reaction rewiring: change input/output chemical indices
        for r in range(MR):
            base = r * PPR
            if ti.random(ti.f32) < CRN_MUTATION_RATE_REWIRE:
                # Randomly rewire one of: input_a, input_b, or output
                which = ti.cast(ti.random(ti.f32) * 3.0, ti.i32) % 3
                crn_weights[new_gid, base + which] = ti.random(ti.f32)
                changed = 1

        # Reaction deletion: zero the rate
        for r in range(MR):
            if ti.random(ti.f32) < CRN_MUTATION_RATE_DELETE:
                crn_weights[new_gid, r * PPR + 5] = 0.0  # rate = 0
                changed = 1

        # Reaction duplication: copy one reaction to another slot with noise
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

        needs_mutation[cell_idx] = 0


def process_crn_mutations(tick=0):
    """Process CRN mutations for newly born cells."""
    _mutation_event_count[None] = 0
    _collect_pending_crn_mutations()
    if _pending_count[None] > 0:
        _apply_crn_mutations_gpu(tick)


def init_crn_genome_table(count: int = INITIAL_CELL_COUNT, seed: int = RANDOM_SEED):
    """Initialize CRN genomes with random reaction parameters."""
    _init_maps()

    rng = np.random.default_rng(seed + 10)

    weights = np.zeros((MAX_GENOMES, CRN_GENOME_SIZE), dtype=np.float32)

    for g in range(count):
        for r in range(MAX_REACTIONS):
            base = r * CRN_PARAMS_PER_REACTION
            # Random chemical indices (stored as floats, converted in kernel)
            weights[g, base + 0] = rng.random()  # input_a
            weights[g, base + 1] = rng.random()  # input_b
            weights[g, base + 2] = rng.random()  # output
            # Thresholds: moderate so reactions can fire
            weights[g, base + 3] = rng.uniform(0.1, 0.5)  # threshold_a
            weights[g, base + 4] = rng.uniform(0.1, 0.5)  # threshold_b
            # Rate: small to avoid runaway
            weights[g, base + 5] = rng.uniform(-0.3, 0.3)  # rate
            # Decay: moderate
            weights[g, base + 6] = rng.uniform(0.01, 0.3)  # decay

    crn_weights.from_numpy(weights)

    # Initialize reference counts and free list through genome.py's infrastructure
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

    # Initialize internal chemicals near threshold for faster bootstrap
    init_chems = np.full((MAX_CELLS, NUM_INTERNAL_CHEMICALS), 0.2,
                         dtype=np.float32)
    crn_chemicals.from_numpy(init_chems)
