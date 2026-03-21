"""CTRNN genome with CfC (Closed-form Continuous-time) dynamics.

16 neurons in three zones:
  - Sensory (0-7):  Blended with environment each tick, fast time constants
  - Hidden (8-11):  Purely recurrent — memory, oscillators, gates
  - Action (12-15): Drive behavioral outputs via sigmoid firing

CfC update rule (Hasani et al. 2022):
  f_i = sigmoid(sum_j(w_ij * tanh(y_j)) + bias_i)
  y_i(t+1) = (y_i(t) + f_i * A_i) / (1 + 1/tau_i + f_i)

Genome layout (188 parameters):
  [0..175]   16 neurons x 11 (tau, bias, A, 4 weights, 4 target indices)
  [176..183]  8 input gains (sensory neurons)
  [184..187]  4 action biases (eat, move, divide, attack)
"""

import numpy as np
import taichi as ti

from config import (
    MAX_CELLS, MAX_GENOMES, CTRNN_NUM_NEURONS, CTRNN_NUM_SENSORY,
    CTRNN_NUM_HIDDEN, CTRNN_NUM_ACTION, CTRNN_RECURRENT_K,
    CTRNN_PARAMS_PER_NEURON, CTRNN_GENOME_SIZE, CTRNN_EXTRA_PARAMS,
    CTRNN_MUTATION_RATE_PERTURB, CTRNN_MUTATION_SIGMA,
    CTRNN_MUTATION_RATE_DUPLICATE, CTRNN_MUTATION_RATE_REWIRE,
    CTRNN_TAU_MIN, CTRNN_TAU_MAX, CTRNN_WEIGHT_BOUND,
    CTRNN_SENSORY_BLEND, CTRNN_AMPLITUDE_INIT,
    CTRNN_ACTION_GAIN, CTRNN_ACTION_CENTER,
    NUM_OUTPUTS, INITIAL_CELL_COUNT, RANDOM_SEED,
    ACTION_THRESHOLD, CRN_GRADIENT_TURN_MIN, CRN_BOND_SIGNAL_BLEND,
)
from cell.cell_state import cell_alive, cell_genome_id, cell_facing, cell_bonds
from cell.genome import sensory_inputs, action_outputs, needs_mutation

# CTRNN genome storage
ctrnn_weights = ti.field(dtype=ti.f32, shape=(MAX_GENOMES, CTRNN_GENOME_SIZE))

# Per-cell neuron activations (persistent between ticks)
ctrnn_neurons = ti.field(dtype=ti.f32, shape=(MAX_CELLS, CTRNN_NUM_NEURONS))

from cell.genome import (
    genome_ref_count, genome_count, genome_free_list, genome_free_count,
    genome_parent_id, genome_birth_tick,
    _mutation_event_count, _mutation_events, _pending_mutations, _pending_count,
)

NN = CTRNN_NUM_NEURONS        # 16
NS = CTRNN_NUM_SENSORY        # 8
NH = CTRNN_NUM_HIDDEN         # 4
NA = CTRNN_NUM_ACTION         # 4
RK = CTRNN_RECURRENT_K       # 4
PPR = CTRNN_PARAMS_PER_NEURON # 11
NEURON_END = NN * PPR         # 176

# Sensory map: same 8 channels as CRN
_SENSORY_MAP = ti.field(dtype=ti.i32, shape=(NS,))
# Action map: same 4 actions as CRN (eat, move_forward, divide, attack)
_ACTION_MAP = ti.field(dtype=ti.i32, shape=(NA,))


def _init_maps():
    """Initialize sensory and action mapping tables."""
    _SENSORY_MAP.from_numpy(np.array([0, 1, 2, 5, 6, 11, 14, 15], dtype=np.int32))
    _ACTION_MAP.from_numpy(np.array([3, 0, 5, 8], dtype=np.int32))


@ti.kernel
def evaluate_all_ctrnn():
    """CTRNN evaluation with CfC dynamics and separated zones."""
    for i in range(MAX_CELLS):
        if cell_alive[i] == 1:
            gid = cell_genome_id[i]

            # 1. Blend sensory inputs into sensory neurons (0-7)
            for s in range(NS):
                sens_idx = _SENSORY_MAP[s]
                sens_val = sensory_inputs[i, sens_idx]
                gain = ctrnn_weights[gid, NEURON_END + s]
                ctrnn_neurons[i, s] = (
                    ctrnn_neurons[i, s] * (1.0 - CTRNN_SENSORY_BLEND) +
                    sens_val * gain * CTRNN_SENSORY_BLEND
                )

            # 1b. Blend incoming bond signals into hidden neurons
            for h in range(NH):
                bond_sig_sum = 0.0
                bond_count = 0
                for b in range(4):
                    if cell_bonds[i, b] >= 0:
                        bond_sig_sum += sensory_inputs[i, 18 + b * 4 + h]
                        bond_count += 1
                if bond_count > 0:
                    avg_sig = bond_sig_sum / ti.cast(bond_count, ti.f32)
                    ctrnn_neurons[i, NS + h] += avg_sig * CRN_BOND_SIGNAL_BLEND

            # 2. CfC integration for hidden neurons (8-11) only.
            #    Sensory neurons (0-7) are pure blend (step 1).
            #    Action neurons (12-15) are feedforward readouts (step 4).
            for h in range(NH):
                n = NS + h
                base = n * PPR
                tau = ti.max(CTRNN_TAU_MIN, ti.min(CTRNN_TAU_MAX,
                        ti.abs(ctrnn_weights[gid, base + 0])))
                bias = ctrnn_weights[gid, base + 1]
                amp = ctrnn_weights[gid, base + 2]

                # f = sigmoid(recurrent_sum + bias)
                acc = bias
                for k in range(RK):
                    target_raw = ctrnn_weights[gid, base + 7 + k]
                    target_idx = ti.cast(ti.abs(target_raw) * NN, ti.i32) % NN
                    w = ctrnn_weights[gid, base + 3 + k]
                    acc += w * ti.tanh(ctrnn_neurons[i, target_idx])

                f = 1.0 / (1.0 + ti.exp(-acc))

                # CfC update: y(t+1) = (y(t) + f * A) / (1 + 1/tau + f)
                y = ctrnn_neurons[i, n]
                ctrnn_neurons[i, n] = (y + f * amp) / (1.0 + 1.0 / tau + f)

            # 3. Clamp all persistent neurons to [-5, 5]
            for n in range(NS + NH):
                ctrnn_neurons[i, n] = ti.max(-5.0, ti.min(5.0, ctrnn_neurons[i, n]))

            # 4. Compute action neurons as feedforward readouts (reset each tick)
            for a in range(NA):
                n = NS + NH + a
                base = n * PPR
                acc = ctrnn_weights[gid, base + 1]  # neuron bias as readout bias
                for k in range(RK):
                    target_raw = ctrnn_weights[gid, base + 7 + k]
                    target_idx = ti.cast(ti.abs(target_raw) * NN, ti.i32) % NN
                    w = ctrnn_weights[gid, base + 3 + k]
                    acc += w * ti.tanh(ctrnn_neurons[i, target_idx])
                ctrnn_neurons[i, n] = acc  # store for diagnostics

                # Sigmoid firing with action bias (same as CRN)
                action_idx = _ACTION_MAP[a]
                val = acc + ctrnn_weights[gid, NEURON_END + NS + a]
                p = 1.0 / (1.0 + ti.exp(
                    -CTRNN_ACTION_GAIN * (val - CTRNN_ACTION_CENTER)))
                if ti.random(ti.f32) < p:
                    action_outputs[i, action_idx] = ti.max(
                        val, ACTION_THRESHOLD + 0.1)
                else:
                    action_outputs[i, action_idx] = 0.0

            # 5. Turns: facing-aware gradient steering (same as CRN)
            grad_x = ctrnn_neurons[i, 3]
            grad_y = ctrnn_neurons[i, 4]
            facing = cell_facing[i]
            perp = 0.0
            if facing == 0:
                perp = grad_x
            elif facing == 1:
                perp = -grad_y
            elif facing == 2:
                perp = -grad_x
            else:
                perp = grad_y
            action_outputs[i, 1] = 0.0
            action_outputs[i, 2] = 0.0
            if ti.abs(perp) > CRN_GRADIENT_TURN_MIN:
                if perp < 0.0:
                    action_outputs[i, 1] = ACTION_THRESHOLD + 0.1
                else:
                    action_outputs[i, 2] = ACTION_THRESHOLD + 0.1

            # 6. Auxiliary actions from hidden neurons
            # Threshold 1.0 (not 0.5 like CRN): CfC steady-state is ~0.85
            # with tau=1.5, A=2. Need genuine recurrent drive to fire.
            if ctrnn_neurons[i, 8] > 1.0:
                action_outputs[i, 4] = ctrnn_neurons[i, 8]  # emit_signal
            else:
                action_outputs[i, 4] = 0.0
            if ctrnn_neurons[i, 9] > 1.0:
                action_outputs[i, 6] = ctrnn_neurons[i, 9]  # bond
            else:
                action_outputs[i, 6] = 0.0

            # 7. Zero unmapped outputs + bond signals from hidden 10-11
            action_outputs[i, 7] = 0.0   # unbond
            action_outputs[i, 9] = 0.0   # repair
            for ch in range(4):
                h_val = ctrnn_neurons[i, 10 + ch // 2]
                if h_val > 1.0:
                    action_outputs[i, 10 + ch] = h_val
                else:
                    action_outputs[i, 10 + ch] = 0.0


@ti.func
def _ctrnn_box_muller() -> ti.f32:
    u1 = ti.max(1e-7, ti.random(ti.f32))
    u2 = ti.random(ti.f32)
    return ti.sqrt(-2.0 * ti.log(u1)) * ti.cos(2.0 * 3.14159265358979 * u2)


@ti.kernel
def _collect_pending_ctrnn_mutations():
    _pending_count[None] = 0
    for i in range(MAX_CELLS):
        if needs_mutation[i] == 1:
            idx = ti.atomic_add(_pending_count[None], 1)
            _pending_mutations[idx] = i


@ti.kernel
def _apply_ctrnn_mutations_gpu(tick: ti.i32):
    """Apply CTRNN mutation operators on GPU."""
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

        # Copy and mutate all params
        for w in range(CTRNN_GENOME_SIZE):
            val = ctrnn_weights[parent_gid, w]
            mut_rate = CTRNN_MUTATION_RATE_PERTURB
            # Extra params (input gains, action biases) get half rate
            if w >= NEURON_END:
                mut_rate = CTRNN_MUTATION_RATE_PERTURB * 0.5
            if ti.random(ti.f32) < mut_rate:
                val += _ctrnn_box_muller() * CTRNN_MUTATION_SIGMA
                changed = 1
            ctrnn_weights[new_gid, w] = val

        # Clamp tau params to [TAU_MIN, TAU_MAX]
        for n in range(NN):
            base = n * PPR
            tau_val = ti.abs(ctrnn_weights[new_gid, base + 0])
            ctrnn_weights[new_gid, base + 0] = ti.max(
                CTRNN_TAU_MIN, ti.min(CTRNN_TAU_MAX, tau_val))

        # Clamp recurrent weights
        for n in range(NN):
            base = n * PPR
            for k in range(RK):
                ctrnn_weights[new_gid, base + 3 + k] = ti.max(
                    -CTRNN_WEIGHT_BOUND,
                    ti.min(CTRNN_WEIGHT_BOUND,
                           ctrnn_weights[new_gid, base + 3 + k]))

        # Target rewiring: change which neurons a neuron connects to
        for n in range(NN):
            if ti.random(ti.f32) < CTRNN_MUTATION_RATE_REWIRE:
                base = n * PPR
                k = ti.cast(ti.random(ti.f32) * RK, ti.i32) % RK
                ctrnn_weights[new_gid, base + 7 + k] = ti.random(ti.f32)
                changed = 1

        # Neuron duplication-divergence
        if ti.random(ti.f32) < CTRNN_MUTATION_RATE_DUPLICATE:
            src_n = ti.cast(ti.random(ti.f32) * NN, ti.i32) % NN
            dst_n = ti.cast(ti.random(ti.f32) * NN, ti.i32) % NN
            if src_n != dst_n:
                src_base = src_n * PPR
                dst_base = dst_n * PPR
                for p in range(PPR):
                    ctrnn_weights[new_gid, dst_base + p] = (
                        ctrnn_weights[new_gid, src_base + p] +
                        _ctrnn_box_muller() * CTRNN_MUTATION_SIGMA * 0.5
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
            # Reset daughter's neurons to clean state
            for n in range(NN):
                ctrnn_neurons[cell_idx, n] = 0.0

        needs_mutation[cell_idx] = 0


def process_ctrnn_mutations(tick=0):
    """Process CTRNN mutations for newly born cells."""
    _mutation_event_count[None] = 0
    _collect_pending_ctrnn_mutations()
    if _pending_count[None] > 0:
        _apply_ctrnn_mutations_gpu(tick)


# Action biases for feedforward readouts.
# Divide bias -0.35: threshold at energy ~35 (matches INITIAL_ENERGY).
# Cells divide at 50% when they reach starting energy, ~95% at energy 40.
# Balances reproduction rate against night-survival buffer.
_DEFAULT_ACTION_BIASES = np.array([0.3, 0.3, -0.35, -0.3], dtype=np.float32)


def init_ctrnn_genome_table(count: int = INITIAL_CELL_COUNT,
                            seed: int = RANDOM_SEED):
    """Initialize CTRNN genomes with bootstrap circuits."""
    _init_maps()
    rng = np.random.default_rng(seed + 20)
    weights = np.zeros((MAX_GENOMES, CTRNN_GENOME_SIZE), dtype=np.float32)
    n = 0.02  # noise scale

    for g in range(count):
        # Per-neuron params: tau, bias, A, 4 weights, 4 targets
        for nn_idx in range(NN):
            base = nn_idx * PPR

            # Zone-dependent tau initialization
            if nn_idx < NS:       # sensory: fast
                tau = 0.3 + rng.normal(0, 0.05)
            elif nn_idx < NS + NH:  # hidden: slow
                tau = 1.5 + rng.normal(0, 0.2)
            else:                   # action: moderate (unused, feedforward)
                tau = 0.7 + rng.normal(0, 0.1)
            weights[g, base + 0] = max(CTRNN_TAU_MIN, min(CTRNN_TAU_MAX, tau))

            # Bias: small random for sensory/hidden, zero for action readouts
            if nn_idx >= NS + NH:
                weights[g, base + 1] = 0.0
            else:
                weights[g, base + 1] = rng.normal(0, 0.1)

            # Amplitude (unused for action neurons since they're feedforward)
            weights[g, base + 2] = CTRNN_AMPLITUDE_INIT + rng.normal(0, 0.1)

            if nn_idx >= NS + NH:
                # Action neurons: zero weights/targets (bootstrap sets specific ones)
                for k in range(RK):
                    weights[g, base + 3 + k] = 0.0
                    weights[g, base + 7 + k] = 0.0
            else:
                # Sensory/hidden: random recurrent weights
                for k in range(RK):
                    weights[g, base + 3 + k] = rng.normal(0, 0.2)
                for k in range(RK):
                    weights[g, base + 7 + k] = rng.uniform(0.0, 1.0)

        # Bootstrap circuits: wire sensory->action readout connections
        # Action neurons are feedforward: val = bias + sum(w*tanh(y_target))
        # Then P(fire) = sigmoid(GAIN * (val + action_bias - CENTER))
        #
        # Neuron 12 (eat) <- neuron 0 (light), w=1.0
        # In light (y0~0.3): val=0+1.0*0.29=0.29, +bias0.3=0.59, P=94%
        # In dark  (y0~0.05): val=0.05, +0.3=0.35, P=sigmoid(-4.5)=1%
        base12 = 12 * PPR
        weights[g, base12 + 3] = 1.0 + rng.normal(0, n)
        weights[g, base12 + 7] = 0.03 + rng.normal(0, n)

        # Neuron 14 (divide) <- neuron 1 (energy), w=2.5
        # Threshold at E=40: P=0.1% at E=30, 5% at E=35, 50% at E=40, 99.8% at E=50
        base14 = 14 * PPR
        weights[g, base14 + 3] = 2.5 + rng.normal(0, n)
        weights[g, base14 + 7] = 0.09 + rng.normal(0, n)

        # Neuron 13 (move) <- neuron 2 (structure) + neuron 7 (waste), w=0.35 each
        # With structure (y2~0.5): val=0.35*tanh(0.5)=0.16, +bias0.3=0.46, P=18%
        base13 = 13 * PPR
        weights[g, base13 + 3] = 0.35 + rng.normal(0, n)
        weights[g, base13 + 7] = 0.15 + rng.normal(0, n)
        weights[g, base13 + 4] = 0.35 + rng.normal(0, n)
        weights[g, base13 + 8] = 0.47 + rng.normal(0, n)

        # Neuron 15 (attack): no bootstrap (action bias -0.3 keeps it suppressed)

        # Input gains (one per sensory neuron)
        for s in range(NS):
            weights[g, NEURON_END + s] = 1.0 + rng.normal(0, 0.05)

        # Action biases
        biases = _DEFAULT_ACTION_BIASES + rng.normal(0, n, NA).astype(np.float32)
        weights[g, NEURON_END + NS:NEURON_END + NS + NA] = biases

    ctrnn_weights.from_numpy(weights)

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

    # Initialize neurons: all zero (CfC dynamics will settle quickly)
    ctrnn_neurons.from_numpy(
        np.zeros((MAX_CELLS, NN), dtype=np.float32))
