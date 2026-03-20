"""Genome table, neural network evaluation, and mutation operators."""

import numpy as np
import taichi as ti

from config import (
    MAX_CELLS, MAX_GENOMES, GENOME_SIZE, NUM_INPUTS, NETWORK_HIDDEN_SIZE,
    NUM_OUTPUTS, SEED_WEIGHT_SIGMA, INITIAL_CELL_COUNT,
    W1_END, B1_END, W2_END, B2_END, W3_END, B3_END,
    MUTATION_RATE_PERTURB, MUTATION_SIGMA, MUTATION_RATE_RESET,
    MUTATION_RATE_KNOCKOUT, RANDOM_SEED, ATTACK_BIAS,
)
from cell.cell_state import cell_alive, cell_genome_id

# Genome weight storage: flattened neural net weights
genome_weights = ti.field(dtype=ti.f32, shape=(MAX_GENOMES, GENOME_SIZE))

# Reference counting: number of alive cells using each genome
genome_ref_count = ti.field(dtype=ti.i32, shape=(MAX_GENOMES,))

# Genome allocation tracking
genome_count = ti.field(dtype=ti.i32, shape=())
genome_free_list = ti.field(dtype=ti.i32, shape=(MAX_GENOMES,))
genome_free_count = ti.field(dtype=ti.i32, shape=())

# Lineage tracking
genome_parent_id = ti.field(dtype=ti.i32, shape=(MAX_GENOMES,))
genome_birth_tick = ti.field(dtype=ti.i32, shape=(MAX_GENOMES,))
_mutation_events = ti.field(dtype=ti.i32, shape=(MAX_CELLS, 3))  # parent_gid, child_gid, tick
_mutation_event_count = ti.field(dtype=ti.i32, shape=())

# Per-cell scratch space for network evaluation
hidden1 = ti.field(dtype=ti.f32, shape=(MAX_CELLS, NETWORK_HIDDEN_SIZE))
hidden2 = ti.field(dtype=ti.f32, shape=(MAX_CELLS, NETWORK_HIDDEN_SIZE))
action_outputs = ti.field(dtype=ti.f32, shape=(MAX_CELLS, NUM_OUTPUTS))

# Sensory input buffer (filled by sensing.py)
sensory_inputs = ti.field(dtype=ti.f32, shape=(MAX_CELLS, NUM_INPUTS))

# Mutation flag for newly born cells
needs_mutation = ti.field(dtype=ti.i32, shape=(MAX_CELLS,))

# Pending mutation list: cell indices that need mutation (filled by GPU, consumed by GPU)
_pending_mutations = ti.field(dtype=ti.i32, shape=(MAX_CELLS,))
_pending_count = ti.field(dtype=ti.i32, shape=())

H = NETWORK_HIDDEN_SIZE


@ti.kernel
def evaluate_all_networks():
    """Forward pass: inputs -> 32 hidden (tanh) -> 32 hidden (tanh) -> 10 outputs (sigmoid)."""
    for i in range(MAX_CELLS):
        if cell_alive[i] == 1:
            gid = cell_genome_id[i]

            # Layer 1: inputs -> hidden1
            for h in range(H):
                acc = genome_weights[gid, W1_END + h]  # bias
                for inp in range(NUM_INPUTS):
                    acc += sensory_inputs[i, inp] * genome_weights[gid, h * NUM_INPUTS + inp]
                hidden1[i, h] = ti.tanh(acc)

            # Layer 2: hidden1 -> hidden2
            for h in range(H):
                acc = genome_weights[gid, B1_END + H * H + h]  # bias at B2 offset
                for h2 in range(H):
                    acc += hidden1[i, h2] * genome_weights[gid, B1_END + h * H + h2]
                hidden2[i, h] = ti.tanh(acc)

            # Layer 3: hidden2 -> outputs
            for o in range(NUM_OUTPUTS):
                acc = genome_weights[gid, W3_END + o]  # bias
                for h in range(H):
                    acc += hidden2[i, h] * genome_weights[gid, B2_END + o * H + h]
                action_outputs[i, o] = 1.0 / (1.0 + ti.exp(-acc))  # sigmoid


def init_genome_table(count: int = INITIAL_CELL_COUNT, seed: int = RANDOM_SEED):
    """Create initial genomes with near-zero weights for diversity."""
    rng = np.random.default_rng(seed + 2)

    # Create one unique genome per initial cell
    weights = rng.normal(0.0, SEED_WEIGHT_SIGMA,
                         size=(count, GENOME_SIZE)).astype(np.float32)

    # Bias output layer biases for viable starting phenotype:
    #   Output 5 (divide): bias=0.5 → sigmoid(0.5)≈0.62, fires reliably
    #   Output 8 (attack): bias=ATTACK_BIAS → suppressed but evolvable
    #   Output 0 (move): bias=0.0 → sigmoid(0)=0.5, neutral (one mutation can activate)
    #   Others: bias=0 → 0.5, at threshold boundary
    # Biases are at offsets W3_END to B3_END
    output_biases = np.array([
         0.0,          # 0: move_forward — neutral, easily activated by mutation
         0.0,          # 1: turn_left
         0.0,          # 2: turn_right
         0.0,          # 3: eat (passive eating handles this now)
         0.0,          # 4: emit_signal
         0.5,          # 5: divide — fires when conditions met
         0.0,          # 6: bond
         0.0,          # 7: unbond
         ATTACK_BIAS,  # 8: attack — suppressed but reachable by 1-2 mutations
         0.0,          # 9: repair
    ], dtype=np.float32)

    for g in range(count):
        weights[g, W3_END:B3_END] += output_biases

    # Write to Taichi field
    full_weights = np.zeros((MAX_GENOMES, GENOME_SIZE), dtype=np.float32)
    full_weights[:count] = weights
    genome_weights.from_numpy(full_weights)

    # Set ref counts
    ref_counts = np.zeros(MAX_GENOMES, dtype=np.int32)
    ref_counts[:count] = 1
    genome_ref_count.from_numpy(ref_counts)

    # Initialize free list with remaining slots
    free_list = np.arange(count, MAX_GENOMES, dtype=np.int32)
    full_free = np.zeros(MAX_GENOMES, dtype=np.int32)
    full_free[:MAX_GENOMES - count] = free_list
    genome_free_list.from_numpy(full_free)
    genome_free_count[None] = MAX_GENOMES - count
    genome_count[None] = count

    # Initialize lineage tracking
    genome_parent_id.from_numpy(np.full(MAX_GENOMES, -1, dtype=np.int32))
    genome_birth_tick.fill(0)


def allocate_genome_python() -> int:
    """Allocate a genome slot from the free list (Python-side)."""
    count = genome_free_count[None]
    if count <= 0:
        return -1
    count -= 1
    genome_free_count[None] = count
    slot = int(genome_free_list[count])
    genome_count[None] = genome_count[None] + 1
    return slot


def deallocate_genome_python(idx: int):
    """Return a genome slot to the free list (Python-side)."""
    count = genome_free_count[None]
    genome_free_list[count] = idx
    genome_free_count[None] = count + 1
    genome_count[None] = genome_count[None] - 1


@ti.func
def _box_muller_normal() -> ti.f32:
    """Generate a standard normal random number on GPU using Box-Muller transform."""
    u1 = ti.max(1e-7, ti.random(ti.f32))  # avoid log(0)
    u2 = ti.random(ti.f32)
    return ti.sqrt(-2.0 * ti.log(u1)) * ti.cos(2.0 * 3.14159265358979 * u2)


@ti.kernel
def _collect_pending_mutations():
    """Collect indices of cells needing mutation into a compact list."""
    _pending_count[None] = 0
    for i in range(MAX_CELLS):
        if needs_mutation[i] == 1:
            idx = ti.atomic_add(_pending_count[None], 1)
            _pending_mutations[idx] = i


@ti.kernel
def _apply_mutations_gpu(tick: ti.i32):
    """Apply mutation operators entirely on GPU — no CPU transfer needed.

    For each cell needing mutation:
    1. Allocate a new genome slot (sequential via atomic)
    2. Copy parent weights to new genome
    3. Apply perturbation, reset, and knockout mutations using ti.random()
    4. Update cell's genome_id and ref counts
    5. Record lineage events for new genomes
    """
    for idx in range(_pending_count[None]):
        cell_idx = _pending_mutations[idx]
        parent_gid = cell_genome_id[cell_idx]

        # Allocate new genome slot
        slot_idx = ti.atomic_sub(genome_free_count[None], 1) - 1
        if slot_idx < 0:
            # Allocation failed — undo and keep parent genome
            ti.atomic_add(genome_free_count[None], 1)
            ti.atomic_add(genome_ref_count[parent_gid], 1)
            needs_mutation[cell_idx] = 0
            continue

        new_gid = genome_free_list[slot_idx]
        ti.atomic_add(genome_count[None], 1)

        # Copy parent weights and apply mutations
        changed = 0
        for w in range(GENOME_SIZE):
            val = genome_weights[parent_gid, w]

            # Weight perturbation
            if ti.random(ti.f32) < MUTATION_RATE_PERTURB:
                val += _box_muller_normal() * MUTATION_SIGMA
                changed = 1

            # Weight reset
            if ti.random(ti.f32) < MUTATION_RATE_RESET:
                val = ti.random(ti.f32) * 2.0 - 1.0  # uniform [-1, 1]
                changed = 1

            genome_weights[new_gid, w] = val

        # Node knockout: hidden layer 1 → layer 2
        for h in range(H):
            if ti.random(ti.f32) < MUTATION_RATE_KNOCKOUT:
                for h2 in range(H):
                    genome_weights[new_gid, B1_END + h2 * H + h] = 0.0
                changed = 1

        # Node knockout: hidden layer 2 → outputs
        for h in range(H):
            if ti.random(ti.f32) < MUTATION_RATE_KNOCKOUT:
                for o in range(NUM_OUTPUTS):
                    genome_weights[new_gid, B2_END + o * H + h] = 0.0
                changed = 1

        if changed == 0:
            # No mutation occurred — free the new genome and share parent's
            ti.atomic_add(genome_free_count[None], 1)
            genome_free_list[slot_idx] = new_gid
            ti.atomic_sub(genome_count[None], 1)
            ti.atomic_add(genome_ref_count[parent_gid], 1)
        else:
            # Mutation occurred — update refs and record lineage
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


def process_mutations(rng=None, tick=0):
    """Process mutations for all newly born cells (GPU-side).

    The rng parameter is kept for API compatibility but is unused —
    all randomness now comes from ti.random() on the GPU.
    """
    _mutation_event_count[None] = 0
    _collect_pending_mutations()
    if _pending_count[None] > 0:
        _apply_mutations_gpu(tick)


def get_mutation_events():
    """Read mutation events recorded this tick. Returns list of (parent_gid, child_gid, tick)."""
    count = int(_mutation_event_count[None])
    if count == 0:
        return []
    count = min(count, MAX_CELLS)
    events_np = _mutation_events.to_numpy()[:count]
    return [(int(events_np[i, 0]), int(events_np[i, 1]), int(events_np[i, 2]))
            for i in range(count)]


@ti.kernel
def _recount_genome_refs():
    """Recompute genome ref counts from scratch."""
    for g in range(MAX_GENOMES):
        genome_ref_count[g] = 0
    for i in range(MAX_CELLS):
        if cell_alive[i] == 1:
            ti.atomic_add(genome_ref_count[cell_genome_id[i]], 1)


def garbage_collect_genomes():
    """Free genomes with zero references."""
    _recount_genome_refs()
    ref_counts = genome_ref_count.to_numpy()

    # Rebuild free list from ref counts
    free_indices = []
    active_count = 0
    for g in range(MAX_GENOMES):
        if ref_counts[g] == 0:
            free_indices.append(g)
        else:
            active_count += 1

    free_arr = np.array(free_indices, dtype=np.int32)
    full_free = np.zeros(MAX_GENOMES, dtype=np.int32)
    full_free[:len(free_arr)] = free_arr
    genome_free_list.from_numpy(full_free)
    genome_free_count[None] = len(free_arr)
    genome_count[None] = active_count
