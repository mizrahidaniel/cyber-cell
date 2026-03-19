"""Genome table, neural network evaluation, and mutation operators."""

import numpy as np
import taichi as ti

from config import (
    MAX_CELLS, MAX_GENOMES, GENOME_SIZE, NUM_INPUTS, NETWORK_HIDDEN_SIZE,
    NUM_OUTPUTS, SEED_WEIGHT_SIGMA, INITIAL_CELL_COUNT,
    W1_END, B1_END, W2_END, B2_END, W3_END, B3_END,
    MUTATION_RATE_PERTURB, MUTATION_SIGMA, MUTATION_RATE_RESET,
    MUTATION_RATE_KNOCKOUT, RANDOM_SEED,
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

# Per-cell scratch space for network evaluation
hidden1 = ti.field(dtype=ti.f32, shape=(MAX_CELLS, NETWORK_HIDDEN_SIZE))
hidden2 = ti.field(dtype=ti.f32, shape=(MAX_CELLS, NETWORK_HIDDEN_SIZE))
action_outputs = ti.field(dtype=ti.f32, shape=(MAX_CELLS, NUM_OUTPUTS))

# Sensory input buffer (filled by sensing.py)
sensory_inputs = ti.field(dtype=ti.f32, shape=(MAX_CELLS, NUM_INPUTS))

# Mutation flag for newly born cells
needs_mutation = ti.field(dtype=ti.i32, shape=(MAX_CELLS,))

H = NETWORK_HIDDEN_SIZE


@ti.kernel
def evaluate_all_networks():
    """Forward pass: 16 inputs -> 32 hidden (tanh) -> 32 hidden (tanh) -> 10 outputs (sigmoid)."""
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
    #   Output 8 (attack): bias=-1.0 → sigmoid(-1)≈0.27, suppressed
    #   Output 0 (move): bias=-0.3 → sigmoid(-0.3)≈0.43, mostly off
    #   Others: bias=0 → 0.5, at threshold boundary
    # Biases are at offsets W3_END to B3_END (positions 1920-1929)
    output_biases = np.array([
        -0.3,   # 0: move_forward — mostly off initially
         0.0,   # 1: turn_left
         0.0,   # 2: turn_right
         0.0,   # 3: eat (passive eating handles this now)
         0.0,   # 4: emit_signal
         0.5,   # 5: divide — fires when conditions met
         0.0,   # 6: bond
         0.0,   # 7: unbond
        -1.0,   # 8: attack — suppressed to prevent random killing
         0.0,   # 9: repair
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


def mutate_genome(parent_genome_id: int, rng: np.random.Generator) -> tuple[np.ndarray, bool]:
    """Apply mutation operators to a copy of the parent genome.

    Returns (mutated_weights, changed) where changed indicates if any mutation occurred.
    """
    weights = np.zeros(GENOME_SIZE, dtype=np.float32)
    for w in range(GENOME_SIZE):
        weights[w] = float(genome_weights[parent_genome_id, w])

    changed = False

    # Weight perturbation
    perturb_mask = rng.random(GENOME_SIZE) < MUTATION_RATE_PERTURB
    if perturb_mask.any():
        weights[perturb_mask] += rng.normal(0.0, MUTATION_SIGMA,
                                            size=int(perturb_mask.sum())).astype(np.float32)
        changed = True

    # Weight reset
    reset_mask = rng.random(GENOME_SIZE) < MUTATION_RATE_RESET
    if reset_mask.any():
        weights[reset_mask] = rng.uniform(-1.0, 1.0,
                                          size=int(reset_mask.sum())).astype(np.float32)
        changed = True

    # Node knockout: zero all outgoing weights of a hidden node
    # Hidden layer 1 nodes -> their outputs go to layer 2
    for h in range(NETWORK_HIDDEN_SIZE):
        if rng.random() < MUTATION_RATE_KNOCKOUT:
            # Zero column h in W2 (weights from hidden1[h] to all hidden2 nodes)
            for h2 in range(NETWORK_HIDDEN_SIZE):
                weights[B1_END + h2 * NETWORK_HIDDEN_SIZE + h] = 0.0
            changed = True

    # Hidden layer 2 nodes -> their outputs go to layer 3
    for h in range(NETWORK_HIDDEN_SIZE):
        if rng.random() < MUTATION_RATE_KNOCKOUT:
            # Zero column h in W3 (weights from hidden2[h] to all output nodes)
            for o in range(NUM_OUTPUTS):
                weights[B2_END + o * NETWORK_HIDDEN_SIZE + h] = 0.0
            changed = True

    return weights, changed


def process_mutations(rng: np.random.Generator):
    """Process mutations for all newly born cells (Python-side loop)."""
    mutation_flags = needs_mutation.to_numpy()
    birth_indices = np.where(mutation_flags == 1)[0]

    for idx in birth_indices:
        parent_gid = int(cell_genome_id[idx])
        weights, changed = mutate_genome(parent_gid, rng)

        if changed:
            new_gid = allocate_genome_python()
            if new_gid >= 0:
                for w in range(GENOME_SIZE):
                    genome_weights[new_gid, w] = float(weights[w])
                genome_ref_count[new_gid] = 1
                genome_ref_count[parent_gid] -= 1
                cell_genome_id[idx] = new_gid
            # If allocation failed, keep parent genome
        else:
            genome_ref_count[parent_gid] += 1

        needs_mutation[idx] = 0


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
    genome_ct = genome_count[None]

    for g in range(MAX_GENOMES):
        if ref_counts[g] == 0 and g < genome_ct:
            # Check if this genome was ever allocated (non-zero in the table)
            # We only deallocate if it's tracked as active
            pass  # Simplified: GC runs on recount, dead genomes auto-freed

    # More precise: rebuild free list from ref counts
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
