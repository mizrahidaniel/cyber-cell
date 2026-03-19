"""CyberCell simulation configuration. Single source of truth for all tunable parameters."""

import math

# =============================================================================
# System
# =============================================================================
BACKEND = "cpu"  # "cpu" for macOS dev (Metal GUI crash workaround), "cuda" for Windows
HEADLESS = False
RANDOM_SEED = 42

# =============================================================================
# World
# =============================================================================
GRID_WIDTH = 500
GRID_HEIGHT = 500
LIGHT_ZONE_END = 166          # x < 166 = bright
DIM_ZONE_END = 333            # 166 <= x < 333 = dim
DAY_LENGTH = 1000             # ticks per full day/night cycle
LIGHT_BRIGHT = 1.0
LIGHT_DIM = 0.3
LIGHT_DARK = 0.0

# =============================================================================
# Chemistry
# =============================================================================
DIFFUSION_RATE_S = 0.01
DIFFUSION_RATE_R = 0.005
DIFFUSION_RATE_G = 0.3
E_DECAY_FLAT = 0.02           # flat internal energy decay per tick (not percentage)
DECAY_RATE_S = 0.001
DECAY_RATE_R = 0.001
DECAY_RATE_G = 0.05
DEPOSIT_REPLENISH_RATE = 0.008
NUM_DEPOSITS_S = 200
NUM_DEPOSITS_R = 150
DEPOSIT_CLUSTER_RADIUS = 15
DEPOSIT_CLUSTER_AMOUNT = 5.0

# =============================================================================
# Cells
# =============================================================================
MAX_CELLS = 50000
INITIAL_CELL_COUNT = 1000
MAX_CELL_AGE = 5000
INITIAL_ENERGY = 25.0
INITIAL_STRUCTURE = 25.0
INITIAL_REPMAT = 10.0
MEMBRANE_INITIAL = 100.0
ENERGY_ZERO_MEMBRANE_DAMAGE = 5.0  # membrane damage per tick when energy is 0
AGE_MEMBRANE_DECAY = 1.0           # membrane loss per tick past max age

# =============================================================================
# Energy costs
# =============================================================================
BASAL_METABOLISM = 0.05
MOVE_COST = 0.3
TURN_COST = 0.02
EAT_COST = 0.02
SIGNAL_COST = 0.1
DIVIDE_COST = 20.0
DIVIDE_R_COST = 5.0
BOND_COST = 0.05
ATTACK_COST = 0.5
REPAIR_COST = 0.1
REPAIR_S_COST = 0.5
REPAIR_MEMBRANE_GAIN = 5.0
NETWORK_COST = 0.01

# =============================================================================
# Energy income
# =============================================================================
PHOTOSYNTHESIS_RATE = 0.5
S_ENERGY_VALUE = 0.1
R_ENERGY_VALUE = 0.2
EAT_ABSORB_CAP = 1.0          # max chemical absorbed per eat action per tick
ATTACK_MEMBRANE_DAMAGE = 5.0

# =============================================================================
# Genome / Neural Network
# =============================================================================
MAX_GENOMES = 50000
NUM_INPUTS = 16
NETWORK_HIDDEN_SIZE = 32
NUM_OUTPUTS = 10
# 16*32 + 32 + 32*32 + 32 + 32*10 + 10 = 1930
GENOME_SIZE = (NUM_INPUTS * NETWORK_HIDDEN_SIZE + NETWORK_HIDDEN_SIZE +
               NETWORK_HIDDEN_SIZE * NETWORK_HIDDEN_SIZE + NETWORK_HIDDEN_SIZE +
               NETWORK_HIDDEN_SIZE * NUM_OUTPUTS + NUM_OUTPUTS)
SEED_WEIGHT_SIGMA = 0.01      # near-zero initial weights to prevent population crash
ACTION_THRESHOLD = 0.5

# Weight layout offsets
W1_END = NUM_INPUTS * NETWORK_HIDDEN_SIZE                    # 512
B1_END = W1_END + NETWORK_HIDDEN_SIZE                        # 544
W2_END = B1_END + NETWORK_HIDDEN_SIZE * NETWORK_HIDDEN_SIZE  # 1568
B2_END = W2_END + NETWORK_HIDDEN_SIZE                        # 1600
W3_END = B2_END + NETWORK_HIDDEN_SIZE * NUM_OUTPUTS           # 1920
B3_END = W3_END + NUM_OUTPUTS                                # 1930

# =============================================================================
# Mutation
# =============================================================================
MUTATION_RATE_PERTURB = 0.01
MUTATION_SIGMA = 0.1
MUTATION_RATE_RESET = 0.001
MUTATION_RATE_KNOCKOUT = 0.0005

# =============================================================================
# Reproduction
# =============================================================================
PARENT_RESOURCE_SHARE = 0.6
DAUGHTER_RESOURCE_SHARE = 0.4

# =============================================================================
# Bonding
# =============================================================================
BOND_SHARE_RATE = 0.1

# =============================================================================
# Visualization
# =============================================================================
GUI_SCALE = 1
GUI_FPS = 60

# =============================================================================
# Analysis
# =============================================================================
SNAPSHOT_INTERVAL = 1000       # ticks between snapshots
GENOME_GC_INTERVAL = 1000     # ticks between genome garbage collection
