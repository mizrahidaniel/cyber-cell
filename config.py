"""CyberCell simulation configuration. Single source of truth for all tunable parameters."""

import math

# =============================================================================
# System
# =============================================================================
BACKEND = "auto"  # "auto" benchmarks and caches; CUDA ~2x faster across all population sizes
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
DIFFUSION_RATE_R = 0.03
DIFFUSION_RATE_G = 0.3
E_DECAY_FLAT = 0.02           # flat internal energy decay per tick (not percentage)
DECAY_RATE_S = 0.001
DECAY_RATE_R = 0.001
DECAY_RATE_G = 0.05
DEPOSIT_REPLENISH_RATE = 0.012
NUM_DEPOSITS_S = 200
NUM_DEPOSITS_R = 200
DEPOSIT_CLUSTER_RADIUS = 10       # tighter clusters for patchier distribution
DEPOSIT_CLUSTER_AMOUNT = 10.0     # higher concentration per deposit
DEPOSIT_RELOCATE_INTERVAL = 25000 # ticks between partial deposit relocation
DEPOSIT_RELOCATE_FRACTION = 0.2   # fraction of deposits relocated each interval

# =============================================================================
# Sensing
# =============================================================================
GRADIENT_NOISE_SIGMA = 0.15       # Gaussian noise on gradient sensing

# =============================================================================
# Cells
# =============================================================================
MAX_CELLS = 50000
INITIAL_CELL_COUNT = 1000
MAX_CELL_AGE = 5000
INITIAL_ENERGY = 35.0
INITIAL_STRUCTURE = 25.0
INITIAL_REPMAT = 10.0
MEMBRANE_INITIAL = 100.0
ENERGY_ZERO_MEMBRANE_DAMAGE = 5.0  # membrane damage per tick when energy is 0
AGE_MEMBRANE_DECAY = 1.0           # membrane loss per tick past max age

# =============================================================================
# Energy costs
# =============================================================================
BASAL_METABOLISM = 0.08
MOVE_COST = 0.1
TURN_COST = 0.02
EAT_COST = 0.02
SIGNAL_COST = 0.1
DIVIDE_COST = 20.0
DIVIDE_R_COST = 5.0
BOND_COST = 0.01
ATTACK_COST = 0.3
REPAIR_COST = 0.1
REPAIR_S_COST = 0.5
REPAIR_MEMBRANE_GAIN = 5.0
NETWORK_COST = 0.01

# =============================================================================
# Energy income
# =============================================================================
PHOTOSYNTHESIS_RATE = 0.45
S_ENERGY_VALUE = 0.3
R_ENERGY_VALUE = 0.5
EAT_ABSORB_CAP = 2.0          # max chemical absorbed per eat action per tick
PASSIVE_EAT_CAP = 0.05        # max chemical absorbed passively per tick (no neural net)
ATTACK_MEMBRANE_DAMAGE = 8.0

# =============================================================================
# Genome / Neural Network
# =============================================================================
MAX_GENOMES = 50000
# 18 base inputs + 4 bonds * 4 signal channels = 34 total
NUM_INPUTS = 34
NETWORK_HIDDEN_SIZE = 32
# 10 base outputs + 4 bond signal outputs = 14 total
NUM_OUTPUTS = 14
# 34*32 + 32 + 32*32 + 32 + 32*14 + 14 = 2638
ATTACK_BIAS = -0.3            # initial attack output bias (sigmoid(-0.3)=0.43, evolvable)
GENOME_SIZE = (NUM_INPUTS * NETWORK_HIDDEN_SIZE + NETWORK_HIDDEN_SIZE +
               NETWORK_HIDDEN_SIZE * NETWORK_HIDDEN_SIZE + NETWORK_HIDDEN_SIZE +
               NETWORK_HIDDEN_SIZE * NUM_OUTPUTS + NUM_OUTPUTS)
SEED_WEIGHT_SIGMA = 0.01      # near-zero initial weights to prevent population crash
GRADIENT_SCALE_S = 3.0        # amplify S gradient for neural network input
GRADIENT_SCALE_R = 5.0        # amplify R gradient for neural network input
ACTION_THRESHOLD = 0.5

# Weight layout offsets
W1_END = NUM_INPUTS * NETWORK_HIDDEN_SIZE                    # 1088
B1_END = W1_END + NETWORK_HIDDEN_SIZE                        # 1120
W2_END = B1_END + NETWORK_HIDDEN_SIZE * NETWORK_HIDDEN_SIZE  # 2144
B2_END = W2_END + NETWORK_HIDDEN_SIZE                        # 2176
W3_END = B2_END + NETWORK_HIDDEN_SIZE * NUM_OUTPUTS           # 2624
B3_END = W3_END + NUM_OUTPUTS                                # 2638

# =============================================================================
# Mutation
# =============================================================================
MUTATION_RATE_PERTURB = 0.03
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
BOND_INITIAL_STRENGTH = 0.5      # starting strength on bond formation
BOND_DECAY_RATE = 0.02           # strength lost per tick without reinforcement
BOND_REINFORCE_RATE = 0.03       # strength gained per tick when both cells fire bond
BOND_BREAK_THRESHOLD = 0.05      # auto-break below this strength
BOND_TRANSFER_LOSS = 0.3         # fraction of shared resources destroyed in transit
BOND_SIGNAL_CHANNELS = 4         # number of signal floats per bond direction
R_LIGHT_ZONE_FRACTION = 0.15    # fraction of R deposits placed in the light zone

# =============================================================================
# Visualization
# =============================================================================
GUI_SCALE = 1
GUI_FPS = 60

# =============================================================================
# Analysis
# =============================================================================
SNAPSHOT_INTERVAL = 1000       # ticks between snapshots
SPATIAL_SNAPSHOT_INTERVAL = 10000  # ticks between spatial snapshots (positions + bonds)
GENOME_GC_INTERVAL = 1000     # ticks between genome garbage collection
BURST_SNAPSHOT_INTERVAL = 50000   # ticks between burst capture windows
BURST_SNAPSHOT_LENGTH = 20        # consecutive frames per burst
GENOME_WEIGHT_SNAPSHOT_INTERVAL = 50000  # ticks between genome weight snapshots
