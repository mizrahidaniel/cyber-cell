# CyberCell: Evolutionary Intelligence Simulation — Project Brief for Claude Code

## How to Use This Document

This is the master project brief for CyberCell, an artificial life simulation designed to evolve intelligence from the bottom up. It contains everything you need to plan, architect, and implement this system.

Read this entire document before writing any code. The design decisions are interdependent — the chemistry affects the energy model, which affects cell viability, which affects whether evolution produces anything interesting. Understand the whole system first.

**Document version:** v2.0 — Updated with research-informed architectural changes. The prototype (Steps 1-5) is largely complete. This revision reframes the project goals and lays out the next phase of work based on findings from the ALife research literature, particularly around computational substrates, open-ended evolution, and diversity maintenance.

---

## 1. Vision and Goal

We are building a simulated world where simple cyber-cells — minimal digital organisms — evolve through natural selection in a rich environment. The cells start with no intelligence. Through mutation, reproduction, death, and environmental pressure, we aim to see the emergence of increasingly complex behaviors: chemotaxis, predation, multicellularity, nervous system analogs, and eventually something recognizable as flexible, general-purpose problem-solving.

This is NOT a game, NOT a traditional genetic algorithm, and NOT a neural network training system. It is a bottom-up evolutionary simulation where intelligence (if it arises) is a byproduct of survival pressure, not an engineered outcome.

### Reframed Goal (Post-Research)

The primary goal is not "evolve intelligence" — it is **sustain open-ended evolution of increasing complexity.** The research is unambiguous: every known ALife system eventually plateaus (the "complexity barrier"). No artificial system has achieved transformational open-endedness — where the rules of evolution themselves evolve, producing qualitatively new kinds of entities. Our goal is to push as far past the complexity barrier as possible by combining insights from the ALife literature:

- **Quality-diversity algorithms** (MAP-Elites, minimal-criterion selection) to maintain genetic variation.
- **Ecological dynamics** (arms races, niche construction, co-evolution) as the primary driver.
- **Hybrid computational substrates** (gene regulatory networks + neural circuits) for richer evolvability.
- **LLM-guided environment design** to keep selection pressure calibrated to organism capability.

Intelligence, if it emerges, will be a side effect of sustained complexification — not a direct optimization target.

### Key Theoretical Framing

When bonded cell clusters communicate through bond signal channels, the multicellular organism is functionally a **dynamic graph neural network (GNN)**: cells are nodes with local computation, bonds are edges carrying messages, and the graph topology itself is evolved. This is a distinct computational architecture from a single monolithic network — it offers modularity (mutations to one cell don't break others), forced abstraction (narrow bond channels require information compression), and distributed sensing (50 cells sense 50 locations simultaneously). Whether this compositional architecture has genuine advantages over a single large network is an open empirical question that this project can help answer.

**Core philosophical principles that must guide every implementation decision:**

1. **No intelligence in the rules.** The simulation physics must be simple and dumb. If you find yourself encoding clever behavior into the substrate, stop. The rules are a substrate, not a solution.
2. **Compositionality over complexity.** Simple pieces combining into novel structures. The genome encoding must allow open-ended growth in program complexity.
3. **The environment does the teaching.** Selection pressure is the optimizer. The environment must be rich enough and varied enough that organisms face a continuous stream of novel problems.
4. **Death is non-negotiable.** Without death, there is no selection. Every cell must have finite energy, finite lifespan, and real consequences for failure.
5. **Emergent properties must be emergent.** Never hard-code high-level behaviors. Provide primitives rich enough that complex behaviors could arise through composition.
6. **Diversity is the fuel.** (NEW) Without maintained diversity, evolution converges to a single dominant strategy and stops innovating. Mechanisms that protect variation are not optional — they are as essential as selection itself.

---

## 2. Development Environment and Constraints

### Current Setup (Primary)
- **Machine:** Windows desktop with NVIDIA RTX 5080 (16GB GDDR7, 10,752 CUDA cores, 960 GB/s bandwidth)
- **Language:** Python 3.11+
- **Framework:** Taichi Lang (`pip install taichi`)
- **GPU Backend:** CUDA (`ti.init(arch=ti.cuda)`)
- **Visualization:** Taichi built-in GUI (`ti.GUI` or `ti.ui.Window`)
- **No network access required for the simulation itself**

### Secondary Setup
- **Machine:** MacBook with Apple Silicon (M-series chip)
- **GPU Backend:** Metal (`ti.init(arch=ti.metal)`)
- **Potential migration:** Rust + wgpu compute shaders for maximum performance (Phase 3)

### Critical Cross-Platform Rules
All code must work identically on both Mac (Metal) and Windows (CUDA) by changing only the `ti.init()` backend. Specifically:

- **Never use NumPy in any hot loop.** All simulation data must live in Taichi fields and be processed in Taichi kernels. NumPy is acceptable only for one-time initialization and offline analysis.
- **Never use Python-level loops over cells or grid positions.** Everything must be parallelized in `@ti.kernel` functions from the start, even if parallelism doesn't matter at small scale.
- **Separate simulation core from visualization completely.** The simulation must be runnable headless (no GUI) for long overnight evolution runs. Visualization is an optional overlay.
- **Use Taichi fields for all persistent state.** No Python lists, dicts, or objects in the simulation loop.

---

## 3. Current Prototype Status (COMPLETED)

The following systems are built and working. This section documents what exists so that future development builds on it correctly.

### 3.1 World — COMPLETE

- **2D grid**, 500 × 500, toroidal wrapping.
- **Three zones**: Light (left third, intensity 1.0), Dim (middle third, intensity 0.3), Dark (right third, intensity 0.0).
- **Day/night cycle:** Sinusoidal global light multiplier, period 1000 ticks.

### 3.2 Chemistry — COMPLETE

4 chemicals: Energy (E), Structure (S), Replication Material (R), Signal (G).
- S and R present as environmental deposits, clustered, slowly replenishing.
- G produced only by cells, diffuses and decays.
- E internal only, produced by photosynthesis.
- Diffusion is double-buffered on the 500×500 grid.

### 3.3 CyberCell — COMPLETE

18 sensory inputs (including prey_energy[16] and prey_membrane[17] for neighbor quality sensing), 10 action outputs. Cells occupy single grid positions. Full state tracking (position, energy, structure, replication material, signal, membrane integrity, age, genome_id, facing, alive flag, bond slots).

### 3.4 Genome — COMPLETE (Neural Network Version)

Fixed-size feedforward network: 18 → 32 → 32 → 10. 1,994 parameters per genome. Stored in genome table indexed by genome_id. Mutation operators: weight perturbation (0.03), weight reset (0.001), node knockout (0.0005).

### 3.5 Energy Model — COMPLETE

Photosynthesis, chemical consumption, predation. Full cost table for all actions. Energy conservation enforced. Death from starvation, membrane failure, and old age.

### 3.6 Cell Lifecycle — COMPLETE

Spawn, tick update (sense → think → act → metabolize → check death), division with mutation, death with chemical spillage.

### 3.7 Bonding — COMPLETE (With Known Issues)

Mutual bond formation, chemical sharing, movement constraints. **Known issue: degenerate chain formation.** Cells evolve always-bond genomes and form long immobile chains that are neither beneficial nor selected against. This is addressed in the next phase (Section 5).

### 3.8 Visualization — COMPLETE

Grid rendering, species coloring by genome hash, chemical heatmap overlays, light overlay, basic stats display.

---

## 4. Project Structure

```
cybercell/
├── CLAUDE.md                  ← This file (project brief)
├── README.md                  ← Public-facing project description
├── requirements.txt           ← Python dependencies
├── config.py                  ← All configurable parameters
├── main.py                    ← Entry point. Initializes simulation, runs main loop.
├── world/
│   ├── __init__.py
│   ├── grid.py                ← World grid, terrain zones, light model, day/night cycle.
│   ├── chemistry.py           ← Chemical fields, diffusion kernel, deposit placement.
│   └── archipelago.py         ← (NEW — to implement) Multi-world management, migration.
├── cell/
│   ├── __init__.py
│   ├── cell_state.py          ← Cell state fields (position, energy, genome_id, etc.)
│   ├── genome.py              ← Neural network genome (current, keep as baseline).
│   ├── genome_crn.py          ← (NEW — to implement) Chemical reaction network genome.
│   ├── sensing.py             ← Compute sensory inputs for all cells (parallel kernel).
│   ├── actions.py             ← Execute cell actions (move, eat, divide, attack, bond, etc.)
│   ├── bonding.py             ← Bonding logic: formation, sharing, group movement.
│   └── lifecycle.py           ← Birth, death, aging, division logic.
├── simulation/
│   ├── __init__.py
│   ├── engine.py              ← Main simulation tick.
│   ├── spawner.py             ← Initial cell seeding and periodic reseeding.
│   ├── checkpoint.py          ← Save/load full simulation state.
│   └── environment_api.py     ← (NEW — to implement) Clean API for environment modification.
├── visualization/
│   ├── __init__.py
│   └── renderer.py            ← Taichi GUI rendering, color mapping, overlays, stats display.
├── analysis/
│   ├── __init__.py
│   ├── metrics.py             ← Population stats, diversity measures, complexity tracking.
│   ├── logger.py              ← Periodic snapshots of simulation state for later analysis.
│   ├── study.py               ← Automated study runner and report generation.
│   ├── lineage_analysis.py    ← Genome lineage tree analysis.
│   ├── burst_analysis.py      ← Short burst frame-by-frame analysis.
│   ├── spatial_analysis.py    ← Spatial distribution analysis.
│   ├── bonding_analysis.py    ← Bond network analysis.
│   ├── oee_metrics.py         ← (NEW — to implement) Open-ended evolution metrics.
│   └── gnn_analysis.py        ← (NEW — to implement) Bond network information flow.
└── tests/
    ├── test_chemistry.py
    ├── test_energy.py
    ├── test_genome.py
    ├── test_lifecycle.py
    ├── test_predation.py
    ├── test_genome_crn.py      ← (NEW — to implement) Tests for CRN genome.
    └── test_bonding.py         ← (NEW — to implement) Tests for bond decay, signals.
```

---

## 5. Next Phase: Research-Informed Upgrades

These are the changes to implement now, in priority order. Each builds on the completed prototype. Implement them sequentially — each one should be validated before starting the next.

### Step 8: Fix Bonding Dynamics (Estimated: 1 day)

The chain formation problem must be solved before bonding can drive meaningful multicellularity.

**8a. Bond decay with active maintenance:**
- Each bond has a `strength` value (0.0 to 1.0). Starts at 0.5 on formation.
- Every tick, strength decreases by `BOND_DECAY_RATE` (default: 0.02).
- If BOTH cells have bond output above 0.5, strength increases by `BOND_REINFORCE_RATE` (default: 0.03).
- When strength reaches 0, the bond breaks.
- Effect: bonds that aren't actively maintained by both cells naturally lapse.

**8b. Lossy resource transfer:**
- Chemical sharing through bonds loses `1 - BOND_TRANSFER_EFFICIENCY` (default: 0.7, so 30% loss).
- Cell A loses 2 energy, Cell B gains only 1.4. The 0.6 is destroyed.
- Effect: long chains become net energy drains. Short purposeful pairs remain viable.

**8c. Expanded bond signal channel:**
- Increase from 1 signal float to `BOND_SIGNAL_WIDTH` (default: 4) floats per bond.
- Each cell has 4 bond signal outputs added to its action space.
- Each cell receives up to 4 × 4 = 16 bond signal inputs (4 values from each of 4 possible bonded neighbors).
- Sensory input vector grows from 18 to 34. Action output vector grows from 10 to 14.
- Effect: provides bandwidth for genuine inter-cell communication, enabling the GNN-like distributed computation.

**8d. Add neighbor quality sensing:** *(Already implemented)*
- prey_energy[16] and prey_membrane[17] already exist in `sensing.py`, providing energy and membrane levels of the cell ahead.
- No additional code changes needed.

**New config values:**
```python
# Bonding (updated)
BOND_DECAY_RATE = 0.02
BOND_REINFORCE_RATE = 0.03
BOND_TRANSFER_EFFICIENCY = 0.7
BOND_SIGNAL_WIDTH = 4
BOND_INITIAL_STRENGTH = 0.5
```

**Test:** Run for 200,000+ ticks. Degenerate chains should disappear. If bonding stops evolving entirely, reduce BOND_DECAY_RATE or increase BOND_TRANSFER_EFFICIENCY. Watch for: short clusters (2-5 cells) that persist and outsurvive solo cells.

### Step 9: Noisy Gradients and Patchy Resources (Estimated: 0.5 days)

Research finding: multicellular aggregates evolve because they perform chemotaxis more efficiently than single cells when gradients are noisy. This creates direct fitness advantage for bonded clusters that integrate sensory information.

**9a. Add Gaussian noise to chemical gradient sensing:**
- Each cell's gradient inputs get additive noise: `gradient + N(0, GRADIENT_NOISE_SIGMA)`.
- Default `GRADIENT_NOISE_SIGMA = 0.15` — enough that single cells frequently misjudge direction, but a cluster averaging multiple readings can triangulate accurately.

**9b. Make resources patchier:**
- Reduce the number of deposits but increase their concentration.
- Resources appear as tight clusters separated by barren zones.
- Resources shift position slowly over time (deposits deplete and new ones spawn elsewhere every ~10,000 ticks).
- Effect: cells must navigate to find food, not just sit on it. Navigation under noisy sensing rewards multicellular collectives.

**New config values:**
```python
# Sensing noise
GRADIENT_NOISE_SIGMA = 0.15

# Patchy resources
NUM_DEPOSITS_S = 80           # fewer but richer (was 200)
NUM_DEPOSITS_R = 40           # fewer but richer (was 100)
DEPOSIT_CONCENTRATION = 5.0   # higher concentration per deposit
DEPOSIT_SHIFT_INTERVAL = 10000  # ticks between deposit relocation
```

**Test:** Solo cells should visibly struggle to navigate to resources. Bonded clusters of 2-4 cells should find resources more reliably. If solo cells still navigate perfectly, increase GRADIENT_NOISE_SIGMA.

### Step 10: Archipelago Model for Diversity Maintenance (Estimated: 1-2 days)

This is the single most impactful change for sustaining open-ended evolution. Without it, the dominant strategy takes over and innovation stops.

**Architecture:**
- Instead of one 500×500 world, run `NUM_ISLANDS` (default: 4) independent worlds of 250×250 each.
- Each island has its own slightly different environment parameters (light intensity, resource distribution, deposit density) drawn from a configurable range around the defaults.
- Every `MIGRATION_INTERVAL` ticks (default: 5,000), copy a small number of cells (`MIGRATION_COUNT`, default: 10) from each island to a random other island.
- Migrants are selected proportionally to fitness (energy level) — successful strategies spread, but slowly.

**Why this works:**
- Each island evolves independently, exploring different regions of genome space.
- Occasional migration introduces novel genomes that can exploit niches the local population hasn't explored.
- Prevents single-strategy takeover because isolated populations can maintain alternative strategies.
- When migrants from one island encounter organisms from another, it creates the sudden competitive pressure (like biological invasive species) that drives rapid adaptation.

**Implementation notes:**
- Each island is a separate set of Taichi fields (grid, chemical fields, cell state).
- The main loop iterates over islands, updating each one per tick.
- Migration is a Python-level operation (runs infrequently, doesn't need GPU acceleration).
- Visualization can show all islands in a 2×2 grid or focus on one at a time.

**New config values:**
```python
# Archipelago
NUM_ISLANDS = 4
ISLAND_WIDTH = 250
ISLAND_HEIGHT = 250
MIGRATION_INTERVAL = 5000
MIGRATION_COUNT = 10
ISLAND_ENV_VARIANCE = 0.2    # each island's params vary ±20% from defaults
```

**Test:** Run 4 islands for 500,000 ticks. Each island should develop a visibly different dominant strategy (different colors/behaviors). After migration events, watch for: periods of rapid change as new genomes compete with established populations.

### Step 11: Enhanced Predation (Estimated: 0.5 days)

Predation is the engine of arms races, which are the engine of complexity. Currently, predation may not be energetically viable enough to evolve reliably.

**Changes:**
- **Direct absorption:** When a cell kills another cell via attack, the killer automatically absorbs 50% of the victim's internal chemicals directly. The remaining 50% spills to the environment as before.
- **Kill reward bonus:** Killer gains a flat `KILL_ENERGY_BONUS` (default: 5.0) energy on kill, representing the efficiency advantage of consuming an already-organized energy source.
- **Predator seeding (optional):** Every `PREDATOR_SEED_INTERVAL` ticks (default: 100,000), if no cells with recent attack behavior exist in a given island, seed 5 cells with hand-designed predator genomes (weights biased toward move-toward-cells and attack). These are not optimized — they just break the chicken-and-egg problem. Once prey evolve defenses, the seeded predators will be outcompeted by evolved ones.

**New config values:**
```python
# Predation (enhanced)
KILL_ABSORPTION_RATE = 0.5
KILL_ENERGY_BONUS = 5.0
PREDATOR_SEED_INTERVAL = 100000
PREDATOR_SEED_COUNT = 5
```

**Test:** Predators should evolve or persist on at least some islands. Look for oscillating predator-prey population dynamics (Lotka-Volterra cycles). If predators go extinct everywhere, increase KILL_ENERGY_BONUS or decrease ATTACK_COST.

### Step 12: Chemical Reaction Network Genome (Estimated: 3-5 days)

This is the most significant architectural change. It replaces the feedforward neural network with a biologically-inspired computational substrate that naturally supports memory, development, and differentiation.

**Architecture — A CRN genome consists of:**
- A set of `NUM_INTERNAL_CHEMICALS` (default: 8) internal chemical species, each with a concentration that persists between ticks.
- A set of `MAX_REACTIONS` (default: 16) reaction rules, each defined as:
  ```
  Reaction {
      input_a: int         — index of input chemical A (0 to NUM_INTERNAL_CHEMICALS-1, or sensory channel index)
      input_b: int         — index of input chemical B
      output: int          — index of output chemical
      threshold_a: float   — minimum concentration of A for reaction to fire
      threshold_b: float   — minimum concentration of B for reaction to fire
      rate: float          — production rate of output when conditions met
      decay_rate: float    — how fast the output chemical decays on its own
  }
  ```
- **Sensory coupling:** External sensory inputs (light, gradients, etc.) directly set the concentrations of specific internal chemicals each tick. E.g., internal chemical 0 = light intensity, chemical 1 = S gradient x, etc.
- **Action coupling:** Action outputs are triggered when specific internal chemicals exceed thresholds. E.g., if internal chemical 6 > 0.5, fire "move forward." If chemical 7 > 0.5, fire "divide."
- **Memory is free:** Chemical concentrations persist between ticks. A reaction that produces a slow-decaying chemical creates memory. A reaction chain where A triggers B triggers C creates sequential behavior. Oscillating reactions (A promotes B, B inhibits A) create rhythmic behavior.
- **Differentiation is natural:** Two cells with the same CRN genome but in different environments (different light, different neighbors) develop different internal chemical profiles, because their sensory inputs differ. Over time, their internal states diverge, driving different behaviors from the same genome.

**Genome size:** 16 reactions × 7 parameters = 112 floats. Much smaller than the neural network (1,994 floats), which means the mutation landscape is far more navigable.

**Mutation operators:**
- **Parameter perturbation:** Each parameter has probability 0.02 of perturbation by N(0, 0.1).
- **Reaction duplication:** Probability 0.005 of duplicating a reaction with slight noise (if space in MAX_REACTIONS).
- **Reaction deletion:** Probability 0.005 of disabling a reaction (setting rate to 0).
- **Rewiring:** Probability 0.01 of changing a reaction's input or output chemical index.

**Evaluation per tick:**
```
For each cell:
    1. Set sensory chemicals from environment.
    2. For each reaction: if input_a and input_b exceed thresholds, add rate * dt to output chemical.
    3. Apply decay to all internal chemicals.
    4. Read action chemicals and fire actions above threshold.
```

This is O(NUM_INTERNAL_CHEMICALS + MAX_REACTIONS) per cell per tick — much cheaper than the neural network evaluation.

**Keep the neural network genome as an alternative.** Both genome types should be selectable via config. This enables A/B comparison: run the same environment with CRN vs neural network genomes and compare evolutionary outcomes.

**New config values:**
```python
# CRN Genome
GENOME_TYPE = "crn"            # "crn" or "neural_network"
NUM_INTERNAL_CHEMICALS = 8
MAX_REACTIONS = 16
CRN_MUTATION_RATE_PERTURB = 0.02
CRN_MUTATION_SIGMA = 0.1
CRN_MUTATION_RATE_DUPLICATE = 0.005
CRN_MUTATION_RATE_DELETE = 0.005
CRN_MUTATION_RATE_REWIRE = 0.01
CRN_ACTION_THRESHOLD = 0.5
```

**Sensory-to-chemical mapping (default):**
```
Internal chemical 0 ← light_here
Internal chemical 1 ← S_gradient_x
Internal chemical 2 ← S_gradient_y
Internal chemical 3 ← R_gradient_x
Internal chemical 4 ← energy_level (normalized)
Internal chemical 5 ← cell_ahead
Internal chemical 6 ← bond_count (normalized)
Internal chemical 7 ← (available for purely internal dynamics)
```

**Chemical-to-action mapping (default):**
```
Internal chemical 0 > threshold → eat (photosynthesis context: light triggers eating)
Internal chemical 1 > threshold → move_forward
Internal chemical 2 > threshold → turn_left
Internal chemical 3 > threshold → turn_right
Internal chemical 4 > threshold → divide
Internal chemical 5 > threshold → attack
Internal chemical 6 > threshold → bond
Internal chemical 7 > threshold → emit_signal
```

Note: the same chemical can be both a sensory input and an action trigger, creating direct sensorimotor loops (e.g., high light → eat automatically). More complex behaviors emerge when reactions create indirect pathways.

**Test:** Run CRN and neural network versions in identical environments for 200,000 ticks. Compare:
- Time to evolve chemotaxis.
- Diversity of strategies (number of distinct behavioral phenotypes).
- Genome complexity over time (active reaction count for CRN, effective network connectivity for NN).
- Whether CRN cells exhibit memory-dependent behavior (different response to same stimulus based on recent history).

### Step 13: Environment Parameter API (Estimated: 0.5 days)

Expose all tunable environment parameters through a clean Python API that an external process can call. This is infrastructure for the eventual LLM-guided environment design.

**`environment_api.py` provides:**
```python
class EnvironmentAPI:
    def get_metrics(self) -> dict:
        """Return current simulation metrics: population, diversity, 
        avg energy, births/deaths rate, genome complexity stats, 
        OEE activity measures."""
    
    def set_parameter(self, name: str, value: float):
        """Modify any config parameter at runtime."""
    
    def add_deposit(self, x: int, y: int, chemical: str, amount: float):
        """Place a new chemical deposit."""
    
    def trigger_event(self, event_type: str, **kwargs):
        """Trigger environmental events: 'catastrophe' (kill % of cells),
        'resource_boom' (flood area with chemicals), 
        'barrier' (create impassable region), etc."""
    
    def get_population_snapshot(self) -> dict:
        """Return detailed population data for analysis."""
```

This does not connect to an LLM yet. It just ensures the interface exists so that when we add LLM integration later, no simulation code needs to change.

### Step 14: OEE Metrics (Estimated: 1 day)

Implement proper open-ended evolution measurement so we can objectively assess whether complexity is still increasing or has plateaued.

**Metrics to implement:**

- **Bedau's evolutionary activity:** Track the frequency of each genome over time. Activity = rate of change of genome frequencies. Bounded activity = plateau. Unbounded activity = still evolving.
- **MODES metrics (Dolson et al.):** Change (is the population still changing?), Novelty (are new genome types appearing?), Complexity (are genomes getting more complex?), Ecology (is ecosystem diversity maintained?).
- **Behavioral diversity:** Cluster organisms by behavioral phenotype (what actions they take in standardized test scenarios), not just genotype. Two different genomes that behave identically are not true diversity.
- **Information-theoretic measures:** Shannon entropy of genome distribution. Mutual information between sensory inputs and action outputs (are organisms actually using their senses, or acting randomly?).
- **Bond network analysis:** For multicellular organisms: entropy of bond signals (is communication carrying information?), graph topology metrics (clustering coefficient, diameter), correlation between bond signals and environmental events (do signals respond to stimuli?).

Log all metrics every 1,000 ticks. Plot rolling averages. Flag when metrics plateau for more than 50,000 ticks — this is the "evolution has stalled" signal.

---

## 6. Configuration Defaults

All defined in `config.py`. Every parameter tunable without changing any other code.

```python
# === World ===
GRID_WIDTH = 500
GRID_HEIGHT = 500
LIGHT_ZONE_END = 166
DIM_ZONE_END = 333
DAY_LENGTH = 1000
LIGHT_BRIGHT = 1.0
LIGHT_DIM = 0.3
LIGHT_DARK = 0.0

# === Archipelago (NEW — to implement) ===
ARCHIPELAGO_ENABLED = False    # enable multi-island mode
NUM_ISLANDS = 4
ISLAND_WIDTH = 250
ISLAND_HEIGHT = 250
MIGRATION_INTERVAL = 5000
MIGRATION_COUNT = 10
ISLAND_ENV_VARIANCE = 0.2

# === Chemistry ===
DIFFUSION_RATE_S = 0.01
DIFFUSION_RATE_R = 0.03        # tuned up from 0.005 for R deposit visibility
DIFFUSION_RATE_G = 0.3
E_DECAY_FLAT = 0.02            # flat internal energy decay per tick
DECAY_RATE_S = 0.001
DECAY_RATE_R = 0.001
DECAY_RATE_G = 0.05
DEPOSIT_REPLENISH_RATE = 0.012 # tuned up from 0.001
NUM_DEPOSITS_S = 200           # actual tuned value (was 80 in draft)
NUM_DEPOSITS_R = 200           # actual tuned value (was 40 in draft)
DEPOSIT_CLUSTER_RADIUS = 15
DEPOSIT_CLUSTER_AMOUNT = 5.0
R_LIGHT_ZONE_FRACTION = 0.15

# === Sensing ===
GRADIENT_NOISE_SIGMA = 0.15    # NEW — to implement in Step 9

# === Cells ===
MAX_CELLS = 50000
MAX_GENOMES = 50000
INITIAL_CELL_COUNT = 1000
MAX_CELL_AGE = 5000
INITIAL_ENERGY = 35.0          # tuned up from 25.0
INITIAL_STRUCTURE = 25.0
INITIAL_REPMAT = 10.0          # tuned up from 5.0
MEMBRANE_INITIAL = 100.0
ENERGY_ZERO_MEMBRANE_DAMAGE = 5.0
AGE_MEMBRANE_DECAY = 1.0

# === Energy Costs ===
BASAL_METABOLISM = 0.08        # tuned up from 0.05
MOVE_COST = 0.1                # tuned down from 0.3
TURN_COST = 0.02
EAT_COST = 0.02
SIGNAL_COST = 0.1
DIVIDE_COST = 20.0
DIVIDE_R_COST = 5.0
BOND_COST = 0.01               # tuned down from 0.05
ATTACK_COST = 0.3              # tuned down from 0.5
REPAIR_COST = 0.1
REPAIR_S_COST = 0.5
REPAIR_MEMBRANE_GAIN = 5.0
NETWORK_COST = 0.01

# === Energy Income ===
PHOTOSYNTHESIS_RATE = 0.45     # tuned down from 0.5
S_ENERGY_VALUE = 0.3           # tuned up from 0.1
R_ENERGY_VALUE = 0.5           # tuned up from 0.2
EAT_ABSORB_CAP = 2.0
PASSIVE_EAT_CAP = 0.05
ATTACK_MEMBRANE_DAMAGE = 8.0   # tuned down from 10.0

# === Predation (NEW — to implement in Step 11) ===
KILL_ABSORPTION_RATE = 0.5
KILL_ENERGY_BONUS = 2.0        # conservative start (CLAUDE.md draft said 5.0)
PREDATOR_SEED_INTERVAL = 100000
PREDATOR_SEED_COUNT = 5

# === Bonding ===
BOND_SHARE_RATE = 0.1
BOND_DECAY_RATE = 0.02         # NEW — Step 8a
BOND_REINFORCE_RATE = 0.03     # NEW — Step 8a
BOND_TRANSFER_LOSS = 0.3       # NEW — Step 8b
BOND_SIGNAL_CHANNELS = 4       # NEW — Step 8c
BOND_INITIAL_STRENGTH = 0.5    # NEW — Step 8a

# === Genome (Neural Network — current default) ===
GENOME_TYPE = "neural"         # "neural" or "crn"
NUM_INPUTS = 18                # 18 base (grows to 34 after Step 8c bond signals)
NETWORK_HIDDEN_SIZE = 32
NUM_OUTPUTS = 10               # 10 base (grows to 14 after Step 8c bond signals)
ATTACK_BIAS = -0.3
SEED_WEIGHT_SIGMA = 0.01
GRADIENT_SCALE_S = 3.0
GRADIENT_SCALE_R = 5.0
ACTION_THRESHOLD = 0.5

# === Mutation ===
MUTATION_RATE_PERTURB = 0.03   # tuned up from 0.01
MUTATION_SIGMA = 0.1
MUTATION_RATE_RESET = 0.001
MUTATION_RATE_KNOCKOUT = 0.0005

# === Genome (CRN — NEW, to implement in Step 12) ===
NUM_INTERNAL_CHEMICALS = 8
MAX_REACTIONS = 16
CRN_MUTATION_RATE_PERTURB = 0.02
CRN_MUTATION_SIGMA = 0.1
CRN_MUTATION_RATE_DUPLICATE = 0.005
CRN_MUTATION_RATE_DELETE = 0.005
CRN_MUTATION_RATE_REWIRE = 0.01
CRN_ACTION_THRESHOLD = 0.5

# === Reproduction ===
PARENT_RESOURCE_SHARE = 0.6
DAUGHTER_RESOURCE_SHARE = 0.4
```

---

## 7. Key Technical Notes

### Taichi-Specific Patterns

**Cell update kernel pattern:**
```python
@ti.kernel
def update_cells():
    for i in range(max_cells):
        if cell_alive[i]:
            # Read sensory inputs
            # Evaluate genome (CRN or neural network based on config)
            # Execute actions
            # Deduct metabolism
```

**CRN evaluation kernel pattern:**
```python
@ti.kernel
def evaluate_crn():
    for i in range(max_cells):
        if cell_alive[i]:
            gid = cell_genome_id[i]
            # Set sensory chemicals from environment
            for s in range(NUM_SENSORY_CHANNELS):
                internal_chemicals[i, sensory_map[s]] = sensory_input[i, s]
            # Evaluate reactions
            for r in range(MAX_REACTIONS):
                input_a = reaction_input_a[gid, r]
                input_b = reaction_input_b[gid, r]
                if (internal_chemicals[i, input_a] > reaction_threshold_a[gid, r] and
                    internal_chemicals[i, input_b] > reaction_threshold_b[gid, r]):
                    output = reaction_output[gid, r]
                    internal_chemicals[i, output] += reaction_rate[gid, r]
            # Apply decay
            for c in range(NUM_INTERNAL_CHEMICALS):
                internal_chemicals[i, c] *= (1.0 - chemical_decay[gid, c])
            # Read action outputs
            for a in range(NUM_ACTIONS):
                if internal_chemicals[i, action_map[a]] > CRN_ACTION_THRESHOLD:
                    action_active[i, a] = 1
```

**Diffusion kernel pattern (double-buffered):**
```python
@ti.kernel
def diffuse(src: ti.template(), dst: ti.template(), rate: float):
    for i, j in src:
        val = src[i, j]
        spread = val * rate * 0.25
        # Use atomic adds to neighbors in dst
        # Wrapping: (i + 1) % grid_size
```

### Avoiding Common Pitfalls

1. **Random number generation in Taichi:** Use `ti.random()` inside kernels. Do not use Python's `random` module inside `@ti.kernel` functions.
2. **Race conditions in cell movement:** Two cells cannot move to the same grid cell. Resolve by processing moves in a random order (shuffle cell indices each tick) and checking occupancy before committing.
3. **Division placement:** Scan 4 adjacent cells for an empty one. If none empty, division fails. Creates density-dependent reproduction pressure.
4. **Genome table growth:** Fixed-size Taichi field, max_genomes = 50,000. Implement garbage collection for genomes with no living references.
5. **Floating-point drift:** Clamp all chemical quantities to ≥ 0 after every subtraction.
6. **Bond strength tracking:** Bond strength is stored per-bond (not per-cell). When accessing bond data, always check that the reciprocal bond exists — if cell A's bond to cell B says strength 0.5 but cell B has no bond to A, there's a data inconsistency bug.

---

## 8. Future Roadmap (Do Not Implement Yet)

These are documented so architectural decisions don't accidentally prevent them.

### Phase 3: Expand CRN Substrate
- Allow reaction count to grow beyond MAX_REACTIONS through duplication mutations (variable-length genome).
- Add regulatory reactions: reactions that enable/disable other reactions based on chemical concentrations.
- Add inter-cell reaction coupling: bond channels carry chemical concentrations, not just signal floats, enabling true chemical communication between bonded cells.

### Phase 4: Richer Environment
- Expand terrain: elevation, biomes, water/land distinction.
- Add weather events, seasonal cycles, resource boom-bust dynamics.
- Add vibration/sound channel for long-range sensing.
- Add spatial vision: cells can sense light in a cone in their facing direction, not just at their position. Multicellular organisms with multiple sensors gain superior spatial awareness.

### Phase 5: LLM-Guided Environment Design
- Connect the environment_api to an LLM that monitors OEE metrics.
- LLM adjusts environment parameters to keep evolution in the productive zone.
- LLM generates novel environmental challenges calibrated to organism capability.
- Based on OMNI/OMNI-EPIC approach: foundation model as "model of interestingness."
- **Critical constraint:** The LLM shapes the environment only. Never modifies organisms.

### Phase 6: Scale to RTX 5080 / Cloud
- Port to CUDA. Increase world size. Target 1M+ cells.
- Run the CRN vs neural network comparison at scale.
- Run the monolithic-vs-compositional comparison: flatten best evolved multicellular organism into a single large network and compare performance.

### Phase 7: Academic Contribution
- Formalize the GNN interpretation of evolved multicellular organisms.
- Publish the CRN-vs-NN comparison results.
- Contribute to the open question: does compositional, modular, distributed intelligence have properties that monolithic intelligence lacks?

---

## 9. Success Criteria (Updated)

### Stage 1 — Ecosystem Viability (ACHIEVED):
- Stable autotroph populations. Chemical cycling. Day/night behavioral shifts.

### Stage 2 — Behavioral Evolution (ACHIEVED):
- Chemotaxis. Multiple coexisting strategies. Stable carrying capacity.

### Stage 3 — Ecological Complexity (IN PROGRESS):
- Reliable predator-prey dynamics with oscillating populations.
- Defensive behaviors (fleeing, clustering).
- Arms race dynamics visible in genome complexity over time.

### Stage 4 — Functional Multicellularity (NEXT TARGET):
- Bonded clusters of 2-5 cells that persist and outsurvive solo cells.
- Cells within clusters show context-dependent behavior (edge vs. interior).
- Bond signals carry measurable information (non-zero entropy, correlated with environmental events).

### Stage 5 — Sustained Open-Ended Evolution:
- OEE metrics show unbounded activity for 1M+ ticks.
- Archipelago maintains distinct strategies across islands.
- Migration events trigger observable adaptation cascades.
- Novel behavioral phenotypes continue appearing without stalling.

### Stage 6 — Distributed Computation:
- Multicellular organisms demonstrate behaviors impossible for single cells (e.g., reliable navigation in noisy gradients, coordinated predator response).
- Bond network topology shows non-trivial structure (not just chains or blobs).
- Information flow analysis shows genuine signal processing through bond channels.

---

## 10. What to Do If Evolution Stalls

In order of probability:

1. **Energy balance is wrong.** Adjust photosynthesis_rate and basal_metabolism.
2. **Mutation rate is wrong.** Try varying from 0.001 to 0.1.
3. **Environment is too uniform.** Make resources patchier, increase GRADIENT_NOISE_SIGMA.
4. **No diversity maintenance.** Enable archipelago if not already. Increase ISLAND_ENV_VARIANCE.
5. **Division too easy or hard.** Adjust until division happens roughly every 500-1000 ticks for successful cells.
6. **No competitive pressure.** Ensure resources are scarce. Seed predators if none evolve.
7. **Computational substrate too limited.** If using neural network, switch to CRN. If CRN with 16 reactions isn't enough, increase MAX_REACTIONS to 32.
8. **Bond channels too narrow.** Increase BOND_SIGNAL_WIDTH from 4 to 8 if multicellular communication seems bandwidth-limited.
9. **Check OEE metrics.** If change metric is positive but complexity metric is flat, evolution is churning but not innovating. The environment likely needs a qualitative change (new resource type, new zone, catastrophic event).

---

## 11. Testing Philosophy

Every subsystem must be testable in isolation:

- **Conservation laws:** Total energy trackable. No energy appears from nowhere or vanishes.
- **Determinism:** Same random seed → identical results.
- **Boundary conditions:** Toroidal wrapping correct for cells, chemicals, and bonds.
- **Edge cases:** Genome table full, division with no empty neighbors, energy going negative.
- **CRN-specific:** Verify reactions fire correctly, chemical decay works, sensory coupling is accurate, action thresholds trigger at the right values.
- **Bond-specific:** Verify bond decay, lossy transfer math, signal channel data flow, reciprocal bond consistency.
- **Archipelago-specific:** Verify migration copies cells correctly, island environments are actually distinct, metrics are tracked per-island.

Run the test suite after every significant change.

---

## 12. Key Research References

These papers informed the design decisions in this document. Consult them when making architectural choices.

- **Flow-Lenia** (Plantec et al., 2025): Mass conservation enables multi-species CA. Informed our chemical conservation approach.
- **Sensorimotor Lenia** (Hamon et al., 2025): ~130 parameters produce genuine agency from local rules. Informed our CRN compactness target.
- **ASAL** (Kumar et al., NeurIPS 2024): Foundation models automate ALife discovery. Informed our LLM environment design plans.
- **OMNI-EPIC** (Zhang et al., ICLR 2025): FM-generated environments achieve "Darwin Completeness." Informed our environment API design.
- **GReaNs** (Wróbel & Joachimczak): Single encoding produces both GRN and neural computation. Informed our hybrid substrate approach.
- **Schramm et al.**: Single GRN controls both morphogenesis and locomotion. Key motivation for CRN genome.
- **Dolson et al. (MODES)**: Formal OEE metrics. Directly implemented in Step 14.
- **ProtoEvo** (Cope, ALIFE 2023): Multicellular evolution with gene regulation. Closest existing project to ours.
- **TensorNEAT** (2024): GPU-accelerated NEAT. Reference for potential neural genome upgrade.
- **Chromaria** (Soros & Stanley, 2014): Niche construction sustains novelty. Informed our archipelago + migration design.

---

## Summary for Claude Code Planning

1. Steps 1-7 are complete. The prototype works.
2. Implement Steps 8-14 in order. Each builds on the previous.
3. Step 8 (bonding fix) and Step 9 (noisy gradients) are small changes with outsized impact — do these first.
4. Step 10 (archipelago) is the most important structural change for sustaining evolution.
5. Step 12 (CRN genome) is the most complex implementation task. Take it slowly, test thoroughly.
6. Keep the neural network genome as a comparison baseline. Never delete it.
7. Keep every module under 300 lines.
8. Commit after each step with descriptive messages.
9. When in doubt about a design decision, ask: "Does this make the fitness landscape smoother (good) or more rugged (bad)?" and "Does this maintain diversity or reduce it?"