# CyberCell: Evolutionary Intelligence Simulation — Project Brief for Claude Code

## How to Use This Document

This is the master project brief for CyberCell, an artificial life simulation designed to evolve intelligence from the bottom up. It contains everything you need to plan, architect, and implement this system.

Read this entire document before writing any code. The design decisions are interdependent — the chemistry affects the energy model, which affects cell viability, which affects whether evolution produces anything interesting. Understand the whole system first.

**Document version:** v3.0 — Steps 8-14 (research-informed upgrades) are implemented and validated on the `feature/research-upgrades` branch. This revision documents the current state of all systems, including bond dynamics, gradient noise, archipelago, enhanced predation, CRN genome, environment API, and OEE metrics.

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

## 3. Current Implementation Status

All systems below are built, working, and validated (200k-tick runs on CUDA).

### 3.1 World — COMPLETE

- **2D grid**, 500 × 500, toroidal wrapping.
- **Three zones**: Light (left third, intensity 1.0), Dim (middle third, intensity 0.3), Dark (right third, intensity 0.0).
- **Day/night cycle:** Sinusoidal global light multiplier, period 1000 ticks.
- **Patchy resources** (Step 9b): Tight deposit clusters (radius 10, amount 10.0). 20% of deposits relocate every 25,000 ticks, forcing cells to navigate.
- **Archipelago** (Step 10): Soft-wall quadrant partitioning — 4 semi-isolated 250×250 islands with ±20% parameter variance. Reduced diffusion at boundaries. Periodic fitness-proportional migration (10 cells every 5,000 ticks).

### 3.2 Chemistry — COMPLETE

4 chemicals: Energy (E), Structure (S), Replication Material (R), Signal (G).
- S and R present as environmental deposits, clustered, slowly replenishing.
- G produced only by cells, diffuses and decays.
- E internal only, produced by photosynthesis.
- Diffusion is double-buffered on the 500×500 grid.
- **Gradient noise** (Step 9a): All 6 gradient sensing channels (S, R, G × x,y) receive additive Gaussian noise (σ=0.15). Single cells misjudge direction; clusters can average readings.

### 3.3 CyberCell — COMPLETE

34 sensory inputs, 14 action outputs (expanded from 18/10 in Step 8c). Cells occupy single grid positions. Full state tracking (position, energy, structure, replication material, signal, membrane integrity, age, genome_id, facing, alive flag, bond slots, bond strength, bond signals, last_attacker).

**Sensory inputs:**
- [0..17]: Original 18 channels (light, internal state, gradients, neighbors, prey_energy, prey_membrane)
- [18..33]: Bond signal channels (4 bonds × 4 signal channels each) — Step 8c

**Action outputs:**
- [0..9]: Original 10 actions (move, turn_l, turn_r, eat, signal, divide, bond, unbond, attack, repair)
- [10..13]: Bond signal emission (4 channels) — Step 8c

### 3.4 Genome — COMPLETE (Two Genome Types)

**Neural Network (default, `GENOME_TYPE = "neural"`):**
Fixed-size feedforward network: 34 → 32 → 32 → 14. 2,638 parameters per genome. Mutation operators: weight perturbation (0.03), weight reset (0.001), node knockout (0.0005). Output biases tuned for viable starting phenotype (divide=+0.5, attack=-0.3).

**Chemical Reaction Network (Step 12, `GENOME_TYPE = "crn"`):**
8 internal chemicals with persistent concentrations, 16 reaction rules. 112 parameters per genome. Sensory-to-chemical and chemical-to-action mappings create direct sensorimotor loops (light→eat, energy→divide, structure→move, cell_ahead→bond, age→attack). Mutation: perturbation (0.02), rewiring (0.01), duplication/deletion (0.005). CRN_ACTION_THRESHOLD = 0.25, sensory blend ratio 0.3/0.7 (memory/input).

**200k-tick comparison results:**
| Metric | Neural | CRN |
|--------|--------|-----|
| Mean population | 856 | 87 |
| Movement fraction | 38.6% | 1.5% |
| Attack fraction | 0.6% | 12.8% |
| Bond fraction | 9.0% | 21.1% |
| Shannon entropy | 9.69 | 4.94 |
| Evolutionary activity | 0.75 | 2.16 |

**Known CRN issues:** Low carrying capacity (~100 cells vs ~800 for neural). High attack rate from age-gated sensorimotor loop. Low movement limits chemotaxis evolution. Population requires respawn safety net (100 cells when below 50). CRN needs further tuning — see Section 10 item 7.

### 3.5 Energy Model — COMPLETE

Photosynthesis, chemical consumption, predation with kill rewards. Full cost table for all actions. Energy conservation enforced. Death from starvation, membrane failure, and old age.

**Enhanced predation** (Step 11): When a cell kills another via attack, the killer absorbs 50% of victim's chemicals + 2.0 energy bonus. Remaining 50% spills to environment. `cell_last_attacker` field tracks who killed whom.

### 3.6 Cell Lifecycle — COMPLETE

Spawn, tick update (sense → think → act → metabolize → check death), division with mutation, death with chemical spillage. Emergency respawn: 100 fresh cells seeded when population drops below 50 (rate-limited to once per 5,000 ticks).

### 3.7 Bonding — COMPLETE (Fixed)

Mutual bond formation, chemical sharing with lossy transfer, coordinated group movement, bond signal relay.

**Bond strength decay** (Step 8a): Bonds start at strength 0.5. Decay by 0.02/tick without reinforcement. Reinforced by 0.03/tick when BOTH cells fire bond output. Auto-break below 0.05 strength. Sharing rate scaled by strength.

**Lossy transfer** (Step 8b): 30% of shared resources destroyed in transit. Long chains become net energy drains; short purposeful pairs remain viable.

**Bond signal channels** (Step 8c): 4 signal floats per bond direction. Cells emit signals via action outputs [10..13], receive partner signals via sensory inputs [18..33]. Enables GNN-like distributed computation. Neural genome: 88.7% of bonded cells have nonzero signals at 200k ticks.

**Validation results:** Degenerate chains eliminated. Bond strength distribution is bimodal (0.5 initial, 1.0 reinforced). Average cluster size 2-5 cells with rare outliers.

### 3.8 Visualization — COMPLETE

Grid rendering, species coloring by genome hash, chemical heatmap overlays, light overlay, basic stats display.

### 3.9 Environment API — COMPLETE (Step 13)

`simulation/env_api.py`: Clean Python API for runtime environment modification. `get_metrics()`, `set_parameter()`, `get_parameter()`, `add_deposit()`, `trigger_event()`, `get_population_snapshot()`. Infrastructure for future LLM-guided environment design.

### 3.10 OEE Metrics — COMPLETE (Step 14)

`analysis/oee_metrics.py`: Comprehensive open-ended evolution measurement. Bedau evolutionary activity, MODES metrics (change, novelty, complexity, ecology), Shannon entropy, mutual information (sense-action coupling), bond density. Plateau detection flags when metrics stall for 50+ snapshots. Logged to `oee_metrics.jsonl` every 1,000 ticks.

### 3.11 Validation & Analysis Tools — COMPLETE

- `validate.py`: 11-check automated validation harness. Tests population stability, bond dynamics, gradient noise, archipelago, predation, genome diversity, energy balance, cluster analysis. Generates diagnostic plots.
- `analysis/compare_runs.py`: Side-by-side Neural vs CRN comparison. 6-panel dynamics plot + 6-panel OEE plot + markdown report.

---

## 4. Project Structure

```
cybercell/
├── CLAUDE.md                  ← This file (project brief)
├── README.md                  ← Public-facing project description
├── requirements.txt           ← Python dependencies (taichi, numpy)
├── config.py                  ← All configurable parameters
├── main.py                    ← Entry point. Initializes simulation, runs main loop.
├── validate.py                ← Automated validation harness (11 checks, plots)
├── world/
│   ├── __init__.py
│   ├── grid.py                ← World grid, terrain zones, light model, day/night cycle.
│   ├── chemistry.py           ← Chemical fields, diffusion, deposits, relocation.
│   └── archipelago.py         ← Soft-wall quadrant islands, migration, param variance.
├── cell/
│   ├── __init__.py
│   ├── cell_state.py          ← Cell state fields (pos, energy, bonds, signals, etc.)
│   ├── genome.py              ← Neural network genome (baseline). 34→32→32→14.
│   ├── crn_genome.py          ← Chemical reaction network genome. 8 chemicals, 16 reactions.
│   ├── sensing.py             ← 34 sensory inputs (18 base + 16 bond signals) with noise.
│   ├── actions.py             ← 14 action outputs (10 base + 4 bond signal emission).
│   ├── bonding.py             ← Bond formation, strength decay, lossy sharing, signal relay.
│   └── lifecycle.py           ← Photosynthesis, metabolism, death, kill rewards.
├── simulation/
│   ├── __init__.py
│   ├── engine.py              ← Main simulation tick (dispatches neural or CRN).
│   ├── spawner.py             ← Initial seeding and emergency respawn.
│   ├── checkpoint.py          ← Save/load full simulation state.
│   └── env_api.py             ← Runtime environment modification API.
├── visualization/
│   ├── __init__.py
│   └── renderer.py            ← Taichi GUI rendering, color mapping, overlays, stats.
├── analysis/
│   ├── __init__.py
│   ├── metrics.py             ← Population stats, diversity, movement, predation.
│   ├── logger.py              ← Periodic snapshots + OEE metrics to disk.
│   ├── oee_metrics.py         ← Open-ended evolution metrics (Bedau, MODES, MI).
│   ├── compare_runs.py        ← Side-by-side Neural vs CRN comparison plots.
│   ├── study.py               ← Automated study runner and report generation.
│   ├── lineage_analysis.py    ← Genome lineage tree analysis.
│   ├── burst_analysis.py      ← Short burst frame-by-frame analysis.
│   ├── spatial_analysis.py    ← Spatial distribution analysis.
│   ├── bonding_analysis.py    ← Bond network analysis.
│   └── gnn_analysis.py        ← (FUTURE) Bond network information flow.
└── tests/
    ├── test_chemistry.py
    ├── test_energy.py
    ├── test_genome.py
    ├── test_lifecycle.py
    ├── test_predation.py
    ├── test_genome_crn.py      ← (FUTURE) Tests for CRN genome.
    └── test_bonding.py         ← (FUTURE) Tests for bond decay, signals.
```

---

## 5. Research-Informed Upgrades — IMPLEMENTED

Steps 8-14 are implemented and validated on the `feature/research-upgrades` branch. This section documents the design intent and implementation details for reference.

### Step 8: Fix Bonding Dynamics — COMPLETE

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

### Step 9: Noisy Gradients and Patchy Resources — COMPLETE

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

### Step 10: Archipelago Model for Diversity Maintenance — COMPLETE

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

**Actual implementation:** Soft-wall quadrant partitioning rather than separate field duplication. The existing 500×500 grid is divided into 4 250×250 quadrants. Boundaries have reduced diffusion, creating semi-isolation. This avoids massive kernel modifications while achieving the key diversity-maintenance benefit. Migration is Python-level, fitness-proportional cell teleportation between quadrants.

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

### Step 11: Enhanced Predation — COMPLETE

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

### Step 12: Chemical Reaction Network Genome — COMPLETE (needs tuning)

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

**Actual sensory-to-chemical mapping (tuned for viability):**
```
Chemical 0 ← light_here        → eat       (light triggers eating)
Chemical 1 ← energy_level      → divide    (energy triggers division)
Chemical 2 ← structure         → move      (fed cells move around)
Chemical 3 ← S_gradient_x      → turn_left (gradients steer turning)
Chemical 4 ← S_gradient_y      → turn_right
Chemical 5 ← cell_ahead        → bond      (neighbor → try to bond, harmless)
Chemical 6 ← bond_count        → signal    (bonded → emit signal)
Chemical 7 ← age               → attack    (age-gated: only old cells attack)
```

Key design choices from tuning:
- **Attack is age-gated** — mapping cell_ahead→attack caused mass extinction (all cells auto-attacked neighbors). Mapping age→attack means young cells don't attack, and predation must evolve through reaction chains that boost chemical 7 earlier.
- **Move is structure-gated** — gradient signals are too weak (~0.03) to cross threshold. Mapping structure→move means fed cells move; starving cells stay still.
- **CRN_ACTION_THRESHOLD = 0.25** (not 0.5) — with 0.3/0.7 sensory blend, most chemicals peak at 0.3-0.7. Threshold 0.5 prevented almost all actions from firing.
- **Sensory blend ratio 0.3/0.7** — 70% environment, 30% memory. Original 50/50 was too conservative; cells couldn't respond to current conditions.

Note: the same chemical is both a sensory input and an action trigger, creating direct sensorimotor loops. More complex behaviors emerge when reactions create indirect pathways.

**Test:** Run CRN and neural network versions in identical environments for 200,000 ticks. Compare:
- Time to evolve chemotaxis.
- Diversity of strategies (number of distinct behavioral phenotypes).
- Genome complexity over time (active reaction count for CRN, effective network connectivity for NN).
- Whether CRN cells exhibit memory-dependent behavior (different response to same stimulus based on recent history).

### Step 13: Environment Parameter API — COMPLETE

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

### Step 14: OEE Metrics — COMPLETE

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

All defined in `config.py`. Every parameter tunable without changing any other code. Values below match the actual config.py on the `feature/research-upgrades` branch.

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

# === Chemistry ===
DIFFUSION_RATE_S = 0.01
DIFFUSION_RATE_R = 0.03
DIFFUSION_RATE_G = 0.3
E_DECAY_FLAT = 0.02
DECAY_RATE_S = 0.001
DECAY_RATE_R = 0.001
DECAY_RATE_G = 0.05
DEPOSIT_REPLENISH_RATE = 0.012
NUM_DEPOSITS_S = 200
NUM_DEPOSITS_R = 200
DEPOSIT_CLUSTER_RADIUS = 10    # tighter clusters (was 15)
DEPOSIT_CLUSTER_AMOUNT = 10.0  # higher concentration (was 5.0)
DEPOSIT_RELOCATE_INTERVAL = 25000
DEPOSIT_RELOCATE_FRACTION = 0.2
R_LIGHT_ZONE_FRACTION = 0.15

# === Archipelago ===
ARCHIPELAGO_ENABLED = True
NUM_ISLANDS = 4
MIGRATION_INTERVAL = 5000
MIGRATION_COUNT = 10
ISLAND_ENV_VARIANCE = 0.2
ISLAND_BOUNDARY_DIFFUSION = 0.002

# === Sensing ===
GRADIENT_NOISE_SIGMA = 0.15

# === Cells ===
MAX_CELLS = 50000
INITIAL_CELL_COUNT = 1000
MIN_POPULATION = 50            # respawn threshold
RESPAWN_COUNT = 100
RESPAWN_INTERVAL = 5000
MAX_CELL_AGE = 5000
INITIAL_ENERGY = 35.0
INITIAL_STRUCTURE = 25.0
INITIAL_REPMAT = 10.0
MEMBRANE_INITIAL = 100.0
ENERGY_ZERO_MEMBRANE_DAMAGE = 5.0
AGE_MEMBRANE_DECAY = 1.0

# === Energy Costs ===
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

# === Energy Income ===
PHOTOSYNTHESIS_RATE = 0.45
S_ENERGY_VALUE = 0.3
R_ENERGY_VALUE = 0.5
EAT_ABSORB_CAP = 2.0
PASSIVE_EAT_CAP = 0.05
ATTACK_MEMBRANE_DAMAGE = 8.0
KILL_ABSORPTION_RATE = 0.5
KILL_ENERGY_BONUS = 2.0

# === Bonding ===
BOND_SHARE_RATE = 0.1
BOND_INITIAL_STRENGTH = 0.5
BOND_DECAY_RATE = 0.02
BOND_REINFORCE_RATE = 0.03
BOND_BREAK_THRESHOLD = 0.05
BOND_TRANSFER_LOSS = 0.3
BOND_SIGNAL_CHANNELS = 4

# === Genome (Neural Network — default) ===
GENOME_TYPE = "neural"         # "neural" or "crn"
MAX_GENOMES = 50000
NUM_INPUTS = 34                # 18 base + 16 bond signal inputs
NETWORK_HIDDEN_SIZE = 32
NUM_OUTPUTS = 14               # 10 base + 4 bond signal outputs
GENOME_SIZE = 2638             # 34*32+32+32*32+32+32*14+14
ATTACK_BIAS = -0.3
SEED_WEIGHT_SIGMA = 0.01
GRADIENT_SCALE_S = 3.0
GRADIENT_SCALE_R = 5.0
ACTION_THRESHOLD = 0.5

# === Mutation (Neural Network) ===
MUTATION_RATE_PERTURB = 0.03
MUTATION_SIGMA = 0.1
MUTATION_RATE_RESET = 0.001
MUTATION_RATE_KNOCKOUT = 0.0005

# === CRN Genome ===
NUM_INTERNAL_CHEMICALS = 8
MAX_REACTIONS = 16
CRN_PARAMS_PER_REACTION = 7
CRN_GENOME_SIZE = 112          # 16 * 7
CRN_MUTATION_RATE_PERTURB = 0.02
CRN_MUTATION_SIGMA = 0.1
CRN_MUTATION_RATE_DUPLICATE = 0.005
CRN_MUTATION_RATE_DELETE = 0.005
CRN_MUTATION_RATE_REWIRE = 0.01
CRN_ACTION_THRESHOLD = 0.25   # tuned down from 0.5

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
- Chemotaxis (38.6% movement fraction at 200k ticks). Multiple coexisting strategies. Stable carrying capacity (~800 cells neural).

### Stage 3 — Ecological Complexity (PARTIALLY ACHIEVED):
- Predation system functional (0.6% attack fraction neural, kill rewards working).
- Bond dynamics fixed: degenerate chains eliminated by strength decay.
- Bond signals active (88.7% nonzero for bonded cells).
- Arms race dynamics not yet clearly visible — requires longer runs.
- **Remaining:** Need stronger predator-prey oscillations and defensive clustering.

### Stage 4 — Functional Multicellularity (IN PROGRESS):
- Small bonded clusters (avg 2-5 cells) persist with active maintenance.
- Bond signal infrastructure in place (4 channels, relay working).
- **Remaining:** Context-dependent behavior within clusters not yet observed. Bond signal information content near zero (need longer evolution).

### Stage 5 — Sustained Open-Ended Evolution (INFRASTRUCTURE READY):
- OEE metrics implemented and logging (Bedau activity, MODES, MI, entropy).
- Archipelago maintains populations across quadrants (4/4 populated at 200k).
- Plateau detection implemented.
- **Remaining:** Need 1M+ tick runs to assess whether activity is truly unbounded.

### Stage 6 — Distributed Computation (FUTURE):
- Bond signal channels provide bandwidth for inter-cell communication.
- CRN genome provides memory substrate for temporal processing.
- **Remaining:** All Stage 6 criteria require longer evolutionary timescales.

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

1. Steps 1-14 are complete. All research-informed upgrades are implemented on `feature/research-upgrades`.
2. Neural network genome is the stable default. CRN genome works but needs further tuning (low carrying capacity, high attack rate).
3. **Priority CRN improvements:** Reduce attack auto-firing, improve movement/chemotaxis, expand action mapping beyond 8 chemicals to cover bond signals and repair.
4. **Next milestone:** Run 1M+ tick simulations to assess OEE metric trajectories and whether evolution is genuinely open-ended.
5. Keep the neural network genome as a comparison baseline. Never delete it.
6. Keep every module under 300 lines.
7. Run `python validate.py --ticks 30000` after significant changes to verify nothing is broken.
8. When in doubt about a design decision, ask: "Does this make the fitness landscape smoother (good) or more rugged (bad)?" and "Does this maintain diversity or reduce it?"
9. Use `python analysis/compare_runs.py --latest` to compare recent long runs.