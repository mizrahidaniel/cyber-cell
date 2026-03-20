# CyberCell: Evolutionary Intelligence Simulation — Project Brief for Claude Code

## How to Use This Document

This is the master project brief for CyberCell, an artificial life simulation designed to evolve intelligence from the bottom up. It contains everything you need to plan, architect, and implement this system.

Read this entire document before writing any code. The design decisions are interdependent — the chemistry affects the energy model, which affects cell viability, which affects whether evolution produces anything interesting. Understand the whole system first.

---

## 1. Vision and Goal

We are building a simulated world where simple cyber-cells — minimal digital organisms — evolve through natural selection in a rich environment. The cells start with no intelligence. Through mutation, reproduction, death, and environmental pressure, we aim to see the emergence of increasingly complex behaviors: chemotaxis, predation, multicellularity, nervous system analogs, and eventually something recognizable as flexible, general-purpose problem-solving.

This is NOT a game, NOT a traditional genetic algorithm, and NOT a neural network training system. It is a bottom-up evolutionary simulation where intelligence (if it arises) is a byproduct of survival pressure, not an engineered outcome.

**Core philosophical principles that must guide every implementation decision:**

1. **No intelligence in the rules.** The simulation physics must be simple and dumb. If you find yourself encoding clever behavior into the substrate, stop. The rules are a substrate, not a solution.
2. **Compositionality over complexity.** Simple pieces combining into novel structures. The genome encoding must allow open-ended growth in program complexity.
3. **The environment does the teaching.** Selection pressure is the optimizer. The environment must be rich enough and varied enough that organisms face a continuous stream of novel problems.
4. **Death is non-negotiable.** Without death, there is no selection. Every cell must have finite energy, finite lifespan, and real consequences for failure.
5. **Emergent properties must be emergent.** Never hard-code high-level behaviors. Provide primitives rich enough that complex behaviors could arise through composition.

---

## 2. Development Environment and Constraints

### Current Setup
- **Machine:** Windows desktop with NVIDIA RTX 5080 (16GB GDDR7, 10,752 CUDA cores, 960 GB/s bandwidth)
- **Language:** Python 3.12+
- **Framework:** Taichi Lang (`pip install taichi`)
- **GPU Backend:** CUDA (`ti.init(arch=ti.cuda)`) — auto-selected via `--backend auto`
- **Visualization:** Taichi built-in GUI (`ti.GUI`)
- **Also tested on:** MacBook with Apple Silicon (Metal backend)
- **Potential migration:** Rust + wgpu compute shaders for maximum performance (future)

### Critical Cross-Platform Rules
All code must work identically on both Mac (Metal) and Windows (CUDA) by changing only the `ti.init()` backend. Specifically:

- **Never use NumPy in any hot loop.** All simulation data must live in Taichi fields and be processed in Taichi kernels. NumPy is acceptable only for one-time initialization and offline analysis.
- **Never use Python-level loops over cells or grid positions.** Everything must be parallelized in `@ti.kernel` functions from the start, even if parallelism doesn't matter at small scale.
- **Separate simulation core from visualization completely.** The simulation must be runnable headless (no GUI) for long overnight evolution runs. Visualization is an optional overlay.
- **Use Taichi fields for all persistent state.** No Python lists, dicts, or objects in the simulation loop.

---

## 3. The Simplified Prototype (Build This First)

We are starting with an aggressively simplified version. This is intentional. The goal is to validate the core evolutionary dynamics before adding complexity. Every feature listed below is the MINIMUM needed. Do not add anything not listed here until the prototype is working and producing interesting evolution.

### 3.1 World

- **2D grid**, not continuous space. Each grid cell either contains a CyberCell or is empty.
- **Default size:** 500 × 500. Configurable via a constant.
- **Toroidal wrapping:** left edge connects to right, top to bottom. No edge effects.
- **Three zones** (defined by x-coordinate ranges, configurable):
  - **Light zone** (left third): Strong illumination. Primary energy source.
  - **Dim zone** (middle third): Weak illumination. Moderate energy available.
  - **Dark zone** (right third): No light. Scattered chemical deposits provide alternative resources.
- **Day/night cycle:** A global light multiplier that oscillates sinusoidally. Configurable period (default: 1000 ticks). At night, the light zone produces no energy. This creates pressure for energy storage.

### 3.2 Chemistry — 4 Chemicals Only

| ID | Name | Role | Diffuses? | Decay Rate |
|----|------|------|-----------|------------|
| E  | Energy | Internal fuel. All actions cost energy. Cells die without it. | No (internal only) | 0.02/tick |
| S  | Structure | Needed to maintain membrane integrity. Consumed by growth and repair. | Yes (slow, 0.01) | 0.001/tick |
| R  | Replication Material | Required for cell division. Scarce. Drives competition. | Yes (slow, 0.03) | 0.001/tick |
| G  | Signal | Diffusible chemical for communication and chemotaxis. | Yes (fast, 0.3) | 0.05/tick |

**Environmental distribution:**
- S deposits are placed in dim + dark zones only (stepping stones for cells leaving the light zone).
- R deposits are mostly in dim + dark zones (85%), with a small fraction (15%, configurable via `R_LIGHT_ZONE_FRACTION`) in the light zone to allow initial reproduction before cells evolve chemotaxis.
- G is only produced by cells. It diffuses outward and decays. Used for trail-following, alarm signals, etc.
- E is never in the environment directly. It is produced by cells via photosynthesis (light + internal process → E) or by consuming other cells/chemicals.

**Diffusion model:**
- Chemicals diffuse on the 500×500 grid itself (no coarser grid needed at this scale).
- Each tick, each chemical at each grid cell spreads a fraction of its concentration to the 4 cardinal neighbors (von Neumann neighborhood).
- Diffusion is a simple `@ti.kernel` that reads from one field and writes to another (double-buffered).

### 3.3 The CyberCell

Each cell occupies exactly one grid position. A grid cell can hold at most one CyberCell.

**State per cell:**
```
position: (int, int)        — grid coordinates
energy: float               — internal E. Dies at 0.
structure: float             — internal S. Membrane damage below threshold.
replication_material: float  — internal R. Needed for division.
signal_internal: float       — internal G store.
membrane_integrity: float    — 0-100. Below 0 = death.
age: int                     — ticks since birth.
genome_id: int               — index into genome table.
facing: int                  — 0-3 (up/right/down/left)
is_alive: bool               — whether this cell slot is active.
is_bonded_to: int[4]         — indices of bonded neighbors (or -1).
```

**Sensory inputs (fed to the neural network each tick):**
```
[0]  light_here           — light intensity at current position (0-1)
[1]  energy_level         — own energy (normalized 0-1)
[2]  structure_level      — own structure (normalized 0-1)
[3]  repmat_level         — own replication material (normalized 0-1)
[4]  membrane_integrity   — own membrane (normalized 0-1)
[5]  S_gradient_x         — chemical gradient of environmental S, x component
[6]  S_gradient_y         — chemical gradient of environmental S, y component
[7]  R_gradient_x         — gradient of environmental R, x component
[8]  R_gradient_y         — gradient of environmental R, y component
[9]  G_gradient_x         — gradient of signal chemical, x component
[10] G_gradient_y         — gradient of signal chemical, y component
[11] cell_ahead           — is there a cell in the direction I'm facing? (0 or 1)
[12] cell_left            — cell to my left? (0 or 1)
[13] cell_right           — cell to my right? (0 or 1)
[14] bond_count           — number of active bonds (0-4)
[15] age_normalized       — age / max_age
[16] prey_energy          — energy of cell ahead (normalized 0-1, 0 if no cell)
[17] prey_membrane        — membrane of cell ahead (normalized 0-1, 0 if no cell)
```
**Total: 18 sensory inputs.**

**Action outputs (from the neural network):**
```
[0]  move_forward         — move one step in facing direction (continuous 0-1, thresholded at 0.5)
[1]  turn_left            — rotate facing counter-clockwise
[2]  turn_right           — rotate facing clockwise
[3]  eat                  — absorb chemicals from environment at current position
[4]  emit_signal          — release G chemical into environment
[5]  divide               — attempt reproduction (requires sufficient E and R)
[6]  bond                 — attempt to bond with cell ahead
[7]  unbond               — release all bonds
[8]  attack               — damage the cell ahead (costs E, damages their membrane)
[9]  repair               — spend S to repair own membrane
```
**Total: 10 action outputs.**

### 3.4 The Genome — Fixed-Size Neural Network (For Now)

For the prototype, each genome is a **fixed-size feedforward neural network:**

- **Architecture:** 18 inputs → 32 hidden (tanh activation) → 32 hidden (tanh activation) → 10 outputs (sigmoid activation)
- **Total parameters:** 18×32 + 32 + 32×32 + 32 + 32×10 + 10 = 1,994 floats per genome.
- **Storage:** A Taichi field of shape `(max_genomes, 1994)` stores all unique genomes. Each cell references a genome by index.

**This is deliberately simpler than the full spec's variable-length regulatory graph.** The fixed-size network is sufficient for stages 1-3 (chemotaxis, predation, basic cooperation). We will upgrade to a more expressive genome representation once we've validated the evolutionary dynamics.

**Mutation operators (applied on GPU when a cell divides):**
- **Weight perturbation:** Each weight has a probability of 0.03 of being perturbed by Gaussian noise (σ = 0.1).
- **Weight reset:** Each weight has a probability of 0.001 of being reset to a random value in [-1, 1].
- **Node knockout:** Each hidden node has a probability of 0.0005 of having all its outgoing weights zeroed (effectively disabling it). This allows structural simplification.

**Genome sharing:** When a cell divides and the daughter's genome differs from the parent's (due to mutation), a new genome is allocated in the genome table. If the mutation produces no changes, the daughter shares the parent's genome_id. This saves memory and enables batched evaluation.

### 3.5 Energy Model

This is the most critical system to get right. If the energy balance is wrong, nothing else matters.

**Energy income:**
- **Photosynthesis:** A cell in a lit area gains `light_intensity * photosynthesis_rate` energy per tick. This is the primary energy source. Passive — does not require neural net activation.
- **Consuming chemicals:** The "eat" action absorbs environmental S and R (capped by `EAT_ABSORB_CAP` per tick). Passive eating also occurs at a lower rate (`PASSIVE_EAT_CAP`).
- **Predation:** The "attack" action damages another cell's membrane. If a cell dies, its internal chemicals spill into the environment and can be consumed by nearby cells. This is energetically viable but risky.

**Energy costs (see `config.py` for current tuned values):**
```
Existing (basal metabolism):     BASAL_METABOLISM per tick
Moving:                          MOVE_COST per move
Turning:                         TURN_COST per turn
Eating:                          EAT_COST per eat action
Emitting signal:                 SIGNAL_COST per emission
Dividing:                        DIVIDE_COST per division (plus requires R >= DIVIDE_R_COST)
Bonding maintenance:             BOND_COST per bond per tick
Attacking:                       ATTACK_COST per attack
Repairing:                       REPAIR_COST per repair action (consumes REPAIR_S_COST S)
Network evaluation:              NETWORK_COST per tick (genome complexity cost)
```

**Note:** These values have been iteratively tuned. The current values in `config.py` are the authoritative source. Key tuning insight: the economy must be tight enough that cells can't stockpile indefinitely (forces competition), but loose enough that random initial genomes survive the first crash (allows evolution to bootstrap).

**Key constraint: energy is conserved.** The sun (light) and chemical deposits are the only external inputs. Everything else is transformation. The ecosystem has a carrying capacity determined by the total energy input rate.

**Death conditions:**
- Energy reaches 0: cell takes 5 membrane damage per tick until energy is restored or it dies.
- Membrane integrity reaches 0: cell dies immediately. Internal chemicals spill into environment.
- Age exceeds max_age (default: 5000 ticks): cell begins losing 1 membrane per tick.

### 3.6 Cell Lifecycle

1. **Spawn:** At simulation start, seed cells are placed randomly in the light zone. Each gets a random genome with near-zero weights and biased output layer (see `SEED_WEIGHT_SIGMA`, `ATTACK_BIAS` in config).
2. **Each tick, for each living cell:**
   a. Compute sensory inputs from environment and internal state.
   b. Evaluate neural network: inputs → hidden → hidden → outputs.
   c. Execute the highest-activated actions (multiple actions can fire per tick if above threshold).
   d. Deduct basal metabolism.
   e. Apply membrane decay if energy is 0 or age exceeds max_age.
   f. Check death conditions.
3. **Division:** When a cell's "divide" output fires AND it has sufficient E (≥ 20) and R (≥ 5):
   - Find an empty adjacent grid cell. If none, division fails.
   - Create a daughter cell with 40% of parent's E and R. Parent keeps 60%.
   - Daughter gets a (possibly mutated) copy of parent's genome.
   - Daughter faces a random direction.
4. **Death:** Cell is marked dead. Internal chemicals are added to the environment at that grid position. The grid cell becomes empty.

### 3.7 Bonding (Multicellularity Foundation)

- A cell can bond with an adjacent cell if both have their "bond" output active in the same tick. Uses two-phase claim system (atomic_min) for conflict resolution.
- Bonded cells share chemicals with each bonded neighbor each tick (flow from high to low concentration). Costs `BOND_COST` energy per bond per tick.
- Either cell can break the bond unilaterally via the "unbond" output.
- **Bonded group movement:** Bonded cells move as a unit. The lowest-index cell in the group that fires "move" is the leader — its facing direction determines the group's movement. All bonded partners shift by the same (dx, dy). If any partner's target is blocked (occupied by a non-group-member), the entire move fails. Implementation: `cell/bonding.py::process_bonded_movement()`.
- Bond status is visible as a sensory input (bond_count).

**Current status (as of Stage 3 tuning):** Bonded clusters of 40-70+ cells are forming. Topologies are primarily pairs (57%), chains (28%), and meshes (8%). ~12-20% of cells are bonded. Clusters are mostly stationary — coordinated group movement is rare because facing alignment within clusters is low.

### 3.8 Visualization

**Real-time display requirements:**
- Show the full grid. Each cell is a colored pixel.
- Color encoding: empty = black, cell colored by species (hash of genome_id to hue), brightness = energy level.
- Overlay: chemical concentration heatmap (toggle between E environment deposits, S, R, and G). Use transparency or a separate layer.
- Overlay: light intensity (subtle gradient showing the three zones and day/night cycle).
- Stats panel (text overlay or separate): tick count, population, number of unique genomes, average energy, births/deaths per 100 ticks.

**Implementation:** Use `ti.GUI` for the prototype. 500×500 is small enough to render every pixel at 60fps.

---

## 4. Project Structure

```
cybercell/
├── CLAUDE.md                  ← This file (project brief)
├── README.md                  ← Public-facing project description
├── requirements.txt           ← Python dependencies
├── config.py                  ← All configurable parameters (single source of truth)
├── main.py                    ← Entry point. Auto backend selection, CLI args, main loop.
├── world/
│   ├── __init__.py
│   ├── grid.py                ← World grid, terrain zones, light model, day/night cycle.
│   └── chemistry.py           ← Chemical fields, diffusion kernel, deposit placement.
├── cell/
│   ├── __init__.py
│   ├── cell_state.py          ← Cell state fields (position, energy, genome_id, etc.)
│   ├── genome.py              ← Genome table, network evaluation kernel, GPU-side mutations.
│   ├── sensing.py             ← Compute sensory inputs for all cells (parallel kernel).
│   ├── actions.py             ← Execute cell actions (move, eat, divide, attack, etc.)
│   ├── bonding.py             ← Bond formation, unbonding, chemical sharing, group movement.
│   └── lifecycle.py           ← Photosynthesis, metabolism, death, chemical spillage.
├── simulation/
│   ├── __init__.py
│   ├── engine.py              ← Main simulation tick: environment → sense → think → act → resolve.
│   ├── spawner.py             ← Initial cell seeding.
│   └── checkpoint.py          ← Save/restore simulation state for backend switching.
├── visualization/
│   ├── __init__.py
│   └── renderer.py            ← Taichi GUI rendering, color mapping, overlays, stats display.
├── analysis/
│   ├── __init__.py
│   ├── metrics.py             ← Population stats, diversity, spatial + genome weight snapshots.
│   ├── logger.py              ← Periodic metric, spatial, burst, genome weight, lineage logging.
│   ├── study.py               ← Evolutionary dynamics analysis and phase detection.
│   ├── spatial_analysis.py    ← Spatial structure detection (clusters, lines, density).
│   ├── bonding_analysis.py    ← Bonding topology, coordination, persistence analysis.
│   ├── lineage_analysis.py    ← Phylogenetic trees, selective sweeps, weight evolution.
│   ├── burst_analysis.py      ← Frame-by-frame movement, division, and death detection.
│   └── output/{timestamp}/    ← Versioned analysis outputs (one folder per run).
├── runs/{timestamp}/          ← Simulation data:
│   ├── metrics.jsonl          ←   Population metrics every SNAPSHOT_INTERVAL ticks.
│   ├── lineage.jsonl          ←   Parent→child genome mutation events (every tick).
│   ├── spatial/               ←   Spatial snapshots (positions + bonds) every SPATIAL_SNAPSHOT_INTERVAL.
│   ├── burst/                 ←   Burst snapshots (20 consecutive frames) every BURST_SNAPSHOT_INTERVAL.
│   │   └── burst_{tick}/      ←     frame_00.npz .. frame_19.npz per burst window.
│   └── genomes/               ←   Genome weight snapshots every GENOME_WEIGHT_SNAPSHOT_INTERVAL.
└── tests/
    ├── test_chemistry.py      ← Verify diffusion conserves mass, decay works correctly.
    ├── test_energy.py         ← Verify energy conservation (total energy in system is bounded).
    ├── test_genome.py         ← Verify network evaluation, mutation produces valid genomes.
    ├── test_lifecycle.py      ← Verify birth/death/division mechanics.
    └── test_predation.py      ← Verify attack, bonding, death spillage, group movement.
```

---

## 5. Implementation Order

Steps 1-6 are complete. Step 7 is ongoing.

### Step 1: World and Chemistry ✅
- Implement the grid with three zones.
- Implement the light model (zone-based intensity + day/night sinusoidal cycle).
- Implement the 4 chemical fields with diffusion and decay.
- Place initial chemical deposits (clustered, using Perlin noise or simple random clusters).
- Write visualization: show the grid with chemical heatmaps and light overlay.
- **Test:** Run diffusion for 10,000 ticks. Verify chemicals spread and decay correctly. Verify light cycle works. Visually inspect that the world looks sensible.

### Step 2: Cells — Static ✅
- Implement cell state fields.
- Place 1,000 seed cells randomly in the light zone.
- Implement basal metabolism (energy drain per tick) and death from energy depletion.
- Implement photosynthesis (energy gain from light).
- Implement the "eat" action (absorb environmental chemicals).
- **Do NOT implement the neural network yet.** Hard-code cell behavior: all cells photosynthesize and eat. No movement, no reproduction.
- **Test:** Run for 10,000 ticks. Do cells survive in the light zone? Do they die in the dark zone? Is the energy balance sustainable — cells should reach an equilibrium, not grow forever or die out.

### Step 3: Genome and Neural Network ✅
- Implement the genome table and network evaluation kernel.
- Replace hard-coded behavior with network-driven behavior.
- Seed initial genomes with small random weights.
- Implement the full sensory input computation.
- Implement all 10 action outputs with their costs and effects.
- **Test:** Cells will initially behave randomly (random genomes). Most will die quickly. That's expected. Verify that the network evaluation runs correctly (outputs are in [0,1] range, inputs are normalized properly).

### Step 4: Reproduction and Mutation ✅
- Implement cell division with energy/material requirements.
- Implement genome mutation on division.
- Implement genome table management (allocate new genomes, track unique genomes).
- **Test:** This is the critical milestone. Run for 100,000+ ticks. Watch for:
  - Do any lineages survive long-term? (If not: energy balance is wrong, or mutation rate is too high.)
  - Does the population stabilize? (If it grows without bound: energy costs are too low.)
  - Do you see genome diversification? (Multiple distinct lineages with different behaviors.)
  - **Most importantly: do cells evolve to move toward food?** This is chemotaxis, the first non-trivial evolved behavior. If you see it, the core system is working.

### Step 5: Predation and Interaction ✅
- Implement the "attack" action (damages adjacent cell's membrane).
- Implement chemical spillage on death (dead cell's internals go to environment).
- Implement bonding (mutual bond, chemical sharing, movement constraints).
- **Test:** Run for 500,000+ ticks. Watch for:
  - Heterotrophs: cells that get energy from attacking others rather than from light.
  - Defensive behaviors: cells that flee from approaching cells.
  - Clustering: bonded groups that share resources.

### Step 6: Metrics and Analysis ✅
- Implement population tracking, genome diversity measures, behavioral classification.
- Implement periodic state snapshots (save to disk for offline analysis).
- Implement lineage tracker: `genome_parent_id` and `genome_birth_tick` fields track ancestry on GPU; mutation events logged to `lineage.jsonl` each tick.
- Implement burst snapshots: 20 consecutive frames captured every `BURST_SNAPSHOT_INTERVAL` ticks for frame-by-frame analysis of movement and division.
- Implement genome weight snapshots: active genome weights saved every `GENOME_WEIGHT_SNAPSHOT_INTERVAL` ticks for weight evolution analysis.
- Analysis scripts: `study.py` (evolutionary dynamics), `spatial_analysis.py` (clusters/density), `bonding_analysis.py` (bond topology), `lineage_analysis.py` (phylogenetic trees, sweeps, weight evolution), `burst_analysis.py` (frame-by-frame movement/division detection).
- Add visualization overlays for species (color by genome similarity), energy flow, births/deaths.

### Step 7: Parameter Tuning and Extended Runs (Ongoing)
- Run systematic parameter sweeps: vary mutation rates, energy costs, world size, chemical distributions.
- Identify the parameter regimes where evolution produces increasing complexity.
- Run extended simulations (millions of ticks) and analyze evolutionary trajectories.

---

## 6. Key Technical Notes

### Taichi-Specific Patterns

**Cell update kernel pattern:**
```python
@ti.kernel
def update_cells():
    for i in range(max_cells):
        if cell_alive[i]:
            # Read sensory inputs
            # Evaluate network
            # Execute actions
            # Deduct metabolism
```

**Diffusion kernel pattern (double-buffered):**
```python
@ti.kernel
def diffuse(src: ti.template(), dst: ti.template(), rate: float):
    for i, j in src:
        val = src[i, j]
        spread = val * rate * 0.25  # split among 4 neighbors
        # Use atomic adds to neighbors in dst
        # Wrapping: (i + 1) % grid_size
```

**Network evaluation:**
Each cell evaluates its neural network independently in a parallel kernel. Genome weights are read from the genome table by genome_id. Note: batching by genome was considered but every cell has a unique genome after a few generations (mutation on every division), so batching provides no benefit.

### Avoiding Common Pitfalls

1. **Random number generation in Taichi:** Use `ti.random()` inside kernels. Do not use Python's `random` module inside `@ti.kernel` functions.

2. **Race conditions in cell movement:** Two cells cannot move to the same grid cell. Resolved via two-phase intent-claim system: phase 1 declares intent and claims target with `atomic_min`, phase 2 only the winner moves. Bonded cells use a separate group movement system with leader election. See `cell/actions.py` and `cell/bonding.py`.

3. **Division placement:** When a cell divides, scan the 4 adjacent cells for an empty one. If none is empty, division fails and the cell retains its energy/materials. This creates density-dependent reproduction pressure.

4. **Genome table growth:** The genome table is a fixed-size Taichi field. Start with max_genomes = 50,000. Track active genomes with a counter. If the table fills, implement garbage collection (remove genomes with no living cells referencing them).

5. **Floating-point drift:** Chemical quantities should never go negative. Clamp to 0 after every subtraction. Energy conservation violations will compound over millions of ticks if not handled carefully.

---

## 7. Configuration Defaults

All parameters are defined in `config.py` as module-level constants. **`config.py` is the single source of truth** — always check it for current values, as parameters are iteratively tuned and the values below may be stale.

Key parameters that have been tuned from their original spec values (and why):

```python
# Economy (tightened to force competition — original was too generous)
BASAL_METABOLISM = 0.08       # was 0.05; prevents infinite energy stockpiling
PHOTOSYNTHESIS_RATE = 0.45    # was 0.5; tighter energy budget
MOVE_COST = 0.1               # was 0.3; cheaper movement encourages chemotaxis
INITIAL_ENERGY = 35.0          # was 25.0; more runway to survive initial crash

# Bonding (made affordable so evolution can experiment)
BOND_COST = 0.01              # was 0.05; old value was 100% of basal metabolism per bond

# Predation (deadlier attacks, arms race pressure)
ATTACK_COST = 0.3             # was 0.5; cheaper to attack
ATTACK_MEMBRANE_DAMAGE = 8.0  # was 10.0; tuned with other combat params

# Resources (scarcer R drives competition for reproduction)
NUM_DEPOSITS_R = 200          # was 100 in spec, tuned up/down during development
DEPOSIT_REPLENISH_RATE = 0.012 # was 0.001 in spec; tuned for viable reproduction
DIFFUSION_RATE_R = 0.03       # was 0.005; faster R diffusion for reachability
R_LIGHT_ZONE_FRACTION = 0.15  # NEW: 15% of R deposits in light zone

# Mutation (higher rate for faster adaptation)
MUTATION_RATE_PERTURB = 0.03  # was 0.01; faster exploration of weight space

# Chemical income
S_ENERGY_VALUE = 0.3          # was 0.1; eating is more rewarding
R_ENERGY_VALUE = 0.5          # was 0.2; incentivizes foraging for R
```

See `config.py` for the complete list including all unchanged defaults.

---

## 8. Future Roadmap (Do Not Implement Yet)

These are documented here so architectural decisions in the prototype don't accidentally prevent them.

### Phase 2: Richer Chemistry (After Stages 1-3 are working)
- Expand from 4 to 8-16 chemicals.
- Add reaction system: cells can evolve to catalyze novel chemical reactions.
- Add toxin/waste chemicals that create environmental hazards.
- Add temperature as a secondary environmental variable.

### Phase 3: Expressive Genomes (After multicellularity is observed)
- Replace fixed-size network with variable-length directed graph (regulatory network).
- Add regulatory nodes that can enable/disable other nodes based on context.
- Add gene duplication, block duplication, and crossover mutation operators.
- This is essential for cell differentiation in multicellular organisms.

### Phase 4: Richer Environment (After behavioral complexity is observed)
- Expand terrain: elevation, biomes, water/land distinction.
- Add weather events, seasonal cycles, resource boom-bust dynamics.
- Add vibration/sound channel for long-range sensing.
- Add photon emission for bioluminescence.

### Phase 5: LLM-Assisted Evolution (Experimental)
- Integrate an LLM as an adaptive environment designer.
- LLM monitors simulation metrics and adjusts environmental parameters to maintain evolutionary pressure.
- LLM analyzes organism behaviors and classifies strategies.
- LLM generates novel environmental challenges calibrated to organism capability.
- **Critical constraint:** The LLM shapes the environment only. It never modifies organisms or overrides natural selection.

### Phase 6: Scale (When moving to RTX 5080 / cloud GPUs)
- Port hot loops to custom CUDA kernels if Taichi performance is insufficient.
- Increase world size to 2000×2000 or larger.
- Target population of 1M+ cells.
- Implement spatial partitioning for neighbor queries.
- Eventually consider Rust + wgpu for maximum control.

---

## 9. Success Criteria

### Stage 1 Success ✅ ACHIEVED
- Stable autotroph populations form in the light zone.
- Chemical deposits get consumed and cycle through the ecosystem.
- Day/night cycle creates visible behavioral shifts (cells are more active during day).

### Stage 2 Success ✅ ACHIEVED
- Cells evolve chemotaxis — directed movement toward resources (48-58% of cells moving).
- Multiple distinct survival strategies coexist (sessile photosynthesizers vs. mobile foragers vs. boundary colonies).
- Population reaches a stable carrying capacity (~5K-11K depending on params).

### Stage 3 Success ✅ PARTIALLY ACHIEVED
- Heterotrophs emerge — cells that attack and consume other cells (~1-2% attacking).
- Attack deaths account for significant mortality (187K-239K attack deaths per run).
- Defensive clustering via bonding (12-20% of cells bonded, clusters up to 70+ cells).
- ⚠️ Not yet observed: clear predator-prey population oscillations or evolved flee behavior.

### Stage 4 Success — IN PROGRESS
- Bonded cell clusters persist (pairs, chains, trees, meshes up to 70 cells).
- ⚠️ Not yet observed: differentiated behavior within clusters (cells still run independent neural networks).
- **Key bottleneck:** Bonds share chemicals but not neural signals. Cells can't coordinate decisions. For differentiation, cells need a way to influence each other's neural inputs through bonds (see Phase 3 in roadmap).
- **Key observation:** Cluster topology is primarily parent-offspring chains. Every cell in a cluster has a unique genome (all mutants of a common ancestor). Facing coordination is weak (~42% of clusters show any alignment), so group movement is rare.

---

## 10. What to Do If Evolution Stalls

If the simulation runs for extended periods without producing interesting evolution, these are the most likely causes and fixes, in order of probability:

1. **Energy balance is wrong.** Either too generous (cells survive without adapting) or too harsh (everything dies before reproducing). Adjust photosynthesis_rate and basal_metabolism first.

2. **Mutation rate is wrong.** Too high: lineages dissolve into noise. Too low: no variation to select from. Try varying mutation_rate_perturb from 0.001 to 0.1.

3. **Environment is too uniform.** If food is everywhere, there's no pressure to move or specialize. Make resources patchier, add more chemical deposit clustering.

4. **Division is too easy or too hard.** If divide_cost is too low, the world fills with identical cells. If too high, nothing reproduces. Adjust until division happens roughly every 500-1000 ticks for successful cells.

5. **The network is too small.** If 32 hidden nodes can't represent the behaviors needed to exploit the environment, evolution will plateau. Try increasing to 64 or 128 hidden nodes (this increases genome size and evaluation cost).

6. **No competitive pressure.** If cells don't interact (no predation, no resource competition), there's no arms race to drive complexity. Ensure resources are scarce enough that cells compete.

---

## 11. Testing Philosophy

Every subsystem must be testable in isolation. Tests should verify:

- **Conservation laws:** Total energy in the system (cells + environment + light input - decay) should be trackable. If energy appears from nowhere or vanishes, there's a bug.
- **Determinism:** Given the same random seed, the simulation should produce identical results. This requires that all Taichi kernels process cells in a deterministic order (or that order-dependent operations are properly synchronized).
- **Boundary conditions:** Cells at grid edges wrap correctly. Chemicals at grid edges diffuse correctly across the wrap.
- **Edge cases:** What happens when the genome table fills up? When a cell tries to divide but all neighbors are occupied? When energy goes slightly negative due to floating point?

Run the test suite after every significant change. Subtle bugs in the energy model or diffusion will produce wrong evolutionary dynamics that are hard to diagnose after the fact.

---

## Summary for Claude Code Planning

When planning the implementation:

1. Start with Step 1 (World and Chemistry). Get it rendering on screen.
2. Move strictly through Steps 2-6 in order. Do not parallelize steps — each validates the previous one.
3. Keep every module under 300 lines. If a file grows beyond that, it needs to be split.
4. Write the config.py file first so all magic numbers have names.
5. Commit after each step with a descriptive message of what was added and what tests pass.
6. The first time you see a cell evolve to move toward food, celebrate. That's the proof of concept.