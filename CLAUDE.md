# CyberCell: Evolutionary Intelligence Simulation — Project Brief

**Document version:** v7.1 — Tuned waste (production 3x, threshold 0.3, toxicity 4x). Waste sensing on input[15] replacing age (CRN C7 reads waste automatically). Attenuation k reduced 0.03→0.02 to compensate. 200k-tick runs: CRN achieved waste-driven zone migration (26% bright, 48% dim, 26% dark at 200k — avg X=255) through passive dispersal, not movement. Neural evolved hyper-predation (43% attack) causing population collapse. **Decision: focus development on CRN genome.** Neural retained as comparison baseline only. Analysis upgraded: 8-panel study figure, environmental pressure comparison, per-zone waste metrics, preflight test runner. Test suite fixed (28/28 passing).

---

## 1. Vision and Goal

We are building a simulated world where simple cyber-cells evolve through natural selection. The cells start with no intelligence. Through mutation, reproduction, death, and environmental pressure, we aim to see the emergence of increasingly complex behaviors.

**The primary goal is to sustain open-ended evolution of increasing complexity.** Every known ALife system eventually plateaus. Our goal is to push past the complexity barrier by combining:

- **Quality-diversity algorithms** (archipelago, migration) to maintain genetic variation.
- **Ecological dynamics** (arms races, niche construction) as the primary driver.
- **Hybrid computational substrates** (CRN + neural network genomes) for richer evolvability.
- **LLM-guided environment design** (future) to keep selection pressure calibrated.

### Key Theoretical Framing

When bonded cell clusters communicate through bond signal channels, the multicellular organism is functionally a **dynamic graph neural network (GNN)**: cells are nodes with local computation, bonds are edges carrying messages, and the graph topology itself is evolved.

### Core Principles

1. **No intelligence in the rules.** Simulation physics must be simple. The rules are a substrate, not a solution.
2. **Compositionality over complexity.** Simple pieces combining into novel structures.
3. **The environment does the teaching.** Selection pressure is the optimizer.
4. **Death is non-negotiable.** Every cell must have finite energy, finite lifespan, and real consequences for failure.
5. **Emergent properties must be emergent.** Never hard-code high-level behaviors.
6. **Diversity is the fuel.** Without maintained diversity, evolution converges and stops innovating.

---

## 2. Development Environment

- **Primary:** Windows desktop, NVIDIA RTX 5080, Python 3.11+, Taichi Lang (CUDA backend)
- **Secondary:** MacBook with Apple Silicon (Metal backend)
- **Rule:** All code must work on both by changing only `ti.init()` backend.

### Critical Rules

- **Never use NumPy in hot loops.** All simulation data in Taichi fields, processed in `@ti.kernel` functions.
- **Never use Python-level loops over cells.** Everything parallelized from the start.
- **Separate simulation from visualization.** Must run headless for overnight evolution runs.

---

## 3. Current Implementation Status

All systems built, working, and validated on CUDA.

### World
500x500 toroidal grid. Three light zones (bright/dim/dark). Day/night cycle (period 1000 ticks). Patchy resource deposits (radius 10, 20% relocate every 25k ticks). Archipelago: 4 soft-wall quadrants with +/-30% parameter variance, uniform-random migration (1 cell every 200 ticks — continuous trickle matching Wright's Nm≈1 rule). **Beer-Lambert light attenuation**: density-dependent shading reduces photosynthesis in crowded areas. `local_density` computed in radius-2 neighborhood; `light *= exp(-k * density)` with k=0.02. Creates 18-33% light reduction at typical occupied densities (10-20 neighbors).

### Chemistry
5 chemicals: Energy (E), Structure (S), Replication Material (R), Signal (G), Waste (W). Double-buffered diffusion. Gradient noise (sigma=0.15) on all 6 gradient sensing channels. **Metabolic waste**: photosynthesis produces waste proportional to energy gained (rate 0.03). Waste diffuses (5%/tick), decays (0.002/tick, half-life ~350 ticks), and causes membrane damage above threshold (0.3) at rate 0.2/tick. **Cells sense waste directly** via sensory input[15] (replaced age sensing in v7.1). CRN C7 reads waste automatically through `_SENSORY_MAP[7]=15`.

### CyberCell
34 sensory inputs (18 base + 16 bond signals), 14 action outputs (10 base + 4 bond signal emission). Full state: position, energy, structure, repmat, signal, membrane, age, genome_id, facing, bonds, bond strength/signals, last_attacker.

### Genome — Two Types

**Chemical Reaction Network (`GENOME_TYPE = "crn"`, primary):**
16 internal chemicals in 3 zones, 16 reaction rules. 120 parameters (112 reaction + 4 action biases + 4 hidden decays). See Section 5 for architecture details. **CRN is the focus genome for all new development.**

**Neural Network (`GENOME_TYPE = "neural"`, comparison baseline):**
Feedforward 34->32->32->14. 2,638 parameters. Mutation: perturbation (0.03), reset (0.001), knockout (0.0005). Retained for comparison but not actively developed — evolves hyper-predation at long timescales that collapses population viability.

### Energy Model
Photosynthesis (density-attenuated), chemical consumption, predation with ecological kill rewards (35% absorption, no flat bonus). Full cost table in `config.py`. Death from starvation, membrane failure, old age.

### Bonding
Near-permanent bonds (decay 0.001/tick, reinforced at 0.03 when both cells fire bond). Auto-bond at division (incomplete cytokinesis, initial strength 0.1 — breaks in ~50 ticks unless reinforced). 30% lossy resource transfer. 4-channel bond signals per direction. Clusters up to 22 cells with mesh/chain/star topologies; 94% facing coordination in 3+ cell clusters.

### Visualization
Grid rendering, species coloring, chemical heatmaps, light overlay, stats display.

### Environment API (`simulation/env_api.py`)
Runtime parameter modification: `get_metrics()`, `set_parameter()`, `add_deposit()`, `trigger_event()`, `get_population_snapshot()`. Infrastructure for future LLM integration.

### OEE Metrics (`analysis/oee_metrics.py`)
Bedau evolutionary activity, MODES (change/novelty/complexity/ecology), Shannon entropy, mutual information, bond density. Plateau detection at 50+ stalled snapshots. Logged to `oee_metrics.jsonl` every 1,000 ticks.

### Analysis Infrastructure
See Section 8 for full tool descriptions and usage.

---

## 4. Project Structure

```
cybercell/
├── CLAUDE.md                  <- This file
├── config.py                  <- All tunable parameters (single source of truth)
├── main.py                    <- Entry point (backend auto-selection, headless mode)
├── validate.py                <- Backward-compat wrapper -> analysis/validate.py
├── world/
│   ├── grid.py                <- World grid, terrain zones, light, day/night cycle
│   ├── chemistry.py           <- Chemical fields, diffusion, deposits, relocation
│   └── archipelago.py         <- Soft-wall quadrants, migration, param variance
├── cell/
│   ├── cell_state.py          <- Cell state fields (pos, energy, bonds, signals, etc.)
│   ├── genome.py              <- Neural network genome. 34->32->32->14
│   ├── crn_genome.py          <- CRN genome. 16 chemicals (3 zones), 16 reactions
│   ├── sensing.py             <- 34 sensory inputs with noise
│   ├── actions.py             <- 14 action outputs
│   ├── bonding.py             <- Bond formation, decay, lossy sharing, signal relay
│   └── lifecycle.py           <- Photosynthesis, metabolism, death, kill rewards
├── simulation/
│   ├── engine.py              <- Main tick loop (dispatches neural or CRN)
│   ├── spawner.py             <- Initial seeding and emergency respawn
│   ├── checkpoint.py          <- Save/load full simulation state
│   └── env_api.py             <- Runtime environment modification API
├── visualization/
│   └── renderer.py            <- Taichi GUI rendering, overlays, stats
├── analysis/
│   ├── metrics.py             <- Population stats, diversity, CRN snapshot extraction
│   ├── logger.py              <- Periodic snapshots to disk (metrics, OEE, CRN, spatial)
│   ├── oee_metrics.py         <- Open-ended evolution metrics (Bedau, MODES, MI)
│   ├── crn_analysis.py        <- 9-panel CRN diagnostics + report
│   ├── compare_runs.py        <- Side-by-side Neural vs CRN comparison
│   ├── validate.py            <- Validation harness (13 checks for neural, 18 for CRN)
│   ├── run_all.py             <- Unified CLI: runs all applicable analyses on a run
│   ├── study.py               <- Evolutionary dynamics study + report
│   ├── lineage_analysis.py    <- Phylogenetic tree, selective sweeps, bias evolution
│   ├── spatial_analysis.py    <- Spatial distribution analysis
│   ├── bonding_analysis.py    <- Bond network analysis
│   └── burst_analysis.py      <- Frame-by-frame burst analysis
└── tests/
    ├── conftest.py            <- Session-scoped ti.init() (fixes pytest double-init)
    ├── test_chemistry.py
    ├── test_energy.py
    ├── test_genome.py
    ├── test_lifecycle.py
    └── test_predation.py
```

---

## 5. CRN Genome Architecture

The CRN is a biologically-inspired computational substrate that supports memory, development, and differentiation. It replaces the feedforward neural network with a chemical reaction network where concentrations persist between ticks.

### 3-Zone Chemical Space (16 chemicals)

| Zone | Indices | Role | Behavior |
|------|---------|------|----------|
| Sensory | 0-7 | Environment inputs | Blended with env each tick (50/50 memory/input) |
| Hidden | 8-11 | Memory/gates | Purely internal, evolved decay rates |
| Action | 12-15 | Action triggers | Reset to evolved biases each tick |

This separation forces reactions to build sensory->hidden->action circuits rather than direct sensorimotor loops (the CRN equivalent of a hidden layer). Negative thresholds enable NOT-gate logic.

### Sensory Mapping
```
C0 <- light_here       C4 <- S_gradient_y
C1 <- energy_level     C5 <- cell_ahead
C2 <- structure         C6 <- bond_count
C3 <- S_gradient_x     C7 <- waste (v7.1; was age)
```

### Action Mapping
```
C12 -> eat    C14 -> divide
C13 -> move   C15 -> attack
```
Turns are handled separately via facing-aware gradient steering. Hidden chemicals 8-9 drive signal/bond via auxiliary thresholds.

### Genome Layout (120 parameters)
- **Reactions** (0-111): 16 reactions x 7 params (input_a, input_b, output, threshold_a, threshold_b, rate, decay)
- **Action biases** (112-115): eat, move, divide, attack — reset to these values each tick
- **Hidden decays** (116-119): per-chemical decay rates for hidden zone

### Key Design Decisions
- **Attack was age-gated** (C7->C15 in v7.0): now waste-gated since C7 reads waste. Mapping cell_ahead->attack caused mass extinction.
- **Move is structure-gated** (C2->C13): bootstrap rate=0.2 gives ~9% initial movement with sigmoid.
- **Sigmoid action firing** (gain=30, center=0.5): replaces hard threshold. P(fire) = sigmoid(30*(chem - 0.5)). At bias 0.3: P≈0.3%. At bias+reaction 0.6: P≈95%. Provides evolutionary gradient below threshold without wasteful spontaneous actions.
- **Hidden basal production** (0.005/tick): steady-state floor at 0.25 (below 0.5 aux threshold). Keeps hidden chemicals positive and computationally useful. Reactions can push above 0.5 to activate signal/bond.
- **Hidden decay 0.02/tick**: half-life ~35 ticks. Long-term memory, prevents runaway.
- **Per-reaction decay restricted to sensory zone only** (0-7): action chemicals reset each tick, hidden has dedicated decay. Fixes bug where actions were decayed 3-5x per tick.
- **Zone-aware clamping**: sensory/hidden [0, 5], action [-1, 5]. Prevents negative concentrations in sensory/hidden zones.

---

## 6. Configuration

All parameters in `config.py` — the single source of truth. Key non-obvious values:

| Parameter | Value | Why |
|-----------|-------|-----|
| `GRADIENT_NOISE_SIGMA` | 0.15 | Forces clusters to be better navigators than solo cells |
| `BOND_DECAY_RATE` | 0.001 | Near-permanent bonds (>500 ticks); enables multicellularity |
| `BOND_TRANSFER_LOSS` | 0.3 | Makes long chains unprofitable; short pairs viable |
| `KILL_ABSORPTION_RATE` | 0.35 | ~3 kills to recoup prey lifetime; viable predation economics |
| `KILL_ENERGY_BONUS` | 0.0 | No flat bonus — predation profit from absorption only |
| `LIGHT_ATTENUATION_ENABLED` | True | Beer-Lambert density-dependent shading |
| `LIGHT_ATTENUATION_K` | 0.02 | Extinction coeff: 82% at density 10, 67% at density 20. Reduced from 0.03 in v7.1 to compensate for waste pressure. |
| `LIGHT_ATTENUATION_RADIUS` | 2 | 5x5-1=24 neighbor count for density |
| `ATTACK_BIAS` | -0.3 | Initial sigmoid output ~0.43 — predation must be evolved |
| `CRN_ACTION_GAIN` | 30.0 | Sigmoid steepness: <0.3% at bias, >95% when boosted by reactions |
| `CRN_ACTION_CENTER` | 0.5 | Sigmoid midpoint matches old threshold |
| `CRN_HIDDEN_BASAL` | 0.005 | Steady-state 0.25, below aux action threshold of 0.5 |
| `CRN_SENSORY_BLEND` | 0.5 | 50% memory / 50% environment per tick |
| `DEPOSIT_RELOCATE_INTERVAL` | 25000 | Forces navigation; static deposits allow sessile strategies |
| `MIGRATION_INTERVAL` | 200 | Continuous trickle (was 5000); Nm≈1.25 per generation |
| `MIGRATION_COUNT` | 1 | Per-island per event; uniform random selection (was fitness-proportional) |
| `ISLAND_ENV_VARIANCE` | 0.3 | ±30% parameter variance; more distinct islands |
| `WASTE_ENABLED` | True | Metabolic waste from photosynthesis |
| `WASTE_PRODUCTION_RATE` | 0.03 | Waste per unit energy gained. SS isolated ~0.26 (safe), cluster interior ~0.50 (above threshold). Tuned up from 0.01 in v7.1. |
| `WASTE_DECAY_RATE` | 0.002 | Half-life ~350 ticks |
| `WASTE_DIFFUSION_RATE` | 0.05 | 5% spreads per tick (same as signal) |
| `WASTE_TOXICITY_THRESHOLD` | 0.3 | Isolated cells (0.26) safe. Cluster edges (~0.40) take damage. Lowered from 0.5 in v7.1. |
| `WASTE_TOXICITY_RATE` | 0.2 | 4x increase from v7.1. Interior damage ~0.04 membrane/tick (~2500 ticks to die). |

---

## 7. Technical Notes

### Avoiding Common Pitfalls

1. **Random numbers in Taichi:** Use `ti.random()` inside kernels, never Python's `random`.
2. **Race conditions:** Two cells can't occupy the same cell. Check occupancy before committing moves.
3. **Division placement:** Scan 4 neighbors for empty. If none, division fails (density-dependent pressure).
4. **Genome table:** Fixed-size field (50k). Garbage collect genomes with no living references.
5. **Floating-point drift:** Clamp all chemical quantities >= 0 after subtraction.
6. **Bond consistency:** Always check reciprocal bonds exist — if A->B says 0.5 but B has no bond to A, it's a bug.

---

## 8. Analysis Infrastructure

### Data Flow

Simulation produces `runs/<timestamp>/` with:
- `metrics.jsonl` — population, energy, movement, attack, bond, light attenuation, zone breakdown, waste stats (every 1k ticks)
- `oee_metrics.jsonl` — Bedau activity, MODES, entropy, MI, bond density (every 1k ticks)
- `crn_metrics.jsonl` — CRN-only: zone activations, biases, decays, reactions, zone flow (every 1k ticks)
- `lineage.jsonl` — parent->child mutation events
- `spatial/` — cell positions + bonds (every 10k ticks)
- `genomes/` — full genome weights (every 50k ticks)
- `burst/` — rapid consecutive frames (every 50k ticks)

### Analysis Scripts

All per-run scripts output to `analysis/output/<run_name>/`. Run individually or via unified CLI:

```bash
python analysis/run_all.py                      # latest run (auto-detect)
python analysis/run_all.py runs/20260319_205444 # specific run
```

| Script | Output | Description |
|--------|--------|-------------|
| `study.py` | STUDY.md, evolution_report.png | Phase detection, rates, 8-panel dynamics (waste + zone panels added v7.1) |
| `crn_analysis.py` | CRN_ANALYSIS.md, crn_evolution.png | 9-panel CRN diagnostics (zone activation, bias drift, reaction graph, zone flow) |
| `lineage_analysis.py` | LINEAGE_ANALYSIS.md, lineage_tree.png | Phylogenetic trees, selective sweeps, bias evolution (CRN-aware) |
| `compare_runs.py` | comparison_*.png, report.md | Side-by-side run comparison (dynamics + OEE + CRN + environmental pressure) |
| `spatial_analysis.py` | SPATIAL_ANALYSIS.md, spatial_*.png | Spatial distribution, zone occupation |
| `bonding_analysis.py` | BONDING_ANALYSIS.md, bonding_*.png | Cluster persistence, bond network |
| `burst_analysis.py` | BURST_ANALYSIS.md, filmstrip_*.png | Frame-by-frame movement tracking |
| `validate.py` | VALIDATION_REPORT.txt, validation.png | 15-20 automated checks incl. "waste creates pressure" (stable output dirs) |

Validate output uses stable dirs (`validate_neural_10000t/`) not timestamped. Compare uses run names (`comparison_<run1>_vs_<run2>/`).

---

## 9. Empirical Findings

Historical results (v5.0 baselines, v6.0 attenuation, v7.0 waste-too-mild) are archived in git history. This section covers the current v7.1 results.

### Phase 3 Results: Tuned Waste + Waste Sensing (v7.1)

**What changed:** WASTE_PRODUCTION_RATE 0.01→0.03. WASTE_TOXICITY_THRESHOLD 0.5→0.3. WASTE_TOXICITY_RATE 0.05→0.2. LIGHT_ATTENUATION_K 0.03→0.02. Sensory input[15] changed from age to waste concentration. CRN C7 reads waste automatically via `_SENSORY_MAP[7]=15`. Toxicity reads from read buffer (bug fix).

### CRN 200k ticks — Waste-Driven Zone Migration (LANDMARK RESULT)

**The most significant emergent behavior observed in the project.** CRN cells achieved full zone migration without evolving movement — waste created a death gradient that passively dispersed the population from the bright zone into dim and dark zones.

**Zone migration trajectory:**

| Tick | Pop | Bright | Dim | Dark | Avg X | Waste | Move |
|------|-----|--------|-----|------|-------|-------|------|
| 0 | 1015 | 100% | 0% | 0% | 85 | 0.00 | 40% |
| 30k | 184 | 64% | 36% | 1% | 150 | 0.30 | 1.1% |
| 70k | 258 | 59% | 29% | 12% | 191 | 0.29 | 0.4% |
| 100k | 410 | 50% | 25% | 25% | 229 | 0.21 | 0.2% |
| 130k | 280 | 28% | 24% | 48% | 301 | 0.08 | 0.4% |
| 200k | 292 | **26%** | **48%** | **26%** | **255** | 0.07 | 0.0% |

**Mechanism — passive dispersal, not active movement:**
1. Cells cluster in bright zone, produce waste (rate 0.03 per energy gained)
2. Waste exceeds 0.3 threshold in dense areas (peak 0.58)
3. Bright-zone cells take membrane damage and die
4. Division places daughter cells at cluster edges, drifting rightward
5. Over 200k ticks, population center of mass migrates from x=85 to x=255
6. Dim/dark zone cells experience near-zero waste (0.046 avg) — survive longer

**CRN genome evolution:**
- Eat bias: +0.30 → **+0.73** (maximizing energy extraction in low-light zones)
- Divide bias: +0.20 → **+0.48** (aggressive reproduction under waste pressure)
- Move bias: +0.30 → +0.21 (never evolved movement — confirms passive mechanism)
- Reaction topology shifted from 12 sensory→action (direct) to multi-layer pipeline: 3 sensory→hidden + 2 hidden→hidden + 2 hidden→action
- NOT-gates: 0% → **15.3%** inverted thresholds
- 12.3/16 reactions active at 200k

**Population:** Mean 274, stable with 2 respawn events. Far more viable than v7.0 (mean 176).

**This is niche construction through waste.** No intelligence in the rules — waste physics + natural selection produced zone colonization that was never hard-coded. The "bright zone as ecological trap" predicted by research.md was confirmed and broken by waste pressure alone.

### Neural 175k ticks — Hyper-Predation Population Collapse

**Neural evolved territorial predators that destabilize the population — an evolutionary dead end.**

**Respawn-massacre cycle (starting ~100k ticks):**

| Tick | Pop before | Attack % | Pop after |
|------|-----------|----------|-----------|
| 101k | 108 | 31% | 60 |
| 131k | 148 | 30% | 55 |
| 145k | 191 | 36% | 85 |
| 156k | 186 | 39% | 72 |
| 166k | 201 | **43%** | 70 |

Attack rate escalates over time (31% → 43%). Established cells immediately kill respawned naive cells. Population oscillates between ~50 (respawn threshold) and ~200 (post-respawn spike) every ~10k ticks.

**Late-run phase shift (150k+):**

| Metric | 50-100k | 150-175k | Change |
|--------|---------|----------|--------|
| Population | 116 | 69 | -41% |
| Movement | 23.3% | **7.2%** | -69% |
| Attack | 1.8% | **5.2%** | +189% |
| Avg energy | 29.8 | **54.9** | +84% |

The remaining cells are territorial predators — high energy, low movement, elevated attack. They also migrated to dim zone (32% bright, 66% dim at 175k) by clearing the bright zone through predation.

**Decision: Focus on CRN.** Neural's hyper-predation is a population viability issue, not an interesting evolutionary strategy. CRN shows genuinely novel emergent behavior (waste-driven niche construction) with stable populations.

### v7.1 Comparison Table

| Metric | CRN (200k) | Neural (175k) |
|--------|------------|---------------|
| Population (mean) | **274** | 138 |
| Bright zone (final) | 26% | 32% |
| Dim zone (final) | **48%** | **66%** |
| Movement (final) | 0.0% | 0.0% |
| Attack (final) | 0.0% | 1.6% |
| Bonding (mean) | 8.1% | 10.4% |
| Respawn events | 2 | **8** |
| Waste at cells (mean) | 0.16 | 0.34 |
| MI (mean) | 0.0056 | **0.0111** |
| Shannon (mean) | 5.49 | 8.03 |
| Viable at 200k | **Yes** | No (death spiral) |

---

## 10. Known Issues — Priority Order

### Fixed in v5.0–v7.0
Issues 1-6, 12 fixed in prior versions. See git history for details.

### Fixed in v7.1
| # | Issue | Fix Applied |
|---|-------|-------------|
| 7c | CRN movement 0%, no waste sensing | Waste sensing added (input[15]=waste, CRN C7 reads automatically). Waste params tuned (production 3x, threshold 0.3, toxicity 4x). CRN still 0% movement but achieves zone migration through passive waste-driven dispersal. |
| 11 | Population viability at 200k | LIGHT_ATTENUATION_K reduced 0.03→0.02. CRN now mean 274 pop at 200k (was 176). |
| 13 | Waste params too mild | WASTE_PRODUCTION_RATE 0.01→0.03, threshold 0.5→0.3. 67-71% of cells now above toxicity. Peak waste 0.58. |
| 14 | Tests failing (22/28) | Taichi double-init in pytest fixed via conftest.py. Stale function signatures updated. 28/28 passing. |

### Remaining Issues
| # | Issue | Root Cause | Fix Direction |
|---|-------|------------|---------------|
| 8b | CRN predation zero | CRN attack bias -0.3, no bootstrap reaction for predation. | Not urgent — waste-driven dispersal is more interesting than predation for CRN. May add gape-limited predators as external pressure (see research.md). |
| 9 | Bond signals unused | CRN maps only 8 actions; bond signal outputs (10-13) are zeroed. | Needs waste-transport-through-bonds or other multicellularity mechanism first. |
| 15 | Neural hyper-predation | Neural evolves 30-43% attack by 100k ticks, creating respawn-massacre cycles. Population oscillates 50-200, no sustained evolution. | Not fixing — focusing on CRN. Neural retained as baseline only. |
| 16 | CRN can't discover waste→move | 16 reactions with 3 bootstrap reactions. 13 random reactions haven't wired waste→move in 200k ticks. Search space too large. | Consider adding 4th bootstrap reaction (waste→move), or expand to 24-32 reactions for more neutral network connectivity. See research.md for duplication-divergence mutation approach. |
| 17 | Bonding declines under waste | Waste punishes clustering. Bonding dropped: CRN 13.9%→8.1%, neural 15.9%→10.4%. Waste may select against multicellularity. | Make bonds solve waste: waste transport through bonds, metabolic efficiency for bonded cells (25-35% less waste). See research.md syntrophy/channel architecture sections. |

---

## 11. Future Roadmap (Do Not Implement Yet)

### Next Priority: Make Bonds Solve Waste (drives multicellularity)
Waste currently punishes clustering — bonds must become the *solution* to waste, not its victim. Three mechanisms from research.md (implement in this order):
1. **Waste transport through bonds.** Bonds distribute waste across the cluster network. Each bonded cell experiences `total_cluster_waste / cluster_size` instead of local waste. Creates increasing returns to cluster connectivity.
2. **Metabolic efficiency for bonded cells.** Clusters produce 25-35% less waste per cell than solo cells. Reflects real thermodynamic efficiency of cooperative metabolism.
3. **Waste as food (syntrophy).** Some cells metabolize waste for energy. Creates obligate mutualism — the strongest path to stable multicellularity. Cheater-resistant because waste removal benefits the consumer directly.

### Then: Expand CRN Evolvability
- Expand genomes to 24-32 reactions (more neutral network connectivity for innovation).
- Duplication-divergence mutation (biology's primary mechanism for new functions).
- Seed 2-3 stepping-stone reactions (waste→signal, signal→move) as discoverable intermediates.
- Consider behavioral diversity pressure (reward novel behavioral profiles alongside fitness).

### Then: Add Predation as Complementary Force
- Gape-limited predators that cannot consume clusters above 4 cells.
- Self-regulating predator population (Lotka-Volterra: reproduce on kills, die from starvation).
- Waste rewards dispersal, predation rewards clustering — organisms that solve both simultaneously occupy a novel fitness peak.

### Later: Richer Environment
- Mosaic of 4-6 qualitatively different zones with gradient corridors.
- POET-style adaptive difficulty (auto-tune zone parameters to keep evolution at capability edge).
- Waste metabolism zone (waste convertible to energy), scarcity zone, predator zone.

### Later: LLM-Guided Environment Design
- Connect env_api to LLM monitoring OEE metrics.
- LLM adjusts environment to keep evolution productive.
- Based on OMNI/OMNI-EPIC approach. LLM shapes environment only, never organisms.

### Later: Scale and Publish
- Increase world size, target 1M+ cells.
- Formalize GNN interpretation of evolved multicellular organisms.
- Publish CRN waste-driven niche construction results.

---

## 12. Success Criteria

### Stage 1 — Ecosystem Viability: ACHIEVED
Stable populations, chemical cycling, day/night shifts.

### Stage 2 — Behavioral Evolution: ACHIEVED
Chemotaxis (30-45% movement), multiple strategies, stable carrying capacity.

### Stage 3 — Ecological Complexity: ACHIEVED (waste-driven)
CRN achieved waste-driven zone migration — niche construction through metabolic byproducts. 26% bright, 48% dim, 26% dark at 200k ticks. Population center migrated from x=85 to x=255. Waste creates genuine density-dependent selection pressure (67-71% of cells above toxicity threshold). This is emergent ecological dynamics without programmed behavior.

### Stage 4 — Functional Multicellularity: BLOCKED
Bonding *decreased* under waste pressure (CRN 13.9%→8.1%) because waste punishes clustering. **Bonds must become the solution to waste** — waste transport through bonds, metabolic efficiency for clusters, syntrophy. This is the immediate next priority. See research.md for biological mechanisms.

### Stage 5 — Sustained OEE: PROGRESSING
CRN reaction topology still evolving at 200k — shifted from direct sensory→action to multi-layer sensory→hidden→action pipeline. NOT-gates at 15.3%. Eat bias evolved from +0.30 to +0.73 adapting to dim/dark zone life. Population stable (mean 274, no death spiral). However, MI (0.0056) is lower than neural — CRN substrate needs expanded evolvability.

### Stage 6 — Distributed Computation: FUTURE
Requires multicellularity to stabilize first.

---

## 13. What to Do If Evolution Stalls

1. **Energy balance wrong.** Adjust PHOTOSYNTHESIS_RATE and BASAL_METABOLISM.
2. **Mutation rate wrong.** Try 0.001 to 0.1.
3. **Environment too uniform.** Patchier resources, more gradient noise.
4. **No diversity.** Enable archipelago, increase ISLAND_ENV_VARIANCE or MIGRATION_COUNT.
5. **Division too easy/hard.** Target ~500-1000 ticks between divisions for successful cells.
6. **No competitive pressure.** Ensure scarcity. Seed predators if needed.
7. **Substrate too limited.** Switch neural->CRN. If CRN with 16 reactions insufficient, try 32.
8. **Check OEE.** If change is positive but complexity is flat, environment needs a qualitative shift.

---

## 14. Testing

**Unit tests (28 tests, ~3s):** Run `python -m pytest tests/ -v` or just start the simulation — `main.py` runs preflight tests automatically (skip with `--skip-tests`). Tests use a shared `conftest.py` for session-scoped `ti.init()` — never call `ti.init()` in individual test files (causes field corruption).

**Validation harness (15-20 checks, 1-5 min):** Run `PYTHONPATH=. python validate.py --ticks 30000 --genome crn` after significant changes. Tests population stability, bond dynamics, gradient noise, archipelago, predation, diversity, energy balance, cluster analysis, waste pressure, plus CRN-specific checks (hidden zone activation, reaction diversification).

Every subsystem must be testable in isolation: conservation laws, determinism, boundary conditions, edge cases.

---

## 15. References

- **Flow-Lenia** (Plantec et al., 2025): Mass conservation in CA. Informed chemical conservation.
- **Sensorimotor Lenia** (Hamon et al., 2025): ~130 params produce agency. Informed CRN compactness.
- **OMNI-EPIC** (Zhang et al., ICLR 2025): FM-generated environments. Informed env API design.
- **GReaNs** (Wrobel & Joachimczak): Single encoding for GRN + neural. Informed hybrid substrate.
- **Dolson et al. (MODES)**: Formal OEE metrics. Directly implemented.
- **ProtoEvo** (Cope, ALIFE 2023): Multicellular evolution with gene regulation. Closest existing project.
- **Chromaria** (Soros & Stanley, 2014): Niche construction sustains novelty. Informed archipelago design.
- **Michod & Nedelcu (2003)**: Dirty work hypothesis — somatic cells evolve to handle mutagenic waste. Informs waste-as-multicellularity-driver.
- **Goldsby et al. (2014)**: Avida digital organisms evolved somatic division of labor. Validates dirty work hypothesis computationally.
- **Boraas et al. (1998)**: *Chlorella* evolved 8-cell colonies under predation in <100 generations. Informs gape-limited predator design.
- **Bendixsen et al. (2019)**: Neutral drift through genotype networks increases adaptation rates. Informs genome expansion to 24-32 reactions.
- See `research.md` for full literature review on syntrophy, waste-driven multicellularity, and CRN evolvability.

---

## Summary for Claude Code

1. **v7.1 state:** Waste is a real selective pressure (67-71% of cells above toxicity). CRN achieved waste-driven zone migration at 200k (26% bright, 48% dim, 26% dark). Neural evolves hyper-predation and collapses — focus all development on CRN. Neural code retained as comparison baseline — never delete it.
2. **Immediate priority:** Make bonds solve the waste problem. Waste currently punishes clustering, which blocks multicellularity. Three mechanisms (in order): waste transport through bonds, metabolic efficiency for bonded cells, waste-as-food (syntrophy). See research.md for biological precedents.
3. **Secondary priority:** Expand CRN evolvability. Current 16 reactions can't discover waste→move. Expand to 24-32 reactions, add duplication-divergence mutation, seed 2-3 stepping-stone reactions. See research.md Section 3.
4. **Later priority:** Add gape-limited predators as complementary pressure. Waste rewards dispersal, predation rewards clustering — conflicting pressures drive innovation. See research.md Section 5.
5. Keep modules under 300 lines.
6. Run `python -m pytest tests/` after changes (28 tests, ~3s). Run `PYTHONPATH=. python validate.py --ticks 30000 --genome crn` for validation. Run `PYTHONPATH=. python analysis/run_all.py` for full analysis. Preflight tests run automatically on `main.py` start.
7. Config values in `config.py` are the source of truth — this document may lag behind.
8. Design question: "Does this create pressure for complex behavior, or can a simple strategy still win?" New corollary from v7.1: "Does this make clustering the *solution* to the problem, not its victim?"
