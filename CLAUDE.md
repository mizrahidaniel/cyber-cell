# CyberCell: Evolutionary Intelligence Simulation — Project Brief

**Document version:** v5.0 — Six critical issues fixed (v4.0 issues 1-6). CRN sigmoid action firing, per-reaction decay bug fix, hidden zone basal production, auto-bonds at division, ecological predation economics, continuous migration trickle. CRN population now 3x neural. Empirical findings from 50k-tick comparative runs documented.

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
500x500 toroidal grid. Three light zones (bright/dim/dark). Day/night cycle (period 1000 ticks). Patchy resource deposits (radius 10, 20% relocate every 25k ticks). Archipelago: 4 soft-wall quadrants with +/-30% parameter variance, uniform-random migration (1 cell every 200 ticks — continuous trickle matching Wright's Nm≈1 rule).

### Chemistry
4 chemicals: Energy (E), Structure (S), Replication Material (R), Signal (G). Double-buffered diffusion. Gradient noise (sigma=0.15) on all 6 gradient sensing channels.

### CyberCell
34 sensory inputs (18 base + 16 bond signals), 14 action outputs (10 base + 4 bond signal emission). Full state: position, energy, structure, repmat, signal, membrane, age, genome_id, facing, bonds, bond strength/signals, last_attacker.

### Genome — Two Types

**Neural Network (`GENOME_TYPE = "neural"`, default):**
Feedforward 34->32->32->14. 2,638 parameters. Mutation: perturbation (0.03), reset (0.001), knockout (0.0005).

**Chemical Reaction Network (`GENOME_TYPE = "crn"`):**
16 internal chemicals in 3 zones, 16 reaction rules. 120 parameters (112 reaction + 4 action biases + 4 hidden decays). See Section 5 for architecture details.

### Energy Model
Photosynthesis, chemical consumption, predation with ecological kill rewards (12% absorption, no flat bonus). Full cost table in `config.py`. Death from starvation, membrane failure, old age.

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
C3 <- S_gradient_x     C7 <- age
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
- **Attack is age-gated** (C7->C15): mapping cell_ahead->attack caused mass extinction.
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
| `KILL_ABSORPTION_RATE` | 0.12 | Ecological ~10-15% trophic transfer; prevents predation spirals |
| `KILL_ENERGY_BONUS` | 0.0 | No flat bonus — predation profit from absorption only |
| `ATTACK_BIAS` | -0.3 | Initial sigmoid output ~0.43 — predation must be evolved |
| `CRN_ACTION_GAIN` | 30.0 | Sigmoid steepness: <0.3% at bias, >95% when boosted by reactions |
| `CRN_ACTION_CENTER` | 0.5 | Sigmoid midpoint matches old threshold |
| `CRN_HIDDEN_BASAL` | 0.005 | Steady-state 0.25, below aux action threshold of 0.5 |
| `CRN_SENSORY_BLEND` | 0.5 | 50% memory / 50% environment per tick |
| `DEPOSIT_RELOCATE_INTERVAL` | 25000 | Forces navigation; static deposits allow sessile strategies |
| `MIGRATION_INTERVAL` | 200 | Continuous trickle (was 5000); Nm≈1.25 per generation |
| `MIGRATION_COUNT` | 1 | Per-island per event; uniform random selection (was fitness-proportional) |
| `ISLAND_ENV_VARIANCE` | 0.3 | ±30% parameter variance; more distinct islands |

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
- `metrics.jsonl` — population, energy, movement, attack, bond stats (every 1k ticks)
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
| `study.py` | STUDY.md, evolution_report.png | Phase detection, rates, 6-panel dynamics |
| `crn_analysis.py` | CRN_ANALYSIS.md, crn_evolution.png | 9-panel CRN diagnostics (zone activation, bias drift, reaction graph, zone flow) |
| `lineage_analysis.py` | LINEAGE_ANALYSIS.md, lineage_tree.png | Phylogenetic trees, selective sweeps, bias evolution (CRN-aware) |
| `compare_runs.py` | comparison_*.png, report.md | Side-by-side run comparison (dynamics + OEE + CRN) |
| `spatial_analysis.py` | SPATIAL_ANALYSIS.md, spatial_*.png | Spatial distribution, zone occupation |
| `bonding_analysis.py` | BONDING_ANALYSIS.md, bonding_*.png | Cluster persistence, bond network |
| `burst_analysis.py` | BURST_ANALYSIS.md, filmstrip_*.png | Frame-by-frame movement tracking |
| `validate.py` | VALIDATION_REPORT.txt, validation.png | 13-18 automated checks (stable output dirs) |

Validate output uses stable dirs (`validate_neural_10000t/`) not timestamped. Compare uses run names (`comparison_<run1>_vs_<run2>/`).

---

## 9. Empirical Findings (50k-tick Comparative Runs, v5.0)

### CRN Genome — 50k ticks (post-fix)

**Population dynamics:**
- Initial crash: 1015 → 348 (66% mortality, purifying selection)
- Exponential recovery: 348 → 3498 by 50k (still rising, carrying capacity not reached)
- Growth rate: 5.1% per 1K ticks post-bottleneck

**Evolved strategy — "efficient plant":**
- Movement: 40% initial → **1.4%** (evolved away — sessile strategy dominates)
- Eating: ~95% via sigmoid (bootstrap reaction light→eat reliably fires)
- Attack: 0.0% (never evolves)
- Bonding: 3% → 7.3% (auto-bonds from division, some persist)
- 96% of cells in light zone, 4% dim, 0% dark

**CRN internals:**
- 14.4/16 reactions active — complex networks maintained
- Hidden zone mean: 0.247 (was -0.04 pre-fix — basal production working)
- Reaction zone flow: 8 sensory→action (direct), 6 sensory→hidden (memory pathway)
- NOT-gate logic: 1.2% inverted thresholds (emerging)
- Action biases stable: eat +0.31, move +0.30, divide +0.18, attack -0.31

**Diversity:**
- Shannon entropy: 6.91 → 8.07 (rising — no convergence)
- 3267 unique genomes at 50k
- Root diversity: 881 → 43 (moderate selective sweep, 3x more diverse than neural)

### Neural Genome — 50k ticks (post-fix)

**Population dynamics:**
- Initial crash: 2000 → 631 (68% mortality)
- Stable oscillation: 800-1200 cells (carrying capacity ~1100)

**Evolved strategy — "active forager":**
- Movement: 25% → **29%** (chemotaxis evolved and maintained)
- Attack: 0.1-0.2% (marginal predation)
- Bonding: 100% → 10.7% (bonds from auto-division, declining)
- Broader spatial distribution due to active movement

**Diversity:**
- Shannon entropy: 6.91 → 7.05 (stable)
- Root diversity: 1013 → 13 (aggressive selective sweep)
- Mutation rate: 2.35/tick (3x CRN due to larger genome)

### CRN vs Neural at 50k ticks

| Metric | CRN | Neural | Winner |
|--------|-----|--------|--------|
| Population | **3498** | 1155 | CRN (3x) |
| Movement | 1.4% | **29%** | Neural |
| Bonding | **7.3%** | 10.7% | Similar |
| Shannon entropy | **8.07** | 7.05 | CRN |
| Root diversity | **43** | 13 | CRN (3x) |
| OEE entropy | **11.64** (rising) | 10.17 (falling) | CRN |
| Mutual information | 0.0009 | **0.0039** | Neural |
| Max cluster size | **22** | 14 | CRN |
| Cluster topologies | mesh, chain, star | pairs, chains | CRN |

**Key insight:** CRN achieves 3x population and higher diversity through metabolic efficiency (sessile strategy), while neural achieves behavioral complexity (chemotaxis) at a carrying capacity cost. Neither develops significant sensory-action coupling (MI near zero in both).

---

## 10. Known Issues — Priority Order

### Fixed in v5.0
| # | Issue | Fix Applied |
|---|-------|-------------|
| 1 | CRN movement dead | Sigmoid action firing (gain=30) + per-reaction decay restricted to sensory zone + bootstrap move rate=0.2. Movement now 9% early, evolves to 1.4% (sessile dominates but movement CAN evolve). |
| 2 | CRN hidden zone suppressed | Basal production (0.005/tick, SS=0.25) + zone-aware clamping [0,5] + hidden decay 0.02. Hidden zone now 0.247. |
| 3 | Neural bonding collapses | Bond decay 0.02→0.001 (near-permanent) + auto-bond at division (strength 0.1, breaks ~50 ticks). Bonding stable at 7-11% through 50k ticks. |
| 5 | Selective sweep too fast | Continuous trickle migration (1 cell/200 ticks, uniform random). CRN root diversity 43 vs old 11. |
| 6 | Predation bifurcation | Kill absorption 0.5→0.12, bonus 2.0→0.0. Ecologically realistic; no predation crashes observed. |

### Remaining Issues
| # | Issue | Root Cause | Fix Direction |
|---|-------|------------|---------------|
| 4 | MI near zero in both genomes | Environment too easy — sessile photosynthesis dominates. No pressure for sensory-driven behavior. | Make environment more dynamic: faster deposit relocation, seasonal disruptions |
| 7 | CRN evolves away movement | Sessile strategy is strictly more efficient (saves 0.1/tick move cost). No fitness benefit to exploring. | Force movement necessity: relocate deposits more aggressively (every 5k ticks, 50%), or add spatially-varying hazards |
| 8 | No predation evolves | With 12% absorption and no bonus, predation barely profitable. Prey density high enough that resources aren't scarce. | May need resource scarcity events to create competitive pressure |
| 9 | Bond signals unused | CRN maps only 8 actions; bond signal outputs (10-13) are zeroed. Neural has signals but no evolutionary pressure to use them. | Needs functional multicellularity pressure first |
| 10 | CRN crn_metrics logger partial | Zone means and biases show as 0 in analysis trajectory extraction (raw JSONL data is correct). | Fix key mapping in metrics extraction code |

---

## 11. Future Roadmap (Do Not Implement Yet)

### Phase 3: Expand CRN Substrate
- Variable-length genomes (reaction count grows via duplication).
- Regulatory reactions (reactions enable/disable other reactions).
- Inter-cell chemical coupling through bonds.

### Phase 4: Richer Environment
- Elevation, biomes, weather events, seasonal cycles.
- Spatial vision (light cone sensing).
- Vibration/sound channel for long-range sensing.

### Phase 5: LLM-Guided Environment Design
- Connect env_api to LLM monitoring OEE metrics.
- LLM adjusts environment to keep evolution productive.
- Based on OMNI/OMNI-EPIC approach. LLM shapes environment only, never organisms.

### Phase 6: Scale
- Increase world size, target 1M+ cells.
- CRN vs neural comparison at scale.

### Phase 7: Academic Contribution
- Formalize GNN interpretation of evolved multicellular organisms.
- Publish CRN-vs-NN comparison results.

---

## 12. Success Criteria

### Stage 1 — Ecosystem Viability: ACHIEVED
Stable populations, chemical cycling, day/night shifts.

### Stage 2 — Behavioral Evolution: ACHIEVED
Chemotaxis (30-45% movement), multiple strategies, stable carrying capacity.

### Stage 3 — Ecological Complexity: PARTIALLY ACHIEVED
Predation economics fixed (ecologically realistic) but predation doesn't evolve (insufficient scarcity pressure). Bonds stable at 7-11% with auto-division bonds. Arms races not visible.

### Stage 4 — Functional Multicellularity: IN PROGRESS
Auto-bonds create clusters up to 22 cells with mesh/chain/star topologies. 94% facing coordination. But clusters are stationary, bond signals unused, and no behavioral differentiation. Needs environmental pressure that rewards group behavior.

### Stage 5 — Sustained OEE: PROMISING
CRN entropy rising (9.99→11.64) at 50k ticks, not plateaued. CRN root diversity 3x neural. But MI near zero — innovation is structural (new genomes), not behavioral (new strategies).

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

Run `python validate.py --ticks 30000` after significant changes. The harness tests population stability, bond dynamics, gradient noise, archipelago, predation, diversity, energy balance, cluster analysis, plus CRN-specific checks (hidden zone activation, reaction diversification).

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

---

## Summary for Claude Code

1. Steps 1-14 complete. Six critical issues (v4.0) fixed. CRN genome now viable and outperforms neural in population (3x) and diversity.
2. CRN is the primary genome for pushing complexity. Neural is the comparison baseline — never delete it.
3. **Immediate priority:** Break the sessile optimum. The environment is too easy — photosynthesis in the bright zone is so profitable that movement, predation, and cooperation provide no fitness advantage. More dynamic deposit relocation or seasonal disruptions needed.
4. **Secondary priority:** Run CRN to 200k ticks to see if complexity increases beyond the 50k snapshot. Entropy is still rising.
5. Keep modules under 300 lines.
6. Run `python validate.py --ticks 30000` after changes. Run `python analysis/run_all.py` for full analysis.
7. Config values in `config.py` are the source of truth — this document may lag behind.
8. Design question: "Does this create pressure for complex behavior, or can a simple strategy still win?"
