# CyberCell: Evolutionary Intelligence Simulation — Project Brief

**Document version:** v6.0 — Beer-Lambert light attenuation (density-dependent shading). CRN movement 1.4% → 8.9% (6.4x increase). MI improved 4.3x (CRN) and 1.5x (neural). Predation absorption 12% → 35%. Sessile optimum partially broken — cells spread into dim zone under density pressure. Empirical findings from 50k-tick comparative runs with attenuation documented.

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
500x500 toroidal grid. Three light zones (bright/dim/dark). Day/night cycle (period 1000 ticks). Patchy resource deposits (radius 10, 20% relocate every 25k ticks). Archipelago: 4 soft-wall quadrants with +/-30% parameter variance, uniform-random migration (1 cell every 200 ticks — continuous trickle matching Wright's Nm≈1 rule). **Beer-Lambert light attenuation**: density-dependent shading reduces photosynthesis in crowded areas. `local_density` computed in radius-2 neighborhood; `light *= exp(-k * density)` with k=0.03. Creates 26-45% light reduction at typical occupied densities (10-20 neighbors).

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
| `KILL_ABSORPTION_RATE` | 0.35 | ~3 kills to recoup prey lifetime; viable predation economics |
| `KILL_ENERGY_BONUS` | 0.0 | No flat bonus — predation profit from absorption only |
| `LIGHT_ATTENUATION_ENABLED` | True | Beer-Lambert density-dependent shading |
| `LIGHT_ATTENUATION_K` | 0.03 | Extinction coeff: 74% at density 10, 55% at density 20 |
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
- `metrics.jsonl` — population, energy, movement, attack, bond, light attenuation stats (every 1k ticks)
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

### Phase 1 Results: Beer-Lambert Light Attenuation (v6.0, k=0.03)

**What changed:** Density-dependent shading via `light *= exp(-0.03 * local_density)`. Predation absorption 12% → 35%. Cells in crowded areas get 55-74% of base light depending on density.

**CRN with attenuation — 50k ticks:**
- Population: 317 (down from 3498 — expected with density pressure)
- Movement: **8.9%** (up from 1.4% — **6.4x increase**, sessile optimum partially broken)
- Bonding: **13.9%** (up from 7.3% — 1.9x)
- Attack: 0.0% (unchanged)
- Avg X position: 101.4 (expanding from light zone center at 83)
- Avg local density: 16.6 (very clustered, 92% of cells have 5+ neighbors)
- MI: **0.0039** (up from 0.0009 — **4.3x increase**)
- Shannon: 5.70 (down from 8.07 — lower pop = fewer unique genomes)
- OEE entropy: 8.23 (down from 11.64)

**Neural with attenuation — 50k ticks:**
- Population: 512 (down from 1155)
- Movement: 25.2% (similar to 29% baseline)
- Bonding: **15.9%** (up from 10.7% — 1.5x)
- Attack: **0.65%** (up from 0.1-0.2% — 3x, predation emerging)
- Avg X position: 125.8 (into dim zone, significant expansion)
- Avg local density: 14.5
- MI: **0.006** (up from 0.0039 — **1.5x increase**)
- Shannon: 6.13 (vs 7.05 baseline)
- Max observed age: 36,601 ticks (7x nominal max age)

**Phase 1 comparison table:**

| Metric | CRN v5.0 | CRN+atten | Δ | Neural v5.0 | Neural+atten | Δ |
|--------|----------|-----------|---|-------------|--------------|---|
| Population | 3498 | 317 | -91% | 1155 | 512 | -56% |
| Movement | 1.4% | **8.9%** | **+6.4x** | 29% | 25.2% | -13% |
| Bonding | 7.3% | **13.9%** | **+1.9x** | 10.7% | **15.9%** | **+1.5x** |
| Attack | 0.0% | 0.0% | — | 0.1% | **0.65%** | **+3x** |
| MI | 0.0009 | **0.0039** | **+4.3x** | 0.0039 | **0.006** | **+1.5x** |
| Shannon | 8.07 | 5.70 | -29% | 7.05 | 6.13 | -13% |

**Key insight:** Light attenuation successfully breaks the sessile optimum for CRN — movement increased 6.4x. Both genomes show increased bonding and MI. Neural cells spread into the dim zone (avg X: 110 → 125.8). However, k must be carefully tuned: k≥0.10 crashes CRN due to cohort die-offs at low carrying capacity. The k=0.03 value balances density pressure against population viability. Population is lower but more behaviorally complex.

**Tuning lessons learned:**
- Cells cluster to density 10-17 at occupied positions (much higher than uniform distribution predicts)
- Day/night cycle halves effective photosynthesis — k must account for this
- CRN carrying capacity is much more sensitive to k than neural (sessile strategy clusters tighter)
- Deposit relocation + attenuation is too aggressive in combination — attenuation alone suffices
- k=0.5 (plan original) → extinction. k=0.10 → CRN extinction at 50k. k=0.05 → CRN extinction at 35k. k=0.03 → both survive 50k

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

### Partially Fixed in v6.0
| # | Issue | Status |
|---|-------|--------|
| 4 | MI near zero | **Improved.** CRN MI 0.0009 → 0.0039 (4.3x). Neural MI 0.0039 → 0.006 (1.5x). Still low but trending up. |
| 7 | CRN evolves away movement | **Partially fixed.** CRN movement 1.4% → 8.9% (6.4x). Sessile optimum weakened but not fully broken. |
| 8 | No predation evolves | **Improved.** Neural attack 0.1% → 0.65% (3x) with 35% absorption. Predation emerging but still marginal. |

### Remaining Issues
| # | Issue | Root Cause | Fix Direction |
|---|-------|------------|---------------|
| 7b | CRN movement still low (8.9%) | k=0.03 is gentle; cells cluster to density 16 and tolerate the light reduction. Movement evolves up but doesn't dominate. | Phase 2: metabolic waste (local toxicity from photosynthesis) or seasonal light to make staying in one spot actively harmful |
| 8b | Predation still marginal | Even at 35% absorption, resources are abundant enough that predation is a poor strategy vs photosynthesis. | Need scarcity events or predation-specific advantages (speed, armor) |
| 9 | Bond signals unused | CRN maps only 8 actions; bond signal outputs (10-13) are zeroed. Neural has signals but no evolutionary pressure to use them. | Needs functional multicellularity pressure first |
| 10 | CRN crn_metrics logger partial | Zone means and biases show as 0 in analysis trajectory extraction (raw JSONL data is correct). | Fix key mapping in metrics extraction code |
| 11 | CRN sensitive to k value | CRN carrying capacity drops sharply with k (cohort die-offs at low pop). k≥0.05 causes extinction at 50k. | Consider adding age jitter to prevent synchronized die-offs, or use CRN-specific k |
| 12 | compare_runs.py swaps labels | Script labels first arg as "Neural" and second as "CRN" regardless of actual genome type. | Detect genome type from presence of crn_metrics.jsonl |

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

### Stage 3 — Ecological Complexity: PROGRESSING
Predation absorption increased to 35%, neural attack rate up 3x to 0.65%. CRN bonding up to 13.9%, neural to 15.9%. Light attenuation creates density-dependent resource competition. Arms races not yet visible but ecological dynamics more complex.

### Stage 4 — Functional Multicellularity: IN PROGRESS
Bonding increased under density pressure (CRN 7.3%→13.9%, neural 10.7%→15.9%). Clusters form but bond signals still unused. Need environmental pressure where group behavior provides survival advantage over solo cells.

### Stage 5 — Sustained OEE: MIXED
MI improved significantly (CRN 4.3x, neural 1.5x) — sensory-action coupling is emerging. Both runs show `plateaued: false` at 50k ticks. But Shannon diversity and OEE entropy decreased due to lower carrying capacity. The trade-off: more behavioral complexity but less genetic diversity.

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

1. v6.0: Beer-Lambert light attenuation implemented. CRN movement 6.4x increase, MI 4.3x increase. Sessile optimum partially broken.
2. CRN is the primary genome for pushing complexity. Neural is the comparison baseline — never delete it.
3. **Immediate priority:** Phase 2 interventions to fully break the sessile optimum. CRN movement at 8.9% is better but not enough for active foraging. Options: metabolic waste system (local toxicity), seasonal light variation, or size-dependent predation resistance.
4. **Secondary priority:** Run both genomes to 200k+ ticks with current settings to see if complexity continues increasing. MI and bonding trends are positive.
5. **Caution:** k=0.03 is the sweet spot for both genomes. Do NOT increase k above 0.05 — CRN goes extinct due to cohort die-offs at low carrying capacity. Do NOT combine fast deposit relocation with attenuation — the combined pressure is too harsh.
6. Keep modules under 300 lines.
7. Run `python validate.py --ticks 30000` after changes. Run `python analysis/run_all.py` for full analysis.
8. Config values in `config.py` are the source of truth — this document may lag behind.
9. Design question: "Does this create pressure for complex behavior, or can a simple strategy still win?"
