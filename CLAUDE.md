# CyberCell: Evolutionary Intelligence Simulation — Project Brief

**Document version:** v8.0 — CTRNN genome type (16 CfC neurons, 188 params). CRN bond signal reception (issue #9b fixed). 867k CTRNN: max cluster 77, zone migration (15% bright), hidden tau differentiation (0.93-1.75). 30k validation: CRN 21/21, CTRNN 20/20. Test suite 39/39.

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
5 chemicals: Energy (E), Structure (S), Replication Material (R), Signal (G), Waste (W). Double-buffered diffusion. Gradient noise (sigma=0.15) on all 6 gradient sensing channels. **Metabolic waste**: photosynthesis produces waste proportional to energy gained (rate 0.03; bonded cells produce 25% less with 2+ bonds). Waste diffuses (5%/tick), decays (0.002/tick, half-life ~350 ticks), and causes membrane damage above threshold (0.3) at rate 0.2/tick. **Cells sense waste directly** via sensory input[15]. CRN C7 reads waste automatically through `_SENSORY_MAP[7]=15`. **Syntrophy**: all cells passively convert waste above 0.3 to energy (0.02 per unit consumed, up to 0.05/tick). **Per-cell waste exposure**: env waste is copied to `cell_waste_exposure` field, equalized through bonds, then used for toxicity — enabling bond-mediated waste sharing.

### CyberCell
34 sensory inputs (18 base + 16 bond signals), 14 action outputs (10 base + 4 bond signal emission). Full state: position, energy, structure, repmat, signal, membrane, age, genome_id, facing, bonds, bond strength/signals, last_attacker, waste_exposure.

### Genome — Three Types

**Chemical Reaction Network (`GENOME_TYPE = "crn"`, primary):**
16 internal chemicals in 3 zones, 24 reaction rules (4 bootstrap + 12 random + 8 silent). 176 parameters (168 reaction + 4 action biases + 4 hidden decays). See Section 5 for architecture details. **Bond signal reception (v8.0):** incoming bond signals blended into hidden chemicals at rate 0.15 (CRN_BOND_SIGNAL_BLEND). CRN is the primary genome for established experiments.

**CTRNN (`GENOME_TYPE = "ctrnn"`, new in v8.0):**
16-neuron continuous-time RNN with CfC dynamics (Hasani et al. 2022). 188 parameters. 3-zone structure (8 sensory + 4 hidden + 4 action), sparse recurrence (4 connections/neuron), evolved time constants. See Section 5b for architecture. **CTRNN is the focus for new development** — attractor-based memory, multi-timescale processing, oscillations.

**Neural Network (`GENOME_TYPE = "neural"`, comparison baseline):**
Feedforward 34->32->32->14. 2,638 parameters. Mutation: perturbation (0.03), reset (0.001), knockout (0.0005). Retained for comparison but not actively developed — evolves hyper-predation at long timescales that collapses population viability.

### Energy Model
Photosynthesis (density-attenuated), chemical consumption, predation with ecological kill rewards (35% absorption, no flat bonus). **Environmental predation**: gape-limited mortality sweep every 100 ticks — solo cells face 1% kill probability per sweep, pair cells 0.5%, cells with 2+ bonds immune. Full cost table in `config.py`. Death from starvation, membrane failure, old age, or environmental predation.

### Bonding
Near-permanent bonds (decay 0.001/tick, reinforced at 0.03 when both cells fire bond). Auto-bond at division (incomplete cytokinesis, initial strength 0.1 — breaks in ~50 ticks unless reinforced). 30% lossy resource transfer. 4-channel bond signals per direction. **Bond signal emission (v7.5):** CRN/CTRNN hidden chemicals 10-11 emit on bond signal channels when >0.5 (chem 10 → ch 0-1, chem 11 → ch 2-3). Partners sense via inputs 18-33. **Bond signal reception (v8.0):** incoming bond signals blended into hidden chemicals (8-11) at rate CRN_BOND_SIGNAL_BLEND=0.15. Channel h from all bonded partners is averaged and added to hidden chemical 8+h. Closes the full emit→relay→receive loop for CRN and CTRNN. **Bond-waste mechanics (v7.2):** waste equalization through bonds (20% of diff per tick per bond, strength-weighted), metabolic efficiency (bonded cells produce 25% less waste with 2+ bonds). Tick order: update_waste_exposure → bond equalization → apply_waste_toxicity. Clusters up to 33 cells observed at 200k (v7.4).

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
│   ├── crn_genome.py          <- CRN genome. 16 chemicals (3 zones), 24 reactions
│   ├── ctrnn_genome.py        <- CTRNN genome. 16 neurons (3 zones), CfC dynamics
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
│   ├── ctrnn_analysis.py      <- 9-panel CTRNN diagnostics + report
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
    ├── test_bonding_waste.py  <- Bond-waste mechanics: equalization, efficiency, syntrophy, death tracking, bond signal reception
    ├── test_chemistry.py
    ├── test_ctrnn.py          <- CTRNN: evaluation, bootstrap, mutation, hidden memory
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
Turns are handled separately via facing-aware gradient steering. Hidden chemicals 8-9 drive signal/bond via auxiliary thresholds. **Hidden chemicals 10-11 emit bond signals** (v7.5): when >0.5, chemical 10 → bond signal channels 0-1, chemical 11 → channels 2-3. Sensed by bonded partners via sensory inputs 18-33.

### Genome Layout (176 parameters)
- **Reactions** (0-167): 24 reactions x 7 params (input_a, input_b, output, threshold_a, threshold_b, rate, decay)
- **Action biases** (168-171): eat, move, divide, attack — reset to these values each tick
- **Hidden decays** (172-175): per-chemical decay rates for hidden zone

### Bootstrap Reactions (4)
```
R0: light > 0.2   → eat    (C0 → C12, rate=0.3)
R1: energy > 0.3  → divide (C1 → C14, rate=0.4)
R2: structure > 0.1 → move  (C2 → C13, rate=0.2)
R3: waste > 0.3   → move   (C7 → C13, rate=0.25)  [v7.2]
```
Reactions 4-15: random with biased zone targeting. R3 enables waste-driven movement but only fires when waste exceeds toxicity threshold — mostly dormant in v7.2 since bond-waste mechanics keep waste below 0.3. **Reactions 16-23: silent slots** (v7.4) — wired with random inputs/outputs/thresholds but rate=0. Activated incrementally by mutation or duplication-divergence. Provides evolvable capacity without increasing initial mutation load.

### Key Design Decisions
- **Attack was age-gated** (C7->C15 in v7.0): now waste-gated since C7 reads waste. Mapping cell_ahead->attack caused mass extinction.
- **Move is structure-gated** (C2->C13): bootstrap rate=0.2 gives ~9% initial movement with sigmoid. **Waste→move** (C7->C13, v7.2): bootstrap rate=0.25, provides 0-2.7% movement when waste is high.
- **Sigmoid action firing** (gain=30, center=0.5): replaces hard threshold. P(fire) = sigmoid(30*(chem - 0.5)). At bias 0.3: P≈0.3%. At bias+reaction 0.6: P≈95%. Provides evolutionary gradient below threshold without wasteful spontaneous actions.
- **Hidden basal production** (0.005/tick): steady-state floor at 0.25 (below 0.5 aux threshold). Keeps hidden chemicals positive and computationally useful. Reactions can push above 0.5 to activate signal/bond.
- **Hidden decay 0.02/tick**: half-life ~35 ticks. Long-term memory, prevents runaway.
- **Per-reaction decay restricted to sensory zone only** (0-7): action chemicals reset each tick, hidden has dedicated decay. Fixes bug where actions were decayed 3-5x per tick.
- **Zone-aware clamping**: sensory/hidden [0, 5], action [-1, 5]. Prevents negative concentrations in sensory/hidden zones.

---

## 5b. CTRNN Genome Architecture (v8.0)

The CTRNN uses Closed-form Continuous-time (CfC) dynamics — a computationally efficient approximation of continuous-time RNNs that supports attractor-based memory, multi-timescale processing, and oscillations.

### CfC Update Rule

```
f_i = sigmoid(sum_j(w_ij * tanh(y_j)) + bias_i)
y_i(t+1) = (y_i(t) + f_i * A_i) / (1 + 1/tau_i + f_i)
```

### 3-Zone Neuron Structure (16 neurons)

| Zone | Indices | Role | tau range |
|------|---------|------|-----------|
| Sensory | 0-7 | Environment inputs, blended 50/50 with env | 0.2-0.5 (fast) |
| Hidden | 8-11 | Memory, oscillators, gates | 1.0-3.0 (slow) |
| Action | 12-15 | Drive behavioral outputs | 0.5-1.0 (moderate) |

Same sensory mapping as CRN (light, energy, structure, gradients, cell_ahead, bonds, waste). Same action mapping (eat, move, divide, attack). Same auxiliary actions from hidden neurons (signal, bond, bond signals).

### Genome Layout (188 parameters)

| Block | Indices | Content |
|-------|---------|---------|
| Per-neuron | 0-175 | 16 × 11 (tau, bias, A, 4 recurrent weights, 4 target indices) |
| Input gains | 176-183 | 8 sensory neuron input coupling strengths |
| Action biases | 184-187 | eat=0.3, move=0.3, divide=0.2, attack=-0.3 |

### Sparse Recurrence

Each neuron connects to 4 others (target indices encoded as continuous values, converted to neuron index via `int(abs(val) * 16) % 16`). Rewiring mutations change targets. This matches CRN's reaction-slot model.

### Bootstrap Circuit

- Neuron 12 (eat) ← neuron 0 (light): strong positive weight
- Neuron 14 (divide) ← neuron 1 (energy): positive weight
- Neuron 13 (move) ← neuron 2 (structure) + neuron 7 (waste): positive weights
- Action neuron biases = -2.0 (suppressed without input), action bias offsets match CRN

### Key Design Decisions

- **Sensory neurons: pure blend, no CfC.** CfC dynamics on sensory neurons created ~0.25 baseline regardless of input. Pure blend (like CRN) gives faithful environment tracking.
- **Action neurons: feedforward readout, no CfC.** Persistent CfC state caused constant action firing (70% movement). Reset-each-tick readout matches CRN's action chemical behavior. Action biases added after readout to shift sigmoid operating point.
- **Auxiliary action threshold 1.0 (not 0.5).** CfC hidden neurons reach steady-state ~0.85 from dynamics alone. Threshold 1.0 requires genuine recurrent drive for signal/bond emission.
- **Action neuron init: zero non-bootstrap weights.** Random connections created positive bias pushing all actions above threshold. Bootstrap-only init lets evolution add connections incrementally.
- **Neurons clamped to [-5, 5]**: prevents runaway activations.
- **Mutation calibrated**: 0.013 per param x 188 params = 2.4 mutations/gen (matches CRN).
- **Bond signal reception**: same mechanism as CRN — incoming bond signals blended into hidden neurons at rate 0.15.

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
| `MAX_REACTIONS` | 24 | 4 bootstrap + 12 random + 8 silent (v7.4, was 16) |
| `CRN_GENOME_SIZE` | 176 | 24×7+8 (v7.4, was 120) |
| `CRN_MUTATION_RATE_PERTURB` | 0.014 | Scaled from 0.02 by 120/176 to hold ~2.4 mutations/gen |
| `CRN_ACTION_GAIN` | 30.0 | Sigmoid steepness: <0.3% at bias, >95% when boosted by reactions |
| `CRN_ACTION_CENTER` | 0.5 | Sigmoid midpoint matches old threshold |
| `CRN_HIDDEN_BASAL` | 0.005 | Steady-state 0.25, below aux action threshold of 0.5 |
| `CRN_SENSORY_BLEND` | 0.5 | 50% memory / 50% environment per tick |
| `CRN_BOND_SIGNAL_BLEND` | 0.15 | Blend rate for incoming bond signals into hidden chemicals |
| `DEPOSIT_RELOCATE_INTERVAL` | 25000 | Forces navigation; static deposits allow sessile strategies |
| `MIGRATION_INTERVAL` | 200 | Continuous trickle (was 5000); Nm≈1.25 per generation |
| `MIGRATION_COUNT` | 1 | Per-island per event; uniform random selection (was fitness-proportional) |
| `ISLAND_ENV_VARIANCE` | 0.3 | ±30% parameter variance; more distinct islands |
| `WASTE_ENABLED` | True | Metabolic waste from photosynthesis |
| `WASTE_PRODUCTION_RATE` | 0.05 | Waste per unit energy gained. Solo SS ~0.35+ (above threshold), bonded pair SS ~0.26 (safe). Tuned up from 0.03 in v7.2. |
| `WASTE_DECAY_RATE` | 0.002 | Half-life ~350 ticks |
| `WASTE_DIFFUSION_RATE` | 0.05 | 5% spreads per tick (same as signal) |
| `WASTE_TOXICITY_THRESHOLD` | 0.3 | Isolated cells (0.26) safe. Cluster edges (~0.40) take damage. Lowered from 0.5 in v7.1. |
| `WASTE_TOXICITY_RATE` | 0.2 | 4x increase from v7.1. Interior damage ~0.04 membrane/tick (~2500 ticks to die). |
| `SYNTROPHY_ENABLED` | True | All cells passively convert waste→energy above threshold |
| `SYNTROPHY_THRESHOLD` | 0.3 | Same as toxicity threshold — syntrophy activates where waste hurts |
| `SYNTROPHY_RATE` | 0.02 | Energy per unit waste consumed (~0.001 energy/tick max) |
| `SYNTROPHY_CONSUMPTION_RATE` | 0.05 | Max waste consumed per tick per cell |
| `BOND_WASTE_EQUALIZATION` | True | Bonds redistribute waste exposure across cluster |
| `BOND_WASTE_EQUALIZE_RATE` | 0.2 | Fraction of waste diff equalized per tick per bond (strength-weighted) |
| `BOND_METABOLIC_EFFICIENCY` | True | Bonded cells produce less waste during photosynthesis |
| `BOND_METABOLIC_EFFICIENCY_RATE` | 0.25 | Max waste reduction (25%) with 2+ bonds. Solo SS ~0.35+, bonded pair SS ~0.26. |
| `PREDATION_ENABLED` | True | Gape-limited environmental predation active |
| `PREDATION_INTERVAL` | 100 | Ticks between predation sweeps |
| `PREDATION_SOLO_KILL_PROB` | 0.01 | Per-sweep kill prob for 0 bonds (~9.5% per 1000 ticks) |
| `PREDATION_PAIR_KILL_PROB` | 0.005 | Per-sweep kill prob for 1 bond (~5% per 1000 ticks) |
| `PREDATION_IMMUNE_BONDS` | 2 | Cells with >= this many bonds are immune |
| `RESPAWN_ENERGY` | 80.0 | Higher than INITIAL_ENERGY (35) so respawned cells survive division + night cycle. Parent gets 36 energy after divide, daughter 24 — both survive a full night. |
| `CTRNN_GENOME_SIZE` | 188 | 16x11 per-neuron + 8 input gains + 4 action biases |
| `CTRNN_MUTATION_RATE_PERTURB` | 0.013 | ~2.4 mutations/gen (188x0.013) |

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
- `metrics.jsonl` — population, energy, movement, attack, bond, light attenuation, zone breakdown, waste stats, **4-way death tracking** (starvation/age/waste/predation + zone×cause matrix), **cluster stats** (num_clusters, avg/max size, bonded_fraction), **division stats** (count, avg daughter dx/dy) (every 1k ticks)
- `oee_metrics.jsonl` — Bedau activity, MODES, entropy, MI, bond density (every 1k ticks)
- `crn_metrics.jsonl` — CRN-only: zone activations, biases, decays, reactions, zone flow (every 1k ticks)
- `ctrnn_metrics.jsonl` — CTRNN-only: neuron activations, tau/bias/amp evolution, action biases (every 1k ticks)
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
| `ctrnn_analysis.py` | CTRNN_ANALYSIS.md, ctrnn_evolution.png | 9-panel CTRNN diagnostics (neuron activations, tau/bias/amp evolution, action patterns) |
| `lineage_analysis.py` | LINEAGE_ANALYSIS.md, lineage_tree.png | Phylogenetic trees, selective sweeps, bias evolution (CRN-aware) |
| `compare_runs.py` | comparison_*.png, report.md | Side-by-side run comparison (dynamics + OEE + CRN + environmental pressure) |
| `spatial_analysis.py` | SPATIAL_ANALYSIS.md, spatial_*.png | Spatial distribution, zone occupation |
| `bonding_analysis.py` | BONDING_ANALYSIS.md, bonding_*.png | Cluster persistence, bond network |
| `burst_analysis.py` | BURST_ANALYSIS.md, filmstrip_*.png | Frame-by-frame movement tracking |
| `validate.py` | VALIDATION_REPORT.txt, validation.png | 15-20 automated checks incl. "waste creates pressure" (stable output dirs) |

Validate output uses stable dirs (`validate_neural_10000t/`) not timestamped. Compare uses run names (`comparison_<run1>_vs_<run2>/`).

---

## 9. Empirical Findings

Historical results (v5.0 baselines, v6.0 attenuation, v7.0 waste-too-mild, v7.1 zone migration) are archived in git history.

### Phase 4 Results: Bonds Solve Waste (v7.2)

**What changed:** Bond-waste equalization (waste shared across cluster). Metabolic efficiency (bonded cells produce 25% less waste). Syntrophy (waste→energy conversion). 4th bootstrap reaction (waste>0.3→move). Per-cell `cell_waste_exposure` field. 4-way death tracking (starvation/age/waste/predation × 3 zones). Cluster metrics (union-find). Division displacement tracking. Improved duplication-divergence mutation (prefers empty slots).

**What was attempted and reverted:** MAX_REACTIONS 16→24 caused population collapse at 30k (mutation load: 176 params × 2% = 3.52 mutations/generation vs 2.4 with 120 params). Reverted to 16 reactions. Expansion deferred to v7.3 with calibrated mutation rates.

### CRN 200k ticks — Bond-Waste Mechanics Drive Multicellularity

**Bonding increased from v7.1 baseline.** Bond-waste mechanics create selective advantage for clustering: bonded cells produce less waste and share waste exposure across the cluster.

**200k trajectory:**

| Tick | Pop | Bond% | MaxClust | Waste | Move | AvgX | Bright |
|------|-----|-------|----------|-------|------|------|--------|
| 0 | 1030 | 5.8% | 2 | 0.00 | 42% | 85 | 100% |
| 20k | 347 | 0.6% | 2 | 0.10 | 0.6% | 137 | 99% |
| 60k | 266 | 3.8% | 2 | 0.10 | 1.1% | 134 | 100% |
| 80k | 338 | 6.5% | 5 | 0.09 | 1.5% | 136 | 97% |
| 100k | 330 | 7.3% | 3 | 0.09 | 2.7% | 124 | 99% |
| 120k | 212 | **9.4%** | 4 | 0.08 | 0.9% | 116 | 98% |
| 140k | 191 | 7.9% | 3 | 0.07 | 1.6% | 105 | 99% |
| 180k | 86 | 5.8% | 3 | 0.03 | 2.3% | 96 | 95% |
| 200k | 170 | — | — | 0.02 | — | 86 | 96% |

**50k run showed stronger bonding:** Peak bond fraction 25.6%, max cluster size 11. Stochastic variation between runs is significant.

**Death cause breakdown (v7.2, new data):**
- Starvation: ~85% of all deaths (dominant cause)
- Age: ~15% of all deaths
- Waste: **0%** (waste at cells 0.09, below 0.3 threshold)
- Predation: <0.1%
- Zone deaths: 98%+ in bright zone (population stays concentrated)

**Key differences from v7.1:**

| Metric | v7.1 CRN (200k) | v7.2 CRN (200k) | Change |
|--------|-----------------|-----------------|--------|
| **Bonding (peak)** | 8.1% | **9.4%** (200k) / **25.6%** (50k) | +16-216% |
| **Max cluster size** | 2 | **5** | +150% |
| Bright zone (final) | 26% | **96%** | Reversed |
| Avg X (final) | 255 | 96 | No migration |
| Waste at cells | 0.16 | **0.09** | -44% |
| Movement (peak) | 0.0% | **2.7%** | New |
| Population (mean) | 274 | ~250 (50-100k) | -9% |
| Waste deaths | unknown | **0 (now measurable)** | New data |
| Respawn events | 2 | ~8 (after 150k) | More |

**The fundamental trade-off:** Bond-waste mechanics reduce waste damage for bonded cells (increasing bonding from 8→9-25%) but ALSO reduce overall waste pressure (waste at cells 0.09 vs 0.16), which eliminates the death gradient that drove zone migration in v7.1. **Bonds solve waste — perhaps too well.** Addressed in v7.3 by increasing waste production and adding environmental predation.

### Phase 5 Results: Dual Selective Pressure (v7.3)

**What changed:** WASTE_PRODUCTION_RATE 0.03→0.05. Environmental predation: gape-limited mortality sweep every 100 ticks (solo 1%, pair 0.5%, 2+ bonds immune). Sentinel -2 for env predation death classification. study.py fix: deaths_by_attack→deaths_by_predation.

**What was attempted and reverted:** (1) PREDATION_SOLO_KILL_PROB=0.02 caused population collapse to 31 at 30k (too aggressive before bonding evolves). Reduced to 0.01. (2) WASTE_TOXICITY_THRESHOLD 0.3→0.15 caused population extinction at 27k — syntrophy at 0.15 consumes waste at exactly the toxicity threshold, preventing damage while combined starvation+predation collapses population. Reverted. At ~200 population density on 500x500 grid, waste doesn't accumulate enough to cross any reasonable threshold.

### CRN 200k — Environmental Predation Drives Multicellularity

**Environmental predation creates strong clustering pressure.** Bonding peaked at 26.6% with max cluster size 19 — nearly 3x v7.2's peak (9.4%, cluster 5). Predation accounts for 9% of all deaths (5,049 total), creating continuous selective advantage for bonded cells.

**200k trajectory:**

| Tick | Pop | Bond% | MaxClust | Waste | Move | AvgX | Bright |
|------|-----|-------|----------|-------|------|------|--------|
| 0 | 1030 | 5.8% | 2 | 0.00 | 42% | 85 | 100% |
| 20k | 293 | 0.7% | 2 | 0.10 | 0.7% | 135 | 100% |
| 50k | 213 | 13.6% | 10 | 0.10 | 0.0% | 137 | 99% |
| 80k | 232 | 10.8% | 12 | 0.10 | 1.7% | 136 | 99% |
| 100k | 238 | 8.4% | 3 | 0.10 | 0.8% | 135 | 100% |
| 106k | 139 | **26.6%** | **19** | 0.09 | — | — | — |
| 120k | 131 | 4.6% | 2 | 0.09 | 4.6% | 137 | 99% |
| 150k | 117 | 12.0% | 4 | 0.09 | 0.0% | 140 | 98% |
| 180k | 108 | 5.6% | 3 | 0.02 | 0.0% | 110 | 96% |
| 200k | 86 | 7.0% | 4 | 0.02 | 0.0% | 101 | 93% |

**Death cause breakdown (v7.3, 200k):**
- Starvation: 47,664 (84.7% — still dominant)
- Predation: 5,049 (9.0% — significant new pressure from env predation)
- Age: 3,583 (6.4%)
- Waste: **0** (waste at cells ~0.10, below 0.3 threshold)

**Key differences from v7.2:**

| Metric | v7.2 CRN (200k) | v7.3 CRN (200k) | Change |
|--------|-----------------|-----------------|--------|
| **Bonding (peak)** | 9.4% | **26.6%** | +183% |
| **Max cluster size** | 5 | **19** | +280% |
| Predation deaths | 0 | **5,049 (9%)** | New pressure |
| Waste at cells | 0.09 | **0.10** | +11% |
| Waste deaths | 0 | 0 | Still below threshold |
| Bright zone (final) | 96% | **93%** | Slight decrease |
| Avg X (final) | 96 | 101 | No migration |
| Population (10-100k mean) | ~250 | **234** | -6% |
| Population (100-200k mean) | ~170 | **119** | -30% |
| Movement (peak) | 2.7% | **4.6%** | +70% |

**What worked:** Environmental predation creates direct selective advantage for bonding. Cells with 2+ bonds are immune to predation, driving bonding from 9.4% to 26.6% peak and cluster size from 5 to 19. This is the strongest multicellularity signal observed in CyberCell.

**What didn't change:** Waste at cells stayed ~0.10 (below 0.3 threshold) despite WASTE_PRODUCTION_RATE increase to 0.05. At ~200 population on 500x500, density is too low for waste to accumulate. Lowering WASTE_TOXICITY_THRESHOLD to 0.15 was attempted and caused extinction (syntrophy consumes waste at threshold, preventing damage while combined pressure collapses population). Zone migration still absent (93% bright). **Waste as selective pressure requires higher population density** — deferred until CRN expansion enables larger carrying capacity, or until specialized waste zones are added.

**Remaining issue:** Population declines in second half (234 mean first 100k, 119 mean second 100k) with respawn cycles after ~155k. Carrying capacity may be lower under predation pressure. Late-phase bonding (7% at 200k) suggests clusters don't persist long enough to dominate.

### Phase 6 Results: Expanded CRN Evolvability (v7.4)

**What changed:** MAX_REACTIONS 16→24 (8 silent slots, rate=0, wiring random). CRN_MUTATION_RATE_PERTURB 0.02→0.014 (calibrated by 120/176 to hold ~2.4 mutations/gen). CRN_GENOME_SIZE 120→176. Respawner updated to zero rate for reactions 16+ (matching init). study.py records_to_arrays fixed for missing keys when pop=0.

**What was attempted and reverted:** CRN_MUTATION_RATE_REWIRE 0.01→0.007 and CRN_MUTATION_RATE_DELETE 0.005→0.003 (scaled by 16/24) caused early extinction at 26k. Reverted — structural mutations on silent slots are no-ops, so effective load on active reactions is unchanged.

### CRN 200k — Expanded Reactions Improve Evolvability

**24-reaction CRN outperforms 16-reaction across all multicellularity metrics.** Bonding peaked at 28.7% (vs 26.6%), max cluster reached 33 cells (vs 19), and movement evolved to 38.2% (vs 4.6%). Population was healthier in the first 100k (mean 268 vs 234). Silent reaction slots activated incrementally — 16/24 reactions active by 192k.

**200k trajectory:**

| Tick | Pop | Bond% | MaxClust | Waste | Move | AvgX | Bright |
|------|-----|-------|----------|-------|------|------|--------|
| 0 | 1031 | 6.0% | 2 | 0.00 | 42% | 84 | 100% |
| 10k | 320 | 0.6% | 2 | 0.10 | 14.1% | 137 | 99% |
| 20k | 339 | 4.4% | 4 | 0.10 | 15.0% | 136 | 99% |
| 50k | 249 | 11.6% | 6 | 0.10 | 24.5% | 135 | 100% |
| 70k | 267 | — | **33** | — | — | — | — |
| 80k | 233 | 5.2% | 2 | 0.10 | **38.2%** | 135 | 100% |
| 100k | 261 | 9.2% | 2 | 0.10 | 24.9% | 136 | 99% |
| 111k | — | **28.7%** | 6 | — | — | — | — |
| 120k | 139 | 8.6% | 3 | 0.08 | 7.9% | 138 | 94% |
| 150k | 73 | 6.8% | 3 | 0.08 | 1.4% | 134 | 100% |
| 192k | 3 | 0.0% | 0 | 0.00 | 0.0% | — | — |

**Death cause breakdown (v7.4, 200k):**
- Starvation: 45,726 (84.1%)
- Predation: 5,112 (9.4%)
- Age: 3,557 (6.5%)
- Waste: 0 (0.0%)

**Key differences from v7.3:**

| Metric | v7.3 CRN (200k) | v7.4 CRN (200k) | Change |
|--------|-----------------|-----------------|--------|
| **Bonding (peak)** | 26.6% | **28.7%** | +8% |
| **Max cluster size** | 19 | **33** | +74% |
| **Movement (peak)** | 4.6% | **38.2%** | +730% |
| **Active reactions** | ~12/16 | **16/24** | Slots activating |
| **Mean pop 10-100k** | 234 | **268** | +15% |
| Mean pop 100-200k | 119 | 91 | -24% |
| Predation deaths | 5,049 (9.0%) | 5,112 (9.4%) | comparable |
| Extinction | survived (pop 86) | extinct at 192k | 8k earlier |

**What worked:** Expanded reaction capacity gives evolution more combinatorial space. Silent slots activate incrementally via mutation and duplication-divergence. Movement evolved to 38.2% — highest ever with CRN — suggesting extra reactions enable more complex sensorimotor circuits. Max cluster of 33 shows potential for larger multicellular structures.

**What didn't change:** Waste at cells ~0.10 (0 waste deaths). Zone migration absent (99-100% bright through 100k). Late-phase population decline (issue #20) led to extinction at 192k.

**Remaining issue:** Late-phase extinction continues (issue #20). Population declines from ~268 (10-100k mean) to extinction at 192k. v7.3 survived to 200k with pop 86 — the difference is likely stochastic variation rather than systematic.

### v7.1 CRN Results (Historical — Waste-Driven Zone Migration)

v7.1 achieved waste-driven zone migration (26% bright, 48% dim, 26% dark at 200k, avg X=255) through passive dispersal — the "landmark result." Mechanism: waste exceeded 0.3 toxicity in dense bright-zone clusters, killing interior cells and passively dispersing population rightward via daughter placement. Bonding declined under waste pressure (13.9%→8.1%). See git history for full v7.1 data.

### Phase 7 Results: Bond Signals + Respawn Fix (v7.5)

**What changed:** Hidden chemicals 10-11 emit bond signals when >0.5 (chem 10 → channels 0-1, chem 11 → channels 2-3). Respawned CRN cells get 4 bootstrap reactions matching init. RESPAWN_ENERGY=80 (was INITIAL_ENERGY=35). R1 bootstrap threshold kept at 0.3.

**What was attempted and reverted:** (1) R1 divide threshold 0.3→0.5 caused immediate extinction — cells couldn't divide fast enough to offset deaths. (2) R1 threshold 0.3→0.4 also collapsed population. The threshold change shifts initial dynamics too drastically for seed 42. Higher RESPAWN_ENERGY is the correct fix — targets respawned cells only.

### CRN 200k — Respawn Fix Prevents Late-Phase Extinction

**Population survived to 200k** (v7.4 went extinct at 192k). Respawned cells with bootstrap reactions and energy 80 establish viable populations in late phase (pop 6-25 from 160k-200k).

**200k trajectory:**

| Tick | Pop | Bond% | MaxClust | Waste | Move | Notes |
|------|-----|-------|----------|-------|------|-------|
| 10k | 234 | 5.1% | 4 | 0.096 | 0% | |
| 20k | 285 | 9.8% | 13 | 0.100 | 0% | Max cluster 39 at 18k |
| 40k | 259 | 14.3% | 11 | 0.099 | 0% | |
| 70k | 202 | 24.8% | 10 | 0.100 | 0% | |
| 100k | 207 | 14.5% | 8 | 0.099 | 0% | |
| 130k | 56 | 26.8% | 3 | 0.087 | 0% | |
| 139k | ~80 | **69.6%** | — | — | — | Peak bonding |
| 150k | 81 | 18.5% | 5 | 0.063 | 0% | |
| 170k | 25 | 8.0% | 2 | 0.006 | 0% | Respawn cycles |
| 200k | 22 | — | — | 0.001 | 0% | Alive (v7.4: extinct) |

**Death cause breakdown (v7.5, 200k):**
- Starvation: 47,836 (87.2%)
- Predation: 4,119 (7.5%)
- Age: 2,872 (5.2%)
- Waste: 0 (0.0%)

**Key differences from v7.4:**

| Metric | v7.4 CRN (200k) | v7.5 CRN (200k) | Change |
|--------|-----------------|-----------------|--------|
| **Bonding (peak)** | 28.7% | **69.6%** | +142% |
| **Max cluster size** | 33 | **39** | +18% |
| **Survived 200k** | No (extinct 192k) | **Yes (pop 22)** | Fixed |
| **Movement (peak)** | 38.2% | **0.0%** | Regression |
| Pop mean (10-100k) | 268 | 240 | -10% |
| Pop mean (150-200k) | extinct | **38** | Alive |

**What worked:** RESPAWN_ENERGY=80 gives respawned cells enough buffer to survive division + night cycle. Bootstrap reactions ensure immediate viability (photosynthesis, division, movement). Population never reaches 0 for sustained periods. Bond fraction peaked at 69.6% — highest ever — possibly due to reduced competition at lower population densities creating more space for stable clusters.

**What didn't change:** 0% waste deaths. Zone migration absent. Late-phase population decline still occurs (issue #20 mitigated, not solved). Movement at 0% — regression from v7.4's 38.2%, likely stochastic variation between seeds.

**Remaining concern:** 0% movement in this run. v7.4 achieved 38.2% movement with the same CRN architecture. This is likely seed-dependent stochastic variation, not a systematic regression from bond signal or respawn changes. Multiple seeds needed to confirm.

### Phase 8 Results: CTRNN Genome (v8.0)

**What changed:** New genome type `GENOME_TYPE = "ctrnn"` with 16-neuron CfC dynamics (Closed-form Continuous-time). 3-zone architecture: 8 sensory (pure blend), 4 hidden (CfC with evolved tau), 4 action (feedforward readout). 188 parameters. Sparse recurrence (4 connections per neuron). Bootstrap circuits match CRN (eat<-light, divide<-energy, move<-structure+waste). CRN bond signal reception added (issue #9b).

**Key design decisions during tuning:**
- **Sensory neurons: pure blend, no CfC.** CfC dynamics on sensory neurons created ~0.25 baseline regardless of input, making action readouts unable to distinguish light from dark. Pure blend (like CRN) gives faithful environment tracking.
- **Action neurons: feedforward readout, no CfC.** Persistent CfC state caused actions to fire constantly (70% movement). Reset-each-tick feedforward readout matches CRN's action chemical behavior.
- **Auxiliary action threshold 1.0 (not 0.5).** CfC hidden neurons reach steady-state ~0.85 from dynamics alone. Signal emission (0.1 energy/tick) at default state was the primary cause of mass starvation. Threshold 1.0 requires genuine recurrent drive.
- **Divide threshold at energy ~35.** Action bias -0.35 with w=2.5 creates sigmoid threshold: P=0.1% at E=25, 5% at E=35, 95% at E=45. Balances reproduction rate against night-survival energy buffer.
- **Action neuron init: zero non-bootstrap weights.** Random connections on action neurons created positive bias pushing all actions above threshold. Bootstrap-only init lets evolution add connections incrementally.

### CTRNN 867k — First Long Run

**CTRNN achieved zone migration and record multicellularity in its first long run.** Max cluster 77 cells (CRN record: 39). Zone migration from 100% bright to 15% bright. Hidden neurons differentiated multi-timescale dynamics (tau spread 0.005 to 0.776).

**867k trajectory:**

| Tick | Pop | Bond% | MaxClust | Move% | BrightPct | Energy |
|------|-----|-------|----------|-------|-----------|--------|
| 0 | 1000 | 0.0% | 0 | 6.1% | 100% | 34.9 |
| 50k | 775 | 10.3% | 6 | 10.3% | 83% | 111.8 |
| 100k | 554 | 10.6% | 5 | 4.2% | 87% | 130.4 |
| 200k | 326 | 18.1% | 15 | 9.5% | 62% | 61.2 |
| 350k | 265 | 6.0% | 4 | 9.4% | 31% | 78.2 |
| 500k | 722 | 8.3% | 4 | 18.1% | 15% | 79.9 |
| 650k | 770 | 11.2% | 11 | 16.4% | 28% | 79.1 |
| 766k | 608 | 21.9% | **77** | — | — | — |
| 867k | 189 | 2.1% | 2 | 4.2% | 33% | 85.8 |

**Hidden neuron tau differentiation (evolved):**

| Neuron | Init tau | Final tau | Change | Interpretation |
|--------|----------|-----------|--------|---------------|
| H0 (signal) | 1.50 | 0.93 | -0.56 | Faster — reactive signaling |
| H1 (bond) | 1.50 | 1.66 | +0.16 | Slower — stable bonding memory |
| H2 (bond sig 0-1) | 1.50 | 1.19 | -0.31 | Moderate — signal relay |
| H3 (bond sig 2-3) | 1.50 | 1.63 | +0.13 | Slower — signal persistence |

**Action bias evolution:**

| Action | Init bias | Final bias | Change | Interpretation |
|--------|-----------|------------|--------|---------------|
| Eat | +0.30 | +0.63 | +0.33 | Stronger eating drive (2x) |
| Move | +0.30 | +0.16 | -0.14 | More selective movement |
| Divide | -0.35 | -0.53 | -0.18 | More conservative division |
| Attack | -0.30 | -0.37 | -0.07 | Stayed suppressed |

**Death cause breakdown (867k):**
- Starvation: 541,970 (86.8%)
- Predation: 50,670 (8.1%)
- Age: 31,923 (5.1%)
- Waste: 0 (0.0%)

**Key differences from CRN v7.5 (200k):**

| Metric | CRN v7.5 (200k) | CTRNN v8.0 (867k) | Notes |
|--------|-----------------|-------------------|-------|
| **Max cluster** | 39 | **77** | +97% (record) |
| **Bonding peak** | 69.6% | **27.7%** | Lower peak but sustained |
| **Zone migration** | No (stayed bright) | **Yes (15% bright)** | Emergent |
| **Movement peak** | 0.0% | **29.0%** | Evolved navigation |
| **Tau differentiation** | N/A | **0.93-1.75** | Multi-timescale |
| Population (mean) | ~100 | **463** | Higher carrying capacity |
| Survived | 200k (barely) | **867k** | Robust |

**What the CTRNN enables that CRN cannot:** Multi-timescale memory through evolved time constants. Hidden neuron H0 evolved a fast tau (0.93) for reactive signaling while H3 evolved a slow tau (1.63) for persistent memory. This temporal differentiation is unique to CTRNN — CRN hidden chemicals all share the same decay rate (0.02/tick). The CfC dynamics also enable attractor-based computation where hidden neuron state represents a continuous "belief" about the environment, not just a chemical concentration.

### Neural Results (Historical — Hyper-Predation Collapse)

Neural genome evolves 30-43% attack by 100k ticks, creating respawn-massacre cycles. Population oscillates 50-200 indefinitely. Not actively developed — retained as comparison baseline only.

---

## 10. Known Issues — Priority Order

### Fixed in v5.0–v7.1
Issues 1-6, 7c, 11-14 fixed in prior versions. See git history for details.

### Fixed in v7.2
| # | Issue | Fix Applied |
|---|-------|-------------|
| 16 | CRN can't discover waste→move | 4th bootstrap reaction added (waste>0.3→move, C7→C13, rate=0.25). Provides 0-2.7% movement at 200k. |
| 17 | Bonding declines under waste | Bond-waste equalization (20% diff/tick/bond), metabolic efficiency (25% less waste with 2+ bonds), syntrophy (waste→energy above threshold). Bonding increased: peak 9.4% at 200k (was 8.1%), 25.6% peak at 50k. Max cluster size 5 (was 2). |
| — | Death causes unknown | 4-way death tracking: starvation/age/waste/predation × 3 zones. Revealed that 85% of deaths are starvation, 15% age, 0% waste in v7.2. |
| — | No cluster metrics | Union-find cluster stats: num_clusters, avg/max size, bonded_fraction. Division displacement tracking (avg daughter dx/dy). |

### Fixed in v7.3
| # | Issue | Fix Applied |
|---|-------|-------------|
| 8b | CRN predation zero | Environmental predation added: gape-limited mortality sweep every 100 ticks. Solo cells 2% kill/sweep, pair cells 0.5%, cells with 2+ bonds immune. Sentinel -2 in last_attacker for death classification. |
| 18 | Bond-waste reduces zone migration | WASTE_PRODUCTION_RATE 0.03→0.05 restores waste pressure on solo cells (SS ~0.35+ vs 0.26 for bonded). Combined with environmental predation, creates dual-pressure regime: waste rewards dispersal, predation rewards clustering. |
| — | study.py stale key | Fixed deaths_by_attack→deaths_by_predation in compute_rates(). |

### Fixed in v7.4
| # | Issue | Fix Applied |
|---|-------|-------------|
| 19 | 24-reaction expansion causes collapse | MAX_REACTIONS 16→24 with CRN_MUTATION_RATE_PERTURB 0.02→0.014 (scaled by 120/176 to hold ~2.4 mutations/gen). Reactions 16-23 initialized as silent slots (rate=0, wiring random). |

### Fixed in v7.5
| # | Issue | Fix Applied |
|---|-------|-------------|
| 9 | Bond signals unused | Hidden chemicals 10-11 emit bond signals when >0.5: chem 10 → channels 0,1; chem 11 → channels 2,3. Pipeline: CRN output → `process_bond_signal_output()` → `process_bond_signal_relay()` → partner's sensory inputs 18-33. **Note:** CRN sensory zone (8 inputs) does not yet map bond signal inputs — neural genome can read them, CRN reception requires future sensory expansion. |
| 20 | Respawned CRN cells non-viable | Respawner now gives CRN cells 4 bootstrap reactions matching init + RESPAWN_ENERGY=80 (2.3x INITIAL_ENERGY). Bootstrap reactions ensure photosynthesis/division; higher energy prevents starvation after division during night. R1 threshold change (0.3→0.4/0.5) was attempted but destabilized initial dynamics — reverted. |

### Fixed in v8.0
| # | Issue | Fix Applied |
|---|-------|-------------|
| 9b | CRN can't receive bond signals | Incoming bond signals blended into hidden chemicals 8-11 at rate CRN_BOND_SIGNAL_BLEND=0.15. Each hidden chemical h receives average of channel h across all bonded partners. Full emit→relay→receive loop now works for CRN and CTRNN. |

### Remaining Issues
| # | Issue | Root Cause | Fix Direction |
|---|-------|------------|---------------|
| 15 | Neural hyper-predation | Neural evolves 30-43% attack by 100k. | Not fixing — focusing on CRN. |
| 20 | Late-phase population decline | Population drops below 50 after ~100-130k in 200k runs. Respawn with bootstrap reactions + RESPAWN_ENERGY=80 prevents extinction (pop 6-25 at 160-200k) but doesn't restore carrying capacity. | May be inherent (genetic drift at Ne~200, mutation accumulation). Higher RESPAWN_COUNT or adaptive predation at low population could help. |

---

## 11. Future Roadmap (Do Not Implement Yet)

### Completed: Waste Tuning + Environmental Predation (v7.3)
WASTE_PRODUCTION_RATE 0.03→0.05. Environmental predation (solo 1%/sweep, pair 0.5%, 2+ bonds immune, every 100 ticks). Result: bonding 26.6% peak (vs 9.4%), max cluster 19 (vs 5), predation 9% of deaths. Waste increase had minimal effect (0.10 at cells, 0 waste deaths) — predation is the dominant clustering force.

### Completed: Expand CRN Evolvability (v7.4)
- MAX_REACTIONS 16→24, CRN_MUTATION_RATE_PERTURB 0.02→0.014 (120/176 scaling to hold ~2.4 mutations/gen).
- Reactions 16-23: silent slots (rate=0, wiring random). Activated by mutation or duplication-divergence.
- Duplication-divergence prefers empty slots — 8 new silent slots provide targets.
- Consider behavioral diversity pressure (reward novel behavioral profiles alongside fitness) — deferred.

### Then: Self-Regulating Predator Cells
- Evolve from environmental predation to cell-on-cell predation (Lotka-Volterra dynamics).
- Gape-limited predators that cannot consume clusters above 4 cells.
- Self-regulating predator population: reproduce on kills, die from starvation.
- Environmental predation (v7.3) can be phased out as evolved predation takes over.

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

### Stage 4 — Functional Multicellularity: PROGRESSING
v7.4 expanded CRN (24 reactions) achieves strongest multicellularity metrics: 28.7% peak bonding, max cluster 33 cells (vs 26.6%/19 in v7.3, 9.4%/5 in v7.2). Movement evolved to 38.2% — highest ever with CRN. 16/24 reaction slots active, with silent slots activating incrementally. **However**, clusters still don't persist — bonding fluctuates rather than monotonically increasing. Next: investigate cluster fragmentation (issue #9 bond signals, or longer bond reinforcement).

### Stage 5 — Sustained OEE: PROGRESSING
CRN reaction topology actively evolving at 200k with 16/24 slots in use. Population mean 268 (10-100k), declining to 91 (100-200k) with extinction at 192k (issue #20). Environmental predation adds 9.4% mortality pressure, keeping selection active. Silent reaction slots provide evolvable capacity for more complex circuits — movement reaching 38.2% suggests richer sensorimotor computation.

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

**Unit tests (39 tests, ~4s):** Run `python -m pytest tests/ -v` or just start the simulation — `main.py` runs preflight tests automatically (skip with `--skip-tests`). Tests use a shared `conftest.py` for session-scoped `ti.init()` — never call `ti.init()` in individual test files (causes field corruption). `test_bonding_waste.py` (7 tests: waste equalization, metabolic efficiency, syntrophy, death cause classification, environmental predation bond immunity, predation death sentinel classification, bond signal emission + reception). `test_ctrnn.py` (4 tests: evaluation produces outputs, bootstrap eat fires in light, mutation changes genome, hidden neurons retain memory).

**Validation harness (15-20 checks, 1-5 min):** Run `PYTHONPATH=. python validate.py --ticks 30000 --genome crn` (or `--genome ctrnn`) after significant changes. Tests population stability, bond dynamics, gradient noise, archipelago, 4-way death tracking, diversity, energy balance, cluster analysis, waste pressure, plus genome-specific checks (CRN: hidden zone activation, reaction diversification; CTRNN: hidden neuron activity, tau diversity, movement). CRN 30k: 21/21 passing (pop=524, 17/24 active reactions). Neural 10k: 15/15 passing.

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

1. **v8.0 state:** Three genome types: CRN (24 reactions, 176 params), CTRNN (16 CfC neurons, 188 params), Neural (baseline). Bond signal reception fixed (issue #9b). CRN and CTRNN both support full emit→relay→receive bond signal loop. Neural code retained as comparison baseline — never delete it.
2. **Immediate priority:** Compare CTRNN vs CRN on matched 200k runs. Fix lineage analysis for CTRNN genomes. Run multi-seed CTRNN validation for robustness.
3. **Secondary priority:** Self-regulating predator cells (Lotka-Volterra). Current environmental predation is a stepping stone.
4. Keep modules under 350 lines.
5. Run `python -m pytest tests/` after changes (39 tests, ~4s). Run `PYTHONPATH=. python validate.py --ticks 30000 --genome crn` (or `--genome ctrnn`) for validation. Run `PYTHONPATH=. python analysis/run_all.py` for full analysis. Preflight tests run automatically on `main.py` start.
6. Config values in `config.py` are the source of truth — this document may lag behind.
7. Design question: "Does this create pressure for complex behavior, or can a simple strategy still win?" v7.1 corollary: "Does this make clustering the *solution* to the problem, not its victim?" v7.3 corollary: "Do the dual pressures (waste + predation) create a fitness peak that requires both dispersal AND clustering to reach?"
