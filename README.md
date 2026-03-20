# CyberCell

An artificial life simulation where simple digital organisms evolve through natural selection. Cells start with no intelligence — through mutation, reproduction, and environmental pressure, complex behaviors emerge from the bottom up.

Built with [Taichi Lang](https://github.com/taichi-dev/taichi) for GPU-accelerated parallel compute.

## Quick Start

```bash
pip install -r requirements.txt

# Run with visualization (auto-selects fastest backend on first run)
python main.py

# Run headless (for long evolution runs)
python main.py --headless --ticks 100000

# Use CRN genome instead of neural network
python main.py --genome crn --headless --ticks 50000

# Force a specific backend
python main.py --backend cuda    # Windows/NVIDIA
python main.py --backend metal   # macOS (Apple Silicon GPU)
python main.py --backend cpu     # Any platform

# Re-benchmark backends
python main.py --rebenchmark

# Resume from a checkpoint
python main.py --resume path/to/checkpoint.npz
```

### Visualization Controls

| Key | Action |
|-----|--------|
| `1` | Cell view (default) — cells colored by genome, brightness = energy |
| `2` | Structure (S) chemical heatmap (green) |
| `3` | Replication (R) chemical heatmap (red) |
| `4` | Signal (G) chemical heatmap (blue) |
| `5` | Membrane integrity heatmap (green=healthy, red=damaged) |
| `Space` | Pause / unpause |
| `Up` | Increase simulation speed (1x, 2x, 5x, 10x, 25x, 50x ticks/frame) |
| `Down` | Decrease simulation speed |
| `Esc` | Quit |

## How It Works

### The World

A 500×500 toroidal grid (edges wrap) divided into three light zones:

```
|<--- Light Zone --->|<--- Dim Zone --->|<--- Dark Zone --->|
|   x < 166          |  166 <= x < 333  |    x >= 333       |
|   Full sunlight     |  30% sunlight    |   No sunlight     |
|   (no deposits)     |  S + R deposits  |   S + R deposits  |
```

A **day/night cycle** (1000 ticks) modulates light sinusoidally. During night, all light drops to zero — cells must survive on reserves.

**Archipelago**: the grid is soft-partitioned into 4 quadrants with reduced chemical diffusion at boundaries. Each quadrant has ±30% parameter variance (light, photosynthesis rate). One cell migrates between random quadrants every 200 ticks, maintaining gene flow without homogenizing populations.

### Chemistry

Four chemicals drive the ecosystem:

| Chemical | Role | Diffusion | Decay |
|----------|------|-----------|-------|
| **E** (Energy) | Internal fuel. All actions cost E. Die at 0. | — | 0.02/tick |
| **S** (Structure) | Membrane repair material. | 0.01 | 0.001/tick |
| **R** (Replication) | Required for division (R ≥ 5). Scarce. | 0.03 | 0.001/tick |
| **G** (Signal) | Communication chemical. Produced by cells. | 0.3 | 0.05/tick |

**Resource deposits** generate S or R each tick (0.012 units), creating gradients that diffuse outward. Deposits are concentrated in dim/dark zones, forcing cells to leave the light zone for replication material. 20% of deposits relocate every 25k ticks.

### The CyberCell

Each cell occupies one grid position and has:

- **Internal state**: energy, structure, replication material, signal, membrane (0-100), age
- **A brain**: either a neural network (2,638 weights) or a chemical reaction network (120 parameters)
- **Bonds**: up to 4 connections to adjacent cells with strength, decay, and 4 signal channels per direction
- **34 sensory inputs** (18 base + 16 bond signals) and **14 action outputs** (10 base + 4 bond signals)

### Two Genome Types

**Neural Network** (`--genome neural`, default):
Feedforward 34→32→32→14. Actions fire when sigmoid output > 0.5. Mutation: weight perturbation (3%), reset (0.1%), knockout (0.05%).

**Chemical Reaction Network** (`--genome crn`):
16 internal chemicals in 3 zones (sensory/hidden/action), 16 reaction rules. Concentrations persist between ticks — the CRN has memory. Actions fire probabilistically via sigmoid: P(fire) = σ(30 × (chemical − 0.5)). Provides smooth evolutionary gradient unlike hard thresholds. See [CLAUDE.md](CLAUDE.md) Section 5 for architecture details.

### Actions (14 outputs)

| # | Action | Cost | Description |
|---|--------|------|-------------|
| 0 | move_forward | 0.1 E | Move one step in facing direction (unbonded only) |
| 1-2 | turn_left/right | 0.02 E | Rotate facing |
| 3 | eat | 0.02 E | Absorb S/R from environment |
| 4 | emit_signal | 0.1 E | Release G chemical |
| 5 | divide | 20 E + 5 R | Reproduce (auto-bonds parent↔daughter) |
| 6 | bond | 0.01 E | Bond with adjacent cell (mutual) |
| 7 | unbond | — | Break all bonds |
| 8 | attack | 0.3 E | Damage cell ahead (8 membrane damage) |
| 9 | repair | 0.1 E + 0.5 S | Repair membrane (+5 integrity) |
| 10-13 | bond_signals | — | 4-channel signals to bonded partners |

### Energy Balance

**Income:**
- Photosynthesis: `light × 0.45` E/tick (passive, always on)
- Eating: 0.3 E per S absorbed, 0.5 E per R absorbed
- Predation: 12% of victim's chemicals on kill (no flat bonus)

**Drains:**
- Basal metabolism: 0.08 E/tick
- Network evaluation: 0.01 E/tick
- Bond maintenance: 0.01 E/tick per bond (automatic)

**Death:** membrane reaches 0 (instant), energy at 0 (5 membrane damage/tick), age > 5000 (1 membrane/tick).

### Evolution

**Reproduction:** divide fires + E ≥ 20 + R ≥ 5 → find empty neighbor → parent pays costs → resources split 60/40 → daughter gets mutated genome copy → **auto-bond** created between parent and daughter (strength 0.1, breaks in ~50 ticks unless reinforced).

**Bonding:** near-permanent bonds (decay 0.001/tick, reinforced at 0.03/tick when both cells fire bond). Bonded cells automatically share resources (with 30% transfer loss). Clusters up to 22 cells form with mesh, chain, and star topologies.

## Current Results (v5.0, 50k-tick runs)

### CRN vs Neural Comparison

| Metric | CRN | Neural |
|--------|-----|--------|
| Population (50k) | **3,498** | 1,155 |
| Movement | 1.4% (sessile) | **29%** (chemotaxis) |
| Bonding | 7.3% | 10.7% |
| Shannon entropy | **8.07** (rising) | 7.05 (stable) |
| Lineage diversity | **43 root lines** | 13 root lines |
| Max cluster size | **22 cells** | 14 cells |
| Cluster topologies | mesh, chain, star | pairs, chains |

**CRN** achieves 3× neural population through metabolic efficiency (sessile "plant" strategy). **Neural** achieves behavioral complexity (active chemotaxis) at a carrying capacity cost. Both maintain high genome diversity with no plateau detected.

### Evolutionary Milestones

| Stage | Status | Evidence |
|-------|--------|----------|
| 1. Stable ecosystems | **Achieved** | Populations survive indefinitely, chemical cycling |
| 2. Behavioral evolution | **Achieved** | Neural: 29% chemotaxis. CRN: efficient sessile strategy |
| 3. Ecological complexity | **Partial** | Predation economics fixed, bonds stable, no arms races yet |
| 4. Proto-multicellularity | **In progress** | Clusters up to 22 cells, 94% facing coordination, mesh topologies |
| 5. Sustained OEE | **Promising** | CRN entropy rising, not plateaued at 50k ticks |

## Analysis

```bash
# Run full analysis suite on latest run
python analysis/run_all.py

# Analyze a specific run
python analysis/run_all.py runs/20260319_221948

# Compare two runs
python analysis/compare_runs.py runs/<crn_run> runs/<neural_run>

# Validation harness (after code changes)
python validate.py --genome crn --ticks 30000
python validate.py --genome neural --ticks 10000
```

Analysis outputs go to `analysis/output/<run_name>/`:

| Script | Output | Description |
|--------|--------|-------------|
| `study.py` | STUDY.md, evolution_report.png | Phase detection, rates, 6-panel dynamics |
| `crn_analysis.py` | CRN_ANALYSIS.md, crn_evolution.png | 9-panel CRN diagnostics |
| `lineage_analysis.py` | LINEAGE_ANALYSIS.md, lineage_tree.png | Phylogenetic trees, selective sweeps |
| `spatial_analysis.py` | SPATIAL_ANALYSIS.md, spatial_*.png | Spatial distribution, zone occupation |
| `bonding_analysis.py` | BONDING_ANALYSIS.md, bonding_*.png | Cluster topology, facing coordination |
| `burst_analysis.py` | BURST_ANALYSIS.md, filmstrip_*.png | Frame-by-frame movement |
| `compare_runs.py` | comparison_*.png, report.md | Side-by-side dynamics + OEE |
| `validate.py` | VALIDATION_REPORT.txt | 11-16 automated correctness checks |

## Project Structure

```
cybercell/
├── config.py                  # All tunable parameters (single source of truth)
├── main.py                    # Entry point (--headless, --ticks, --genome, --backend)
├── validate.py                # Backward-compat wrapper → analysis/validate.py
├── CLAUDE.md                  # Full project brief and architecture docs
├── RESEARCH.md                # Literature review informing design decisions
├── world/
│   ├── grid.py                # Light field, zones, day/night cycle
│   ├── chemistry.py           # Chemical fields, diffusion, deposits
│   └── archipelago.py         # Soft-wall quadrants, migration
├── cell/
│   ├── cell_state.py          # Cell state fields, grid occupancy, free-slot stack
│   ├── genome.py              # Neural network genome (34→32→32→14)
│   ├── crn_genome.py          # CRN genome (16 chemicals, 3 zones, 16 reactions)
│   ├── sensing.py             # 34 sensory inputs with gradient noise
│   ├── actions.py             # 14 actions, two-phase conflict resolution, auto-bonds
│   ├── bonding.py             # Bond formation, decay, lossy sharing, signal relay
│   └── lifecycle.py           # Photosynthesis, metabolism, death, kill rewards
├── simulation/
│   ├── engine.py              # Tick loop (dispatches neural or CRN)
│   ├── spawner.py             # Initial seeding and emergency respawn
│   ├── checkpoint.py          # Save/load full simulation state
│   └── env_api.py             # Runtime environment modification API
├── visualization/
│   └── renderer.py            # Taichi GUI rendering, overlays, stats
├── analysis/
│   ├── metrics.py             # Population stats, diversity, CRN snapshots
│   ├── logger.py              # Periodic snapshots to disk
│   ├── oee_metrics.py         # Open-ended evolution metrics (Bedau, MODES, MI)
│   ├── crn_analysis.py        # CRN deep diagnostics
│   ├── compare_runs.py        # Side-by-side run comparison
│   ├── validate.py            # Validation harness (11-16 checks)
│   ├── run_all.py             # Unified CLI for all analyses
│   ├── study.py               # Evolutionary dynamics study
│   ├── lineage_analysis.py    # Phylogenetic analysis
│   ├── spatial_analysis.py    # Spatial distribution
│   ├── bonding_analysis.py    # Bond network analysis
│   └── burst_analysis.py      # Frame-by-frame analysis
└── tests/
    ├── test_chemistry.py
    ├── test_energy.py
    ├── test_genome.py
    ├── test_lifecycle.py
    └── test_predation.py
```

## Performance

| Platform | Backend | Ticks/sec (~1K cells) |
|----------|---------|----------------------|
| Windows RTX 5080 | CUDA | ~900 |
| Windows RTX 5080 | CPU | ~450 |
| macOS M2 Pro | Metal | ~TBD |

Use `--backend auto` (default) to benchmark and cache the fastest backend. `--rebenchmark` to re-test.

## Configuration

All parameters in `config.py`. Key levers:

| Problem | Adjust |
|---------|--------|
| Population crashes | Increase `PHOTOSYNTHESIS_RATE` or decrease `BASAL_METABOLISM` |
| No reproduction | Increase `DEPOSIT_REPLENISH_RATE` or decrease `DIVIDE_R_COST` |
| Genomes converge | Increase mutation rates or `ISLAND_ENV_VARIANCE` |
| No movement evolves | Make deposits relocate more frequently |
| Bonding collapses | Decrease `BOND_DECAY_RATE` |

## Requirements

- Python 3.11+
- Taichi >= 1.7.0
- NumPy >= 1.24.0
- matplotlib (for analysis)
