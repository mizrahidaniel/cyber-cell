# CyberCell

An artificial life simulation where simple digital organisms evolve through natural selection. Cells start with no intelligence — through mutation, reproduction, and environmental pressure, complex behaviors emerge from the bottom up.

Built with [Taichi Lang](https://github.com/taichi-dev/taichi) for GPU-accelerated parallel compute.

## Quick Start

```bash
pip install -r requirements.txt

# Run with visualization (auto-selects fastest backend on first run)
python main.py

# Run headless (for long evolution runs)
python main.py --headless --ticks 100000 --log-interval 5000

# Force a specific backend
python main.py --backend metal   # macOS (Apple Silicon GPU)
python main.py --backend cuda    # Windows/NVIDIA
python main.py --backend cpu     # Any platform

# Re-benchmark backends (e.g. after hardware changes)
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
| `Space` | Pause / unpause |
| `Up` | Increase simulation speed (1x, 2x, 5x, 10x, 25x, 50x ticks/frame) |
| `Down` | Decrease simulation speed |
| `Esc` | Quit |

## How It Works

### The World

A 500x500 toroidal grid (edges wrap) divided into three zones:

```
|<--- Light Zone --->|<--- Dim Zone --->|<--- Dark Zone --->|
|   x < 166          |  166 <= x < 333  |    x >= 333       |
|   Full sunlight     |  30% sunlight    |   No sunlight     |
|   (no deposits)     |  S + R deposits  |   S + R deposits  |
```

A **day/night cycle** (1000 ticks per cycle) modulates light with a sinusoidal curve. During "night" (~500 ticks), all light drops to zero. Cells must accumulate enough energy during the day to survive the night.

### Chemistry

Four chemicals drive the simulation:

| Chemical | Role | In Environment? | Diffusion | Decay |
|----------|------|-----------------|-----------|-------|
| **E** (Energy) | Internal fuel. All actions cost E. Die at 0. | No (internal only) | — | 0.02/tick (flat) |
| **S** (Structure) | Membrane repair material. | Yes | Slow (0.01) | 0.001/tick |
| **R** (Replication) | Required for cell division (R >= 5). Scarce. | Yes | Slow (0.03) | 0.001/tick |
| **G** (Signal) | Communication chemical. Only produced by cells. | Yes | Fast (0.3) | 0.05/tick |

**Resource deposits** are fixed points on the grid that generate S or R each tick (0.015 units/tick). They create concentration gradients that diffuse outward and decay. Both S and R deposits are concentrated in the dim and dark zones (x >= 166), forcing cells to leave the light zone to find replication material. This spatial separation of energy (left) and materials (right) creates strong evolutionary pressure for chemotaxis.

### The CyberCell

Each cell occupies one grid position and has:

- **Internal state**: energy, structure, replication material, signal store, membrane integrity (0-100), age
- **A neural network brain**: 16 sensory inputs → 32 hidden (tanh) → 32 hidden (tanh) → 10 action outputs (sigmoid)
- **A genome**: 1,930 floating-point weights that define the neural network

#### Sensory Inputs (16)

| # | Input | Description |
|---|-------|-------------|
| 0 | light_here | Light intensity at current position (0-1) |
| 1 | energy_level | Own energy, normalized |
| 2 | structure_level | Own structure, normalized |
| 3 | repmat_level | Own replication material, normalized |
| 4 | membrane_integrity | Own membrane health, normalized |
| 5-6 | S_gradient_x/y | Direction of increasing S concentration |
| 7-8 | R_gradient_x/y | Direction of increasing R concentration |
| 9-10 | G_gradient_x/y | Direction of increasing signal concentration |
| 11 | cell_ahead | Is there a cell in front of me? (0/1) |
| 12 | cell_left | Cell to my left? (0/1) |
| 13 | cell_right | Cell to my right? (0/1) |
| 14 | bond_count | Number of active bonds (0-1, normalized) |
| 15 | age_normalized | Age / max_age |

#### Action Outputs (10)

| # | Action | Energy Cost | Threshold | Description |
|---|--------|-------------|-----------|-------------|
| 0 | move_forward | 0.1 | 0.5 | Move one step in facing direction |
| 1 | turn_left | 0.02 | 0.5 | Rotate counter-clockwise |
| 2 | turn_right | 0.02 | 0.5 | Rotate clockwise |
| 3 | eat | 0.02 | 0.5 | Absorb extra S/R from environment |
| 4 | emit_signal | 0.1 | 0.5 | Release G chemical |
| 5 | divide | 20.0 + 5R | 0.5 | Reproduce (requires E >= 20, R >= 5) |
| 6 | bond | — | 0.5 | Bond with adjacent cell (not yet implemented) |
| 7 | unbond | — | 0.5 | Break bonds (not yet implemented) |
| 8 | attack | 0.5 | 0.5 | Damage cell ahead (5 membrane damage) |
| 9 | repair | 0.1 + 0.5S | 0.5 | Repair own membrane (+5 integrity) |

Actions fire when the sigmoid output exceeds 0.5. Multiple actions can fire per tick.

Additionally, **photosynthesis** and **passive eating** (small chemical absorption) are always active — they are not gated by the neural network.

### Energy Balance

**Income:**
- Photosynthesis: `light_intensity * 0.5` E/tick (passive, always on)
- Eating chemicals: 0.3 E per S absorbed, 0.5 E per R absorbed

**Passive drains (every tick):**
- Basal metabolism: 0.05
- Internal energy decay: 0.02
- Network evaluation: 0.01
- Total passive: **0.08 E/tick**

A cell in the bright zone at peak daylight earns 0.5 E/tick, giving a net surplus of ~0.42 E/tick. Moving costs 0.1 E, making mobile strategies viable (~24% of surplus). During night, income drops to zero and cells must survive on reserves.

**Death conditions:**
- Membrane reaches 0 → instant death, internal chemicals spill into environment
- Energy reaches 0 → takes 5 membrane damage/tick until energy is restored
- Age exceeds 5000 → loses 1 membrane/tick (death within 100 ticks)

### Evolution

**Reproduction:** When a cell's divide output fires and it has E >= 20 and R >= 5:
1. An empty adjacent cell is found (or division fails)
2. Parent pays 20 E and 5 R
3. Remaining resources split: parent keeps 60%, daughter gets 40%
4. Daughter receives a (possibly mutated) copy of the parent's genome

**Mutation operators** (applied on division):
- **Weight perturbation** (p=0.03 per weight): add Gaussian noise (sigma=0.1)
- **Weight reset** (p=0.001 per weight): set to random value in [-1, 1]
- **Node knockout** (p=0.0005 per hidden node): zero all outgoing weights

**Conflict resolution:** When multiple cells try to move to (or divide into) the same position, the lowest cell index wins. This is implemented with a two-phase atomic claim system using `ti.atomic_min`.

### Initial Conditions

1,000 cells are placed randomly in the light zone, each with a unique genome. Weights are drawn from N(0, 0.01) so initial behavior is near-random, except for biased output layer biases:
- Divide (output 5): bias=+0.5 → fires reliably when resources are available
- Attack (output 8): bias=-1.0 → suppressed to prevent random killing
- Move (output 0): bias=0.0 → neutral (easily activated by a single mutation)

This gives cells a viable starting phenotype (photosynthesize, accumulate resources, divide) that evolution can modify.

## Project Structure

```
cybercell/
├── config.py                  # All tunable parameters (single source of truth)
├── main.py                    # Entry point (CLI args: --headless, --ticks, --backend, --log-interval)
├── world/
│   ├── grid.py                # Light field, zones, day/night cycle
│   └── chemistry.py           # Chemical fields, diffusion, decay, deposits
├── cell/
│   ├── cell_state.py          # Cell state fields + grid occupancy + free-slot stack
│   ├── genome.py              # Genome table, network evaluation, mutation
│   ├── sensing.py             # 16 sensory inputs (parallel kernel)
│   ├── actions.py             # 10 actions with two-phase conflict resolution
│   └── lifecycle.py           # Photosynthesis, passive eating, metabolism, death
├── simulation/
│   ├── engine.py              # Tick loop: environment → sense → think → act → resolve
│   ├── checkpoint.py          # Save/load full simulation state for resume & backend switching
│   └── spawner.py             # Initial cell seeding
├── visualization/
│   └── renderer.py            # ti.GUI rendering, overlays, stats
├── analysis/
│   ├── metrics.py             # Population stats, Shannon diversity, movement/chemotaxis metrics
│   ├── logger.py              # JSONL snapshots to runs/ directory
│   ├── study.py               # Evolutionary dynamics analysis, phase detection, report generation
│   └── output/                # Generated plots and writeups (from study.py)
└── tests/
    ├── test_chemistry.py      # Diffusion conservation, wrapping, decay
    ├── test_energy.py         # Photosynthesis, metabolism, death mechanics
    ├── test_genome.py         # Network outputs, weight layout, mutation validity
    └── test_lifecycle.py      # Division requirements, resource splitting
```

### Tick Sequence

Each simulation tick executes these steps in order:

1. `compute_light` — update light field for day/night cycle
2. `diffuse_all` — diffuse S, R, G chemicals across the grid
3. `replenish_deposits` — add chemicals at deposit locations
4. `photosynthesis` — passive energy gain from light
5. `eat_passive` — passive chemical absorption (small amount)
6. `compute_sensory_inputs` — read environment into 16-input vector
7. `evaluate_all_networks` — forward pass through each cell's neural net
8. `clear_intentions` — reset movement/division claim fields
9. `process_turns` — handle turning
10. `process_movement` (phase 1 + 2) — claim targets, resolve conflicts, move
11. `process_eat` — neural-net-gated eating (bonus absorption)
12. `process_emit_signal` — release G chemical
13. `process_repair` — spend S to fix membrane
14. `process_attack` — damage adjacent cell
15. `process_divide` (phase 1 + 2) — claim targets, resolve, create daughters
16. `process_mutations` — apply mutations to new genomes (GPU kernel)
17. `apply_metabolism` — deduct energy, age cells, apply membrane decay
18. `check_death` — kill depleted cells, spill chemicals
19. `swap_buffers` — toggle diffusion double-buffer
20. Periodic: genome garbage collection, metric snapshots

## Evolutionary Milestones

### Stage 1 (achieved): Stable Autotroph Populations
- Cells survive and reproduce in the light zone
- Chemical deposits cycle through the ecosystem
- Multi-generational populations with genome diversity

### Stage 2 (achieved): Chemotaxis
- Cells evolved directed movement toward chemical deposits (movement fraction: 4.7% to 63.6%)
- Population spread from light zone (avg x=86) into the dim zone (avg x=273)
- Multiple survival strategies coexist (Shannon diversity index: 9.38)
- See `analysis/output/STUDY.md` for the full writeup and `analysis/output/evolution_report.png` for plots

### Stage 3 (future): Predator-Prey Dynamics
- Heterotrophs emerge (cells that attack and consume others)
- Defensive behaviors evolve (fleeing, clustering)

### Stage 4 (future): Multicellularity
- Bonded cell clusters with differentiated behavior

## Analysis

After a simulation run, analyze the evolutionary dynamics:

```bash
# Analyze all runs (picks the longest automatically)
python analysis/study.py

# Analyze a specific run
python analysis/study.py runs/20260318_202818

# Compare two runs side by side
python analysis/study.py --compare runs/<run_a> runs/<run_b>
```

Outputs are saved to `analysis/output/`:
- `evolution_report.png` — 6-panel figure (population, movement, spatial, diversity, energy, phases)
- `comparison.png` — side-by-side comparison of two runs
- `STUDY.md` — full markdown writeup with detected evolutionary phases and key findings

Requires `matplotlib` (`pip install matplotlib`).

## Running Tests

```bash
python tests/test_chemistry.py
python tests/test_energy.py
python tests/test_genome.py
python tests/test_lifecycle.py
```

## Configuration

All parameters are in `config.py`. Key tuning levers if evolution stalls:

| Problem | Adjust |
|---------|--------|
| Population crashes | Increase `PHOTOSYNTHESIS_RATE` or decrease `BASAL_METABOLISM` |
| Population explodes | Decrease `PHOTOSYNTHESIS_RATE` or increase `MOVE_COST` |
| No reproduction | Increase `DEPOSIT_REPLENISH_RATE` or decrease `DIVIDE_R_COST` |
| Genomes converge | Increase `MUTATION_RATE_PERTURB` |
| Genomes dissolve into noise | Decrease `MUTATION_RATE_PERTURB` |

## Performance

The `--backend auto` mode (default) benchmarks available backends on first run and caches the result.

| Platform | Backend | Ticks/sec (~1K cells) |
|----------|---------|----------------------|
| Windows RTX 5080 | CUDA | ~900 |
| Windows RTX 5080 | CPU | ~450 |
| macOS M2 Pro | Metal | ~TBD |
| macOS M2 Pro | CPU | ~50 |

CUDA performance is roughly constant across population sizes (all mutation/evolution logic runs on GPU). Use `--rebenchmark` after significant config changes.

## Known Issues

- **macOS Metal + GUI**: Previously reported as crashing WindowServer ([taichi-dev/taichi#8775](https://github.com/taichi-dev/taichi/issues/8775)), but tested working on macOS 15 + Apple Silicon with Taichi 1.7.4. The `--backend auto` mode will benchmark Metal and use it if fastest.
- **Bonding** (actions 6-7) is parsed but not yet functional. The outputs are read but have no effect.

## Requirements

- Python 3.11+
- Taichi >= 1.7.0
- NumPy >= 1.24.0
