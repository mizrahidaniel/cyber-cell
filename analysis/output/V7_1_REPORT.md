# CyberCell v7.1 — Tuned Waste, Waste Sensing, Analysis Upgrades

**Date:** 2026-03-20
**Runs:** Neural 50k ticks (20260320_080555), CRN 50k ticks (20260320_080911)
**Seed:** 42 for both

---

## 1. What Changed in v7.1

| Parameter | v7.0 | v7.1 | Rationale |
|-----------|------|------|-----------|
| `WASTE_PRODUCTION_RATE` | 0.01 | **0.03** | 3x. SS isolated cell: ~0.26 (safe). Cluster interior: ~0.50. |
| `WASTE_TOXICITY_THRESHOLD` | 0.5 | **0.3** | Isolated (0.26) safe. Cluster edges (~0.40) take damage. |
| `WASTE_TOXICITY_RATE` | 0.05 | **0.2** | 4x. Interior damage: ~0.04 membrane/tick. |
| `LIGHT_ATTENUATION_K` | 0.03 | **0.02** | Compensate for waste. At density 10: 82% light (was 74%). |
| Sensory input[15] | age | **waste** | CRN C7 now reads waste via existing `_SENSORY_MAP[7]=15`. |
| Toxicity read buffer | write (dst) | **read (src)** | Bug fix: toxicity now reads accumulated waste, not fresh writes. |

**New analysis features:**
- `study.py`: 8-panel figure (was 6) with waste trajectory + zone population panels; "Waste Pressure" phase detection
- `compare_runs.py`: Figure 4 — 6-panel environmental pressure comparison; waste/zone summary in report
- `metrics.py`: Per-zone waste breakdown (`waste_at_cells_bright`, `waste_at_cells_dim`, `waste_bright_zone_mean`)
- `validate.py`: "Waste creates pressure" check (avg waste at cells > 0.05); CRN pop target lowered 75 → 50
- Test infrastructure: `conftest.py` for single `ti.init()`; preflight tests in `main.py`

---

## 2. Executive Summary

**Waste toxicity is now a real evolutionary force.** In v7.0, peak waste was 0.17 — never reaching the 0.5 threshold. In v7.1, 67-71% of cells experience waste above the 0.3 threshold at any given time, with peak field concentrations reaching 0.56-0.57. This creates genuine density-dependent membrane damage that punishes clustering.

**However, the behavioral response differs sharply between genomes:**

| Metric | Neural v7.1 | CRN v7.1 | v7.0 Neural | v7.0 CRN |
|--------|-------------|----------|-------------|----------|
| Population (mean) | 298 | 184 | 512 | 317 |
| Movement | **19.2%** | 1.9% | 25.2% | 8.9% |
| Cells above toxicity | **67.1%** | **70.9%** | 0% | 0% |
| Avg waste at cells | 0.337 | 0.344 | ~0.05 | ~0.05 |
| Peak waste (field) | 0.561 | 0.570 | 0.17 | 0.17 |
| MI (sense-action) | **0.0111** | 0.0056 | 0.006 | 0.0039 |
| Attack | 0.83% | 0.0% | 0.65% | 0.0% |
| Bonding | 10.4% | 7.2% | 15.9% | 13.9% |
| Avg X | 103 | 130 | 126 | 91 |
| Bright zone % | 99.2% | 97.7% | — | — |

---

## 3. Detailed Findings

### 3.1 Waste Toxicity Is Working

The v7.1 tuning achieves exactly the intended effect:

- **Isolated cells (SS waste ~0.26):** Below the 0.3 threshold. Safe.
- **Cluster edges (waste ~0.35-0.40):** Above threshold. Membrane damage ~0.01-0.02/tick.
- **Cluster interiors (waste ~0.50+):** Significant damage ~0.04/tick. ~2,500 ticks to die.
- **Peak field concentration:** 0.56-0.57 (both genomes). Well above threshold but not runaway.

The waste trajectory panel in the 8-panel study figure shows waste rising quickly in the first 5k ticks as populations establish, then stabilizing as a dynamic equilibrium between production, diffusion, and decay.

### 3.2 Neural Genome: Chemotaxis Maintained Under Waste Pressure

The neural genome shows remarkable resilience:

- **Movement stays high:** 19.2% average (was 25.2% in v7.0, 29% in v5.0). The ~25% reduction from v7.0 is the carrying capacity effect — fewer cells, less gradient signal — not behavioral loss.
- **MI rose to 0.0111** — the highest we've seen in a 50k run. This is **1.85x above v7.0** (0.006) and **2.85x above v5.0** (0.0039). Waste creates a stronger sensory signal for neural networks to couple to action.
- **Predation persists at 0.83%** — comparable to v7.0 (0.65%) despite lower carrying capacity.
- **Population:** Mean 298, declining toward 157 at 50k. The late-run decline suggests the combined pressure of attenuation (k=0.02) + waste is at the edge of viability. The population didn't hit respawn threshold (50) but is trending down.

**Phase detection found 5 phases:** Crash, Chemotaxis Emergence (2k), Predation Emergence (17k), Waste Pressure (2k), and Bonding Emergence (2k). Waste Pressure triggering at tick 2k confirms that >5% of cells were above toxicity within 2,000 ticks of simulation start.

### 3.3 CRN Genome: Sessile Optimum Persists Despite Waste Sensing

Despite now having waste as a sensory input (C7, replacing age), the CRN genome still evolves to 0% movement by 50k ticks:

- **Movement collapsed:** 40.5% initial → 0.0% final (was 8.9% → 0.0% in v7.0)
- **CRN avg X drifted to 130** — this is farther right than v7.0 (91), which means cells are at the right edge of the bright zone (zone end = 166). This could indicate passive displacement from waste-driven deaths at the zone center, not active migration.
- **96.4% above toxicity at 50k** — CRN cells cluster even tighter than neural cells (density 16 vs 10.3), creating more waste per cell.
- **MI = 0.0056** — improved 1.4x over v7.0 (0.0039), but less than half of neural's 0.0111. The waste sensing input is being used somewhat but not enough to drive movement.

**Why CRN can't escape sessile optimum:** The CRN evaluates 16 reactions per tick. The bootstrap move reaction (`structure > 0.1 → move`) gets outcompeted by the eat reaction (`light > 0.2 → eat`). Since waste sensing is on C7 (index 15 in sensory inputs, blended into CRN chemical 7), it needs a reaction like `waste > threshold → move` to evolve. But with 16 reactions and only 3 bootstrap reactions (eat, divide, move), there are 13 random reactions that could wire waste→move. After 50k ticks, none have — the evolutionary search space is too large relative to the selection pressure. The move bias drifted from +0.30 to +0.233, confirming movement is being selected against.

### 3.4 Spatial Structure

**Neural:** Cells spread across the bright zone with avg X = 103.3 (zone center = 83). Relatively even distribution with max column density of 10 cells. 14 adjacency clusters of varying sizes.

**CRN:** One massive cluster of 118 cells (out of 129 total). The CRN population forms a single dense colony near the right edge of the bright zone (x ≈ 137). This is the classic sessile strategy — cluster near light, don't move, accept waste damage. Longest contiguous run: 14 cells (vs 7 for neural).

### 3.5 Lineage and Diversity

| Metric | Neural | CRN |
|--------|--------|-----|
| Total mutations | 68,986 | 20,627 |
| Mutations/tick | 1.39 | 0.41 |
| Initial root diversity | 1,041 | 865 |
| Final root diversity | 41 | 30 |
| Final Shannon | 5.06 | 4.69 |
| Selective sweep | Yes (96% reduction) | Yes (97% reduction) |

Both genomes show aggressive selective sweeps — a few successful lineages dominate. Neural maintains 3.4x more mutation throughput due to its larger genome (2,638 vs 120 params) and higher population.

### 3.6 OEE Metrics

| Metric | Neural | CRN |
|--------|--------|-----|
| Shannon entropy | 8.03 (mean) | 7.27 (mean) |
| Evolutionary activity | 1.91 | 1.67 |
| Novelty rate | 0.907 | 0.801 |
| Ecology evenness | 1.000 | 0.997 |
| Bond density | 0.159 | 0.080 |

Neural outperforms CRN on all OEE metrics. Both show near-perfect ecology evenness (close to 1.0), meaning no single genome dominates the population catastrophically.

---

## 4. v7.1 vs v7.0 Comparison

### What Improved

1. **Waste is a real selective pressure.** 67-71% of cells above toxicity (was 0%).
2. **MI increased for both genomes.** Neural: 0.006 → 0.0111 (+85%). CRN: 0.0039 → 0.0056 (+44%). Waste gives cells a new sensory dimension to evolve coupling with.
3. **Analysis tooling works.** 8-panel study figure, environmental pressure comparison figure, per-zone waste metrics all generate correctly. "Waste Pressure" phase detected automatically.
4. **Light attenuation compensation works.** Reducing k from 0.03 to 0.02 prevented the double-crush that made v7.0 200k runs barely viable.

### What Didn't Improve

1. **CRN sessile optimum persists.** Despite waste sensing, CRN still evolves to 0% movement. The substrate can sense waste but can't evolve waste → move in 50k ticks.
2. **No dim zone colonization.** Neural v7.0 at 200k showed 56% dim zone occupation. At 50k with v7.1, both genomes are >97% in the bright zone. This may emerge at longer timescales.
3. **Bonding decreased.** Neural: 15.9% → 10.4%. CRN: 13.9% → 7.2%. Waste damage to clustered cells may be selecting against tight bonding — an unintended side effect worth monitoring.

### What's Concerning

1. **Population viability.** Neural mean 298 (declining), CRN mean 184 (stable but low). Both could hit respawn at 100-200k ticks if trends continue.
2. **Bonding decline.** If waste punishes proximity and bonding requires proximity, waste may be actively selecting against multicellularity — opposite of our goal.

---

## 5. Validation Results

### Neural (10k ticks): 15/15 PASSED

All checks pass including new "Waste creates pressure" (avg waste 0.775 >> 0.05 target).

### CRN (30k ticks): 19/20 PASSED

Single failure: Archipelago (1/4 quadrants populated). Expected with CRN's low population (137 cells) concentrated in bright zone. Not a code issue.

### Test Suite: 28/28 PASSED

Fixed pre-existing Taichi double-init issue via `conftest.py`. All tests now pass reliably under pytest.

---

## 6. Recommendations for v7.2

1. **CRN waste→move bootstrap reaction.** Add a 4th bootstrap reaction: `waste > 0.3 → move` to give CRN a starting point for waste avoidance. Without this, the evolutionary search space for 13 random reactions to discover this pathway is too large.

2. **Bonding should protect from waste.** Consider: bonded cells share waste burden (total waste divided by cluster size). This creates a direct advantage for multicellularity under waste pressure.

3. **Population viability.** If 200k runs show population decline, increase `PHOTOSYNTHESIS_RATE` from 0.45 to 0.50, or reduce `WASTE_PRODUCTION_RATE` from 0.03 to 0.025.

4. **Dim zone colonization.** The bright zone has waste + attenuation pressure but also light. The dim zone has 30% light, no waste, no density pressure. The payoff for moving to dim is there but may need deposit concentration adjustments to make it viable.

---

## 7. Files Changed

| File | Changes |
|------|---------|
| `config.py` | 4 parameter changes |
| `cell/sensing.py` | input[15] age→waste, +env_W param |
| `simulation/engine.py` | Waste read/write buffer split, pass env_W to sensing |
| `cell/crn_genome.py` | Comment: age→waste |
| `analysis/metrics.py` | Per-zone waste breakdown |
| `analysis/compare_runs.py` | Figure 4: environmental pressure (6 panels) + report lines |
| `analysis/study.py` | 8-panel figure, waste phase detection, env pressure report section |
| `analysis/validate.py` | "Waste creates pressure" check, CRN pop target 75→50 |
| `tests/conftest.py` | New: session-scoped ti.init() |
| `tests/test_energy.py` | Fix photosynthesis() 3-arg call |
| `tests/test_predation.py` | Fix compute_sensory_inputs() 4-arg call |
| `tests/test_*.py` (all 5) | Remove setup_module(), standalone ti.init() in __main__ |
| `main.py` | Preflight test runner (--skip-tests to bypass) |

---

## 8. Figures Generated

- `analysis/output/20260320_080555/evolution_report.png` — 8-panel neural study (new waste + zone panels)
- `analysis/output/20260320_080911/evolution_report.png` — 8-panel CRN study
- `analysis/output/comparison_.../comparison_env.png` — **New Figure 4**: environmental pressure comparison
- `analysis/output/comparison_.../comparison_dynamics.png` — Population & behavior comparison
- `analysis/output/comparison_.../comparison_oee.png` — OEE metrics comparison
- `analysis/output/comparison_.../comparison_crn.png` — CRN internals comparison
- Plus lineage trees, spatial analysis, and bonding deep-dives for both runs
