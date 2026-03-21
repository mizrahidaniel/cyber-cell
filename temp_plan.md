# v8.0 Next Steps Plan

## Context

v8.0 is implemented and validated. Three genome types: CRN (24 reactions, 176 params), CTRNN (16 CfC neurons, 188 params), Neural (baseline). CRN bond signal reception fixed (issue #9b). First CTRNN long run (867k ticks) shows record multicellularity and zone migration.

### What Was Done This Session

1. **CRN bond signal reception (issue #9b)**: Incoming bond signals blended into hidden chemicals 8-11 at rate CRN_BOND_SIGNAL_BLEND=0.15. ~10 lines in `crn_genome.py`. Verified by extended unit test.

2. **CTRNN genome type**: Complete implementation including:
   - `cell/ctrnn_genome.py` (~280 lines): CfC dynamics, feedforward action readout, mutations, init
   - `analysis/ctrnn_analysis.py` (~230 lines): 9-panel diagnostics
   - `tests/test_ctrnn.py` (~115 lines): 4 unit tests
   - Engine dispatch (3 points), checkpoint save/load, spawner respawn, logger, metrics, validate, run_all

3. **CTRNN bootstrap tuning** (multiple iterations to achieve viability):
   - Sensory neurons: pure blend (no CfC) — CfC created baseline activation masking external input
   - Action neurons: feedforward readout (no CfC) — persistent state caused 70% movement
   - Auxiliary action threshold: 1.0 (not 0.5) — CfC steady-state ~0.85 triggered signal emission draining 0.1 energy/tick
   - Divide threshold: energy ~35 (w=2.5, bias=-0.35) — conservative enough for night survival, fast enough for cohort replacement
   - Action neuron init: zero non-bootstrap weights — random connections pushed all actions above threshold
   - Move weights: 0.35 — ~18% movement for resource foraging

4. **Validation results**: CRN 21/21, CTRNN 20/20, 39/39 unit tests

5. **867k CTRNN run results** (user's CUDA run):
   - Population survived 867k ticks (min 161, mean 463, max 1308)
   - Max cluster 77 cells (record, CRN was 39)
   - Zone migration: 100% bright → 15% bright
   - Hidden tau differentiation: H0 1.50→0.93, H3 1.50→1.63 (multi-timescale)
   - Action biases evolved: eat +111%, move -47%, divide more conservative
   - Bonding peaked 27.7%, movement peaked 29%

### Key Architecture Decisions (for future reference)

**CTRNN 3-zone split:**
- Sensory (0-7): Pure blend with environment, NO CfC. Faithful tracking like CRN.
- Hidden (8-11): Full CfC dynamics with evolved tau. Memory and oscillation.
- Action (12-15): Feedforward readout each tick (like CRN action chemical reset). No persistence.

**Why not CfC everywhere:** CfC sensory neurons converge to ~0.25 baseline regardless of input (fast tau + recurrence dominates external input). CfC action neurons accumulate and never reset, causing constant firing. The 3-zone split gives the best of both: faithful sensing, rich memory, and controllable actions.

**Bootstrap circuit values:**
- Eat: w=1.0 → neuron 0 (light), action_bias=0.3. P≈99% in light, 0.2% in dark.
- Move: w=0.35 each → neurons 2,7 (structure, waste), action_bias=0.3. P≈18%.
- Divide: w=2.5 → neuron 1 (energy), action_bias=-0.35. Threshold at E=35.
- Attack: no bootstrap, action_bias=-0.3. Suppressed.

**Config params (in config.py):**
- CTRNN_GENOME_SIZE = 188
- CTRNN_PARAMS_PER_NEURON = 11 (tau, bias, A, 4 weights, 4 targets)
- CTRNN_EXTRA_PARAMS = 12 (8 input gains + 4 action biases)
- CTRNN_MUTATION_RATE_PERTURB = 0.013 (~2.4 mutations/gen)
- CTRNN_SENSORY_BLEND = 0.5 (match CRN)
- CTRNN_ACTION_GAIN = 30.0, CTRNN_ACTION_CENTER = 0.5 (match CRN sigmoid)
- CRN_BOND_SIGNAL_BLEND = 0.15 (new, for bond signal reception)

---

## Immediate Next Steps

### 1. Fix Lineage Analysis for CTRNN
The lineage_analysis.py failed on the 867k run with `IndexError: index 0 is out of bounds for axis 1 with size 0`. Likely tries to read CRN-specific genome weight columns. Needs CTRNN-aware genome weight handling.

### 2. Fix OEE Metrics for CTRNN
OEE MODES metrics (change, novelty, complexity, ecology) are all zero in the 867k run. The `oee_metrics.py` likely computes genome distance using `genome_weights` (neural) or `crn_weights` but not `ctrnn_weights`. Need to add CTRNN branch.

### 3. Matched CRN vs CTRNN Comparison Run
Run both genome types for 200k ticks with the same seed to compare:
- Population stability
- Bonding and cluster formation
- Zone migration
- Movement evolution
- Energy economy

Use `analysis/compare_runs.py` for side-by-side visualization.

### 4. Multi-Seed CTRNN Robustness Test
Run CTRNN with 3-5 different seeds for 100k ticks each. Check if the 867k results (zone migration, large clusters) are reproducible or seed-dependent. The user's first 20k run (different seed/backend) went extinct, suggesting sensitivity.

### 5. Study Report Update
The auto-generated STUDY.md incorrectly labels the genome as "neural network" — the study.py template assumes neural genome. Need to make study.py CTRNN-aware (or at least genome-type-aware in its text).

---

## Later Priorities (from CLAUDE.md roadmap)

### Self-Regulating Predator Cells
- Evolve from environmental predation to cell-on-cell predation (Lotka-Volterra)
- Gape-limited predators that cannot consume clusters above 4 cells
- Environmental predation (v7.3) can be phased out as evolved predation takes over
- **Caution:** Neural genome's hyper-predation collapse is the cautionary tale

### CRN Sensory Expansion
- CRN sensory zone maps 8 inputs but bond signal inputs (18-33) aren't in the sensory map
- CTRNN receives bond signals through hidden blending (same mechanism)
- CRN could benefit from expanded sensory zone or similar blending

### Richer Environment
- Mosaic of 4-6 qualitatively different zones
- POET-style adaptive difficulty
- Waste metabolism zone

---

## Files Modified in This Session

| File | Change |
|------|--------|
| `config.py` | Added CRN_BOND_SIGNAL_BLEND, CTRNN section (18 params), genome type "ctrnn" |
| `cell/crn_genome.py` | Bond signal reception (step 2c), imported cell_bonds + blend param |
| **`cell/ctrnn_genome.py`** | **New file** — CTRNN genome, CfC eval, mutations, init |
| **`analysis/ctrnn_analysis.py`** | **New file** — 9-panel CTRNN diagnostics |
| **`tests/test_ctrnn.py`** | **New file** — 4 CTRNN unit tests |
| `simulation/engine.py` | 3 CTRNN dispatch points (init, eval, mutation) |
| `simulation/checkpoint.py` | CTRNN save/load blocks |
| `simulation/spawner.py` | CTRNN respawn with bootstrap |
| `analysis/logger.py` | ctrnn_metrics.jsonl file handle |
| `analysis/metrics.py` | get_ctrnn_snapshot(), CTRNN in weight snapshot |
| `analysis/validate.py` | CTRNN validation checks, --genome ctrnn |
| `analysis/run_all.py` | CTRNN analysis dispatch |
| `main.py` | --genome ctrnn choice |
| `tests/test_bonding_waste.py` | Bond signal reception test |
| `tests/test_lifecycle.py` | Fixed stale action_outputs cross-test pollution |
| `CLAUDE.md` | v8.0 docs, Section 5b, architecture, results |
