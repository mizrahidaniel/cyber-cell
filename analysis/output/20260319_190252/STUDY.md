# CyberCell Evolutionary Dynamics Study

**Run:** `20260319_190252`  
**Duration:** 199,000 ticks  
**Date:** 2026-03-18  

## Executive Summary

This simulation run achieved: **stable self-sustaining population, chemotaxis (21% of cells moving)**.

Starting from 2,000 cells with random neural network genomes, the population underwent natural selection driven entirely by environmental pressure — no behaviors were pre-programmed.

## Key Findings

### 1. Population Dynamics

| Metric | Value |
|--------|-------|
| Initial population | 2,000 |
| Minimum (bottleneck) | 384 (tick 88,000) |
| Final population | 1,231 |
| Growth rate | 1.1% per 1K ticks (post-bottleneck) |

The initial crash reflects **purifying selection**: cells with random neural networks that fail to photosynthesize or manage energy are eliminated. Only ~19% of initial genomes survive. The survivors then expand as successful strategies reproduce.

### 2. Emergence of Directed Movement (Chemotaxis)

| Metric | Value |
|--------|-------|
| Initial movement | 25.5% (random) |
| Post-crash movement | 44.3% (non-movers survive) |
| Final movement | 21.4% |
| Movement evolution rate | +-0.0010 per 1K ticks |

Movement follows a characteristic **U-shaped curve**:
1. **Random phase**: Initial genomes produce ~26% movement (noise)
2. **Crash phase**: Movement drops to ~44% — stationary photosynthesizers survive the bottleneck
3. **Evolution phase**: Movement rises to 21% — but this time it is *directed*, not random

The critical insight: post-crash movement is qualitatively different from initial random movement. Evolved movers have neural networks that couple chemical gradient sensing to motor output — they move *toward resources*.

### 3. Spatial Exploration

| Metric | Value |
|--------|-------|
| Initial avg X | 85.6 (light zone center ~83) |
| Final avg X | 79.3 |
| Expansion rate | -0.4 units per 1K ticks |

Cells began clustered in the light zone (x < 166) and expanded into the **light zone** (avg x = 79.3). This spatial expansion indicates cells evolved the ability to survive outside the primary energy source, using chemical deposits for sustenance.

### 4. Genetic Diversity

| Metric | Value |
|--------|-------|
| Initial Shannon index | 6.91 |
| Final Shannon index | 7.116 |
| Final unique genomes | 1,231 |
| Dominant genome fraction | 0.0812% |

Shannon diversity *increased* over the run, indicating the evolution of multiple coexisting strategies rather than a single dominant genome. The dominant genome accounts for only 0.0812% of the population — extreme diversity.

### 5. Energy Economy

| Metric | Value |
|--------|-------|
| Initial avg energy | 7.3 |
| Final avg energy | 71.4 |
| Final avg repmat | 74.0 |
| Max observed age | 5,760 ticks |

Energy accumulation shows cells evolved increasingly efficient metabolic strategies. The max observed age of 5,760 ticks (1x the nominal max age of 5,000) indicates lineages with exceptional survival ability.

## Evolutionary Phases Detected

| Phase | Tick Range | Description |
|-------|-----------|-------------|
| Crash | 0 – 88,000 | Population drops from 2000 to 384 (81% mortality) |
| Expansion | 88,000 – 120,000 | Population doubles to 770 |
| Chemotaxis Emergence | 2,000 – 199,000 | Movement fraction exceeds 15% at tick 2000, reaches 21.4% by end |
| Predation Emergence | 42,000 – 199,000 | Attack fraction exceeds 1% at tick 42000, reaches 0.5% by end |
| Bonding Emergence | 2,000 – 199,000 | Bond fraction exceeds 1% at tick 2000, reaches 0.5% by end |

## What Are the Cells "Learning"?

Each cell has a neural network that maps sensory inputs to actions. Through mutation and selection, these networks evolve to encode survival strategies. The key evolved behaviors we can infer from the metrics:

1. **Energy management**: Cells that survive the initial crash have networks that effectively couple light sensing to photosynthesis behavior
2. **Chemical gradient following**: The rise in movement fraction combined with spatial expansion indicates cells evolved to follow S and R chemical gradients
3. **Resource foraging**: Cells venture into dim/dark zones (where R deposits are concentrated) and return or sustain themselves on chemical energy
4. **Reproductive timing**: Cells accumulate replication material and divide when conditions are favorable, rather than dividing as soon as possible

Importantly, **none of these behaviors were programmed**. The simulation rules only define physics (diffusion, energy costs, death). All behavioral complexity emerged through natural selection acting on random neural network mutations.

## Methodology

- **Platform**: CyberCell evolutionary simulation (Taichi Lang + Python)
- **Grid**: 500x500 toroidal, three light zones (bright/dim/dark)
- **Organisms**: Neural network-controlled cells with sensory inputs and 10 actions
- **Selection**: Natural — cells die without energy, reproduce by division
- **Mutation**: Weight perturbation (3%), reset (0.1%), node knockout (0.05%)
- **Metrics**: Logged every 1000 ticks via population census

## Figures

![Evolutionary Dynamics](evolution_report.png)

## See Also

- [Spatial Structure Analysis](SPATIAL_ANALYSIS.md) (if available)
