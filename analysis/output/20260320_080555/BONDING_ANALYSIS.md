# Bonding Structure Analysis

**Run:** `20260320_080555`  
**Snapshot:** tick 40,000  
**Snapshots analyzed:** 5

## Overview

- Total cells: 178
- Bonded cells: 9 (5.1%)
- Bond pairs: 7
- Bonded clusters: 4

## Largest Bonded Clusters

| Rank | Size | Topology | Linearity | Alignment | Dominant Facing | Center |
|------|------|----------|-----------|-----------|-----------------|--------|
| 1 | 3 | mesh | 1.000 | 0.67 | left | (75, 467) |
| 2 | 2 | pair | 1.000 | 1.00 | right | (138, 30) |
| 3 | 2 | pair | 1.000 | 1.00 | up | (126, 15) |
| 4 | 2 | pair | 1.000 | 0.50 | left | (75, 466) |

## Topology Breakdown

| Type | Count | Description |
|------|-------|-------------|
| pair | 3 | Two cells bonded together |
| mesh | 1 | Dense connections with loops |

## Facing Coordination

Of 1 clusters with 3+ cells, **1** (100%) show coordinated facing (>50% cells face same direction).

Coordinated clusters face predominantly:
- left: 1 clusters

## Cluster Movement

Tracking clusters (3+ cells) between snapshots (10K tick intervals):
- 1/9 (11%) are stationary (moved < 5 cells)
- Average movement: 239.3 cells per 10K ticks
- Max movement: 447.7 cells

## Genome Diversity Within Clusters

- 1/1 clusters have ALL unique genomes (every cell is a distinct mutant)
- Average homogeneity: 0.000
- This means bonded cells are genetically related (parent-offspring chains) but each has undergone mutation, giving unique genome IDs.

## Spatial Distribution

- Bonded cells avg X: 100.6
- Unbonded cells avg X: 99.1
- Bonded clusters in light zone: majority centered at x < 166

## Implications for Multicellularity

### What's working
- Bond cost reduction (0.05 -> 0.01) made bonding evolutionarily viable
- Clusters up to 70+ cells are forming — genuine proto-multicellular structures
- Tree and chain topologies dominate — cells divide and bond with offspring

### Current limitations
- Bonded groups are mostly stationary — group movement is rare
- No neural signal propagation through bonds — only chemical sharing
- Cells share energy/structure/repmat but can't coordinate behavior
- Every cell runs the same neural network independently

### Path toward 'brain-like' cooperation
- **Signal relay**: Allow bonded cells to pass their G (signal) chemical directly to bonded partners, not just the environment. This creates a bond-based communication channel.
- **Sensory specialization**: Edge cells in a cluster sense the environment; interior cells sense only their bonded neighbors' signals. Different positions in the cluster would select for different neural network weights.
- **Bond-count-dependent behavior**: Cells already sense their bond_count. If interior cells (bond_count=4) evolve different behavior from edge cells (bond_count=1-2), that's the beginning of cell differentiation.

## Figures

![Bonding Deep Dive](bonding_deep_dive.png)
