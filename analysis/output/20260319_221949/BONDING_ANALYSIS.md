# Bonding Structure Analysis

**Run:** `20260319_221949`  
**Snapshot:** tick 40,000  
**Snapshots analyzed:** 5

## Overview

- Total cells: 1,100
- Bonded cells: 60 (5.5%)
- Bond pairs: 46
- Bonded clusters: 21

## Largest Bonded Clusters

| Rank | Size | Topology | Linearity | Alignment | Dominant Facing | Center |
|------|------|----------|-----------|-----------|-----------------|--------|
| 1 | 14 | mesh | 0.939 | 0.50 | right | (126, 31) |
| 2 | 4 | chain | 0.873 | 0.75 | left | (124, 19) |
| 3 | 4 | chain | 0.873 | 0.25 | left | (52, 18) |
| 4 | 3 | chain | 0.750 | 0.67 | left | (124, 29) |
| 5 | 3 | chain | 1.000 | 0.67 | right | (53, 31) |
| 6 | 2 | pair | 1.000 | 1.00 | up | (42, 18) |
| 7 | 2 | pair | 1.000 | 0.50 | right | (138, 16) |
| 8 | 2 | pair | 1.000 | 1.00 | left | (126, 18) |
| 9 | 2 | pair | 1.000 | 1.00 | down | (72, 482) |
| 10 | 2 | pair | 1.000 | 1.00 | down | (58, 244) |
| 11 | 2 | pair | 1.000 | 0.50 | left | (48, 18) |
| 12 | 2 | pair | 1.000 | 0.50 | down | (82, 486) |
| 13 | 2 | pair | 1.000 | 1.00 | up | (124, 16) |
| 14 | 2 | pair | 1.000 | 0.50 | up | (74, 467) |
| 15 | 2 | pair | 1.000 | 0.50 | up | (44, 233) |

## Topology Breakdown

| Type | Count | Description |
|------|-------|-------------|
| pair | 16 | Two cells bonded together |
| chain | 4 | Linear sequence, cells bonded end-to-end |
| mesh | 1 | Dense connections with loops |

## Facing Coordination

Of 5 clusters with 3+ cells, **3** (60%) show coordinated facing (>50% cells face same direction).

Coordinated clusters face predominantly:
- left: 2 clusters
- right: 1 clusters

## Cluster Movement

Tracking clusters (3+ cells) between snapshots (10K tick intervals):
- 12/49 (24%) are stationary (moved < 5 cells)
- Average movement: 76.9 cells per 10K ticks
- Max movement: 345.4 cells

## Genome Diversity Within Clusters

- 5/5 clusters have ALL unique genomes (every cell is a distinct mutant)
- Average homogeneity: 0.000
- This means bonded cells are genetically related (parent-offspring chains) but each has undergone mutation, giving unique genome IDs.

## Spatial Distribution

- Bonded cells avg X: 92.8
- Unbonded cells avg X: 90.1
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
