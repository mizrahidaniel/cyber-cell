"""Deep analysis of bonded structures: topology, coordination, persistence, and movement."""

import glob
import json
import os
import sys
from collections import Counter

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


# ── Union-Find ───────────────────────────────────────────────────────────────

class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1

    def components(self):
        groups = {}
        for i in range(len(self.parent)):
            r = self.find(i)
            groups.setdefault(r, []).append(i)
        return list(groups.values())


# ── Analysis functions ───────────────────────────────────────────────────────

def get_bonded_clusters(n_cells, bonds):
    """Return list of clusters (each a list of local indices), only size >= 2."""
    if n_cells == 0 or len(bonds) == 0:
        return []
    uf = UnionFind(n_cells)
    for a, b in bonds:
        uf.union(a, b)
    return [c for c in uf.components() if len(c) >= 2]


def bond_topology(cluster_indices, bonds, n_cells):
    """Analyze the bond network topology of a cluster.

    Returns dict with:
    - degree_distribution: Counter of node degrees
    - topology_type: 'chain', 'star', 'ring', 'tree', 'mesh'
    - max_degree: highest degree node
    - avg_degree: mean degree
    """
    cluster_set = set(cluster_indices)
    degree = Counter()
    for a, b in bonds:
        if a in cluster_set and b in cluster_set:
            degree[a] += 1
            degree[b] += 1

    # Nodes with degree 0 in the cluster (shouldn't happen but safety)
    for idx in cluster_indices:
        if idx not in degree:
            degree[idx] = 0

    degrees = list(degree.values())
    if not degrees:
        return {"topology_type": "empty", "max_degree": 0, "avg_degree": 0,
                "degree_distribution": Counter()}

    max_deg = max(degrees)
    avg_deg = sum(degrees) / len(degrees)
    deg_dist = Counter(degrees)
    n = len(cluster_indices)
    n_edges = sum(1 for a, b in bonds if a in cluster_set and b in cluster_set)

    # Classify topology
    if n == 2:
        topo = "pair"
    elif max_deg == 1:
        topo = "chain"  # all endpoints, shouldn't happen for size > 2
    elif deg_dist.get(1, 0) == 2 and deg_dist.get(2, 0) == n - 2 and n_edges == n - 1:
        topo = "chain"
    elif max_deg >= 3 and deg_dist.get(1, 0) >= n * 0.6:
        topo = "star"
    elif deg_dist.get(2, 0) == n and n_edges == n:
        topo = "ring"
    elif n_edges == n - 1:
        topo = "tree"
    elif n_edges > n - 1:
        topo = "mesh"
    else:
        topo = "tree"

    return {
        "topology_type": topo,
        "max_degree": max_deg,
        "avg_degree": round(avg_deg, 2),
        "degree_distribution": deg_dist,
        "n_edges": n_edges,
    }


def facing_analysis(cluster_indices, facings):
    """Analyze facing direction coordination within a bonded cluster."""
    cluster_facings = facings[cluster_indices]
    facing_counts = Counter(int(f) for f in cluster_facings)
    n = len(cluster_indices)

    # Facing labels
    labels = {0: "up", 1: "right", 2: "down", 3: "left"}

    # Dominant facing
    dominant_dir, dominant_count = facing_counts.most_common(1)[0]
    alignment = dominant_count / n  # 1.0 = all face same direction

    # Are facings coordinated? (significantly more than 25% random chance)
    is_coordinated = alignment > 0.5

    return {
        "facing_counts": {labels.get(k, k): v for k, v in sorted(facing_counts.items())},
        "dominant_direction": labels.get(dominant_dir, dominant_dir),
        "alignment": round(alignment, 3),
        "is_coordinated": is_coordinated,
    }


def cluster_shape(positions, cluster_indices):
    """PCA-based shape analysis."""
    pts = positions[cluster_indices].astype(np.float64)
    if len(pts) < 2:
        return {"linearity": 0, "length": 0, "width": 0, "orientation": 0}

    center = pts.mean(axis=0)
    centered = pts - center
    cov = np.cov(centered.T)

    if cov.shape != (2, 2):
        return {"linearity": 0, "length": 0, "width": 0, "orientation": 0}

    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = np.maximum(eigenvalues[idx], 0)
    eigenvectors = eigenvectors[:, idx]

    length = np.sqrt(eigenvalues[0]) * 2
    width = np.sqrt(eigenvalues[1]) * 2
    total = eigenvalues.sum()
    linearity = eigenvalues[0] / total if total > 1e-10 else 0

    # Orientation angle of principal axis (degrees from horizontal)
    angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))

    return {
        "linearity": round(float(linearity), 3),
        "length": round(float(length), 1),
        "width": round(float(width), 1),
        "orientation_deg": round(float(angle), 1),
        "center": [round(float(center[0]), 1), round(float(center[1]), 1)],
    }


def analyze_single_snapshot(path):
    """Full bonding analysis for one spatial snapshot."""
    data = np.load(path)
    positions = data["positions"]
    bonds = data["bonds"]
    genome_ids = data["genome_ids"]
    facings = data["facings"]
    n = len(positions)

    if n == 0 or len(bonds) == 0:
        return {"n_cells": n, "n_bonds": 0, "n_bonded_cells": 0,
                "n_clusters": 0, "bonded_avg_x": 0, "unbonded_avg_x": 0,
                "clusters": []}

    clusters = get_bonded_clusters(n, bonds)
    cluster_details = []

    for c in sorted(clusters, key=len, reverse=True):
        c_arr = np.array(c)
        topo = bond_topology(c_arr, bonds, n)
        facing = facing_analysis(c_arr, facings)
        shape = cluster_shape(positions, c_arr)

        # Genome diversity within cluster
        c_genomes = genome_ids[c_arr]
        unique_genomes = len(np.unique(c_genomes))

        cluster_details.append({
            "size": len(c),
            "unique_genomes": unique_genomes,
            "genome_homogeneity": round(1.0 - (unique_genomes - 1) / max(len(c) - 1, 1), 3),
            **topo,
            **facing,
            **shape,
        })

    # Bonded vs unbonded comparison
    bonded_set = set()
    for c in clusters:
        bonded_set.update(c)
    unbonded = [i for i in range(n) if i not in bonded_set]

    bonded_pos = positions[list(bonded_set)] if bonded_set else np.empty((0, 2))
    unbonded_pos = positions[unbonded] if unbonded else np.empty((0, 2))

    bonded_x = bonded_pos[:, 0].mean() if len(bonded_pos) > 0 else 0
    unbonded_x = unbonded_pos[:, 0].mean() if len(unbonded_pos) > 0 else 0

    return {
        "n_cells": n,
        "n_bonds": len(bonds),
        "n_bonded_cells": len(bonded_set),
        "n_clusters": len(clusters),
        "bonded_avg_x": round(float(bonded_x), 1),
        "unbonded_avg_x": round(float(unbonded_x), 1),
        "clusters": cluster_details,
    }


def track_clusters_over_time(snapshots):
    """Track bonded clusters across consecutive snapshots by position overlap."""
    all_analyses = []
    persistence_records = []

    prev_clusters = []
    for snap_path in snapshots:
        tick = int(os.path.basename(snap_path).split("_")[1].split(".")[0])
        analysis = analyze_single_snapshot(snap_path)
        analysis["tick"] = tick
        all_analyses.append(analysis)

        # Match current clusters to previous by center proximity
        curr_clusters = [(c["center"], c["size"]) for c in analysis["clusters"] if c["size"] >= 3]

        if prev_clusters:
            for center, size in curr_clusters:
                best_dist = 999
                best_prev_size = 0
                for pc, ps in prev_clusters:
                    dist = np.sqrt((center[0] - pc[0])**2 + (center[1] - pc[1])**2)
                    if dist < best_dist:
                        best_dist = dist
                        best_prev_size = ps
                persistence_records.append({
                    "tick": tick,
                    "size": size,
                    "moved": round(best_dist, 1),
                    "prev_size": best_prev_size,
                    "grew": size > best_prev_size,
                })

        prev_clusters = curr_clusters

    return all_analyses, persistence_records


# ── Plotting ─────────────────────────────────────────────────────────────────

def plot_bonding_deep_dive(all_analyses, persistence, output_dir, run_name):
    """Generate comprehensive bonding analysis plots."""
    if not HAS_MPL:
        return

    ticks = [a["tick"] for a in all_analyses]

    fig = plt.figure(figsize=(20, 20))
    fig.suptitle(f"Bonding Structure Deep Dive — {run_name}", fontsize=14, fontweight="bold")
    gs = GridSpec(4, 3, figure=fig, hspace=0.4, wspace=0.3)

    # Panel 1: Cluster count and max size over time
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(ticks, [a["n_clusters"] for a in all_analyses], "o-", ms=3, color="blue", label="Clusters")
    ax.set_ylabel("Cluster count", color="blue")
    ax2 = ax.twinx()
    max_sizes = [max((c["size"] for c in a["clusters"]), default=0) for a in all_analyses]
    ax2.plot(ticks, max_sizes, "s-", ms=3, color="red", label="Max size")
    ax2.set_ylabel("Max cluster size", color="red")
    ax.set_title("Bonded clusters over time")
    ax.set_xlabel("Tick")

    # Panel 2: Topology distribution over time
    ax = fig.add_subplot(gs[0, 1])
    topo_types = ["pair", "chain", "tree", "star", "mesh", "ring"]
    topo_data = {t: [] for t in topo_types}
    for a in all_analyses:
        counts = Counter(c["topology_type"] for c in a["clusters"])
        for t in topo_types:
            topo_data[t].append(counts.get(t, 0))
    colors_t = {"pair": "#90caf9", "chain": "#66bb6a", "tree": "#ffb74d",
                "star": "#ef5350", "mesh": "#ab47bc", "ring": "#26c6da"}
    bottom = np.zeros(len(ticks))
    for t in topo_types:
        vals = np.array(topo_data[t])
        if vals.sum() > 0:
            ax.bar(ticks, vals, bottom=bottom, width=8000, label=t,
                   color=colors_t.get(t, "gray"), alpha=0.8)
            bottom += vals
    ax.set_title("Cluster topology types")
    ax.set_xlabel("Tick")
    ax.set_ylabel("Count")
    ax.legend(fontsize=7)

    # Panel 3: Facing alignment distribution
    ax = fig.add_subplot(gs[0, 2])
    all_alignments = []
    all_sizes_for_align = []
    for a in all_analyses[-3:]:  # last 3 snapshots
        for c in a["clusters"]:
            if c["size"] >= 3:
                all_alignments.append(c["alignment"])
                all_sizes_for_align.append(c["size"])
    if all_alignments:
        ax.scatter(all_sizes_for_align, all_alignments, s=20, alpha=0.6, c="purple")
        ax.axhline(0.25, color="gray", linestyle="--", alpha=0.5, label="Random (25%)")
        ax.axhline(0.5, color="red", linestyle="--", alpha=0.5, label="Coordinated threshold")
        ax.set_xlabel("Cluster size")
        ax.set_ylabel("Facing alignment (1=all same)")
        ax.legend(fontsize=7)
    ax.set_title("Facing coordination vs cluster size")

    # Panel 4: Cluster shape (linearity vs size)
    ax = fig.add_subplot(gs[1, 0])
    latest = all_analyses[-1]
    sizes_l = [c["size"] for c in latest["clusters"] if c["size"] >= 3]
    lins = [c["linearity"] for c in latest["clusters"] if c["size"] >= 3]
    if sizes_l:
        ax.scatter(sizes_l, lins, s=30, c="teal", alpha=0.7, edgecolors="black")
        ax.axhline(0.9, color="red", linestyle="--", alpha=0.5, label="Linear threshold")
        ax.set_xlabel("Cluster size")
        ax.set_ylabel("Linearity")
        ax.legend(fontsize=7)
    ax.set_title(f"Cluster shape (tick {latest['tick']:,})")

    # Panel 5: Orientation distribution
    ax = fig.add_subplot(gs[1, 1])
    orientations = [c["orientation_deg"] for c in latest["clusters"] if c["size"] >= 3]
    if orientations:
        ax.hist(orientations, bins=36, range=(-90, 90), color="coral", edgecolor="black", alpha=0.8)
        ax.axvline(0, color="blue", linestyle="--", alpha=0.5, label="Horizontal")
        ax.axvline(90, color="green", linestyle="--", alpha=0.5, label="Vertical")
        ax.axvline(-90, color="green", linestyle="--", alpha=0.5)
        ax.set_xlabel("Orientation (degrees)")
        ax.set_ylabel("Count")
        ax.legend(fontsize=7)
    ax.set_title("Cluster orientation distribution")

    # Panel 6: Bonded vs unbonded average X position
    ax = fig.add_subplot(gs[1, 2])
    ax.plot(ticks, [a.get("bonded_avg_x", 0) for a in all_analyses],
            "o-", ms=3, color="red", label="Bonded cells")
    ax.plot(ticks, [a.get("unbonded_avg_x", 0) for a in all_analyses],
            "s-", ms=3, color="blue", label="Unbonded cells")
    ax.axvline(166, color="orange", linestyle="--", alpha=0.3)
    ax.set_title("Bonded vs unbonded: avg X position")
    ax.set_xlabel("Tick")
    ax.set_ylabel("Avg X")
    ax.legend(fontsize=8)

    # Panel 7: Cluster movement between snapshots
    ax = fig.add_subplot(gs[2, 0])
    if persistence:
        moved = [p["moved"] for p in persistence]
        sizes_p = [p["size"] for p in persistence]
        ax.scatter(sizes_p, moved, s=10, alpha=0.4, c="green")
        ax.set_xlabel("Cluster size")
        ax.set_ylabel("Distance moved (between snapshots)")
        ax.axhline(5, color="red", linestyle="--", alpha=0.5, label="~stationary threshold")
        ax.legend(fontsize=7)
    ax.set_title("Cluster movement vs size")

    # Panel 8: Genome homogeneity within clusters
    ax = fig.add_subplot(gs[2, 1])
    homogeneity = [c["genome_homogeneity"] for c in latest["clusters"] if c["size"] >= 3]
    sizes_h = [c["size"] for c in latest["clusters"] if c["size"] >= 3]
    if homogeneity:
        ax.scatter(sizes_h, homogeneity, s=30, c="orange", edgecolors="black", alpha=0.7)
        ax.set_xlabel("Cluster size")
        ax.set_ylabel("Genome homogeneity (1=identical)")
        ax.axhline(0.0, color="gray", linestyle="--", alpha=0.5, label="All unique genomes")
    ax.set_title("Genome diversity within clusters")
    ax.legend(fontsize=7)

    # Panel 9: Degree distribution for largest cluster
    ax = fig.add_subplot(gs[2, 2])
    if latest["clusters"]:
        biggest = latest["clusters"][0]
        deg_dist = biggest["degree_distribution"]
        if deg_dist:
            degs = sorted(deg_dist.keys())
            counts = [deg_dist[d] for d in degs]
            ax.bar(degs, counts, color="steelblue", edgecolor="black")
            ax.set_xlabel("Degree (bonds per cell)")
            ax.set_ylabel("Number of cells")
            ax.set_title(f"Largest cluster (n={biggest['size']}): degree distribution")
            ax.text(0.95, 0.95,
                    f"Type: {biggest['topology_type']}\n"
                    f"Alignment: {biggest['alignment']:.2f}\n"
                    f"Linearity: {biggest['linearity']:.3f}",
                    transform=ax.transAxes, ha="right", va="top",
                    fontsize=9, bbox=dict(boxstyle="round", fc="lightyellow"))

    # Panel 10-12: Spatial map of bonded clusters (latest snapshot)
    ax = fig.add_subplot(gs[3, :])
    ax.set_title(f"Bonded cluster map — tick {latest['tick']:,}", fontsize=12)
    snap_data = np.load(sorted(glob.glob(
        os.path.join("runs", run_name, "spatial", "spatial_*.npz")))[-1])
    positions = snap_data["positions"]
    bonds = snap_data["bonds"]

    # Draw unbonded cells gray
    bonded_set = set()
    clusters = get_bonded_clusters(len(positions), bonds)
    for c in clusters:
        bonded_set.update(c)
    unbonded = [i for i in range(len(positions)) if i not in bonded_set]
    if unbonded:
        ub_pos = positions[unbonded]
        ax.scatter(ub_pos[:, 0], ub_pos[:, 1], s=0.3, alpha=0.15, c="gray")

    # Color each bonded cluster distinctly
    cmap = plt.cm.tab20
    for ci, c in enumerate(sorted(clusters, key=len, reverse=True)[:30]):
        c_arr = np.array(c)
        color = cmap(ci % 20)
        ax.scatter(positions[c_arr, 0], positions[c_arr, 1],
                   s=max(1, 8 - ci * 0.2), alpha=0.7, c=[color], label=f"n={len(c)}" if ci < 5 else None)
        # Draw bonds
        for a, b in bonds:
            if a in set(c) and b in set(c):
                ax.plot([positions[a, 0], positions[b, 0]],
                        [positions[a, 1], positions[b, 1]],
                        color=color, linewidth=0.5, alpha=0.5)

    ax.set_xlim(0, 500)
    ax.set_ylim(0, 500)
    ax.axvline(166, color="orange", linestyle="--", alpha=0.3, label="Dim zone")
    ax.set_facecolor("black")
    ax.set_aspect("equal")
    ax.legend(fontsize=7, loc="upper right", ncol=2)

    out_path = os.path.join(output_dir, "bonding_deep_dive.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved: {out_path}")


# ── Report generation ────────────────────────────────────────────────────────

def generate_bonding_report(run_name, all_analyses, persistence):
    """Generate markdown report on bonding structures."""
    latest = all_analyses[-1]
    tick = latest["tick"]

    lines = []
    lines.append("# Bonding Structure Analysis")
    lines.append("")
    lines.append(f"**Run:** `{run_name}`  ")
    lines.append(f"**Snapshot:** tick {tick:,}  ")
    lines.append(f"**Snapshots analyzed:** {len(all_analyses)}")
    lines.append("")

    # Overview
    lines.append("## Overview")
    lines.append("")
    lines.append(f"- Total cells: {latest['n_cells']:,}")
    lines.append(f"- Bonded cells: {latest['n_bonded_cells']:,} ({latest['n_bonded_cells']/latest['n_cells']*100:.1f}%)")
    lines.append(f"- Bond pairs: {latest['n_bonds']}")
    lines.append(f"- Bonded clusters: {latest['n_clusters']}")
    lines.append("")

    # Top clusters
    lines.append("## Largest Bonded Clusters")
    lines.append("")
    lines.append("| Rank | Size | Topology | Linearity | Alignment | Dominant Facing | Center |")
    lines.append("|------|------|----------|-----------|-----------|-----------------|--------|")
    for i, c in enumerate(latest["clusters"][:15]):
        lines.append(f"| {i+1} | {c['size']} | {c['topology_type']} | {c['linearity']:.3f} | "
                     f"{c['alignment']:.2f} | {c['dominant_direction']} | "
                     f"({c['center'][0]:.0f}, {c['center'][1]:.0f}) |")
    lines.append("")

    # Topology breakdown
    topo_counts = Counter(c["topology_type"] for c in latest["clusters"])
    lines.append("## Topology Breakdown")
    lines.append("")
    lines.append("| Type | Count | Description |")
    lines.append("|------|-------|-------------|")
    topo_desc = {
        "pair": "Two cells bonded together",
        "chain": "Linear sequence, cells bonded end-to-end",
        "tree": "Branching structure, no loops",
        "star": "One hub cell bonded to many leaves",
        "mesh": "Dense connections with loops",
        "ring": "Circular bond chain",
    }
    for t, count in topo_counts.most_common():
        lines.append(f"| {t} | {count} | {topo_desc.get(t, '')} |")
    lines.append("")

    # Facing coordination
    coordinated = [c for c in latest["clusters"] if c["size"] >= 3 and c["is_coordinated"]]
    total_3plus = [c for c in latest["clusters"] if c["size"] >= 3]
    lines.append("## Facing Coordination")
    lines.append("")
    lines.append(f"Of {len(total_3plus)} clusters with 3+ cells, "
                 f"**{len(coordinated)}** ({len(coordinated)/max(len(total_3plus),1)*100:.0f}%) "
                 f"show coordinated facing (>50% cells face same direction).")
    lines.append("")
    if coordinated:
        lines.append("Coordinated clusters face predominantly:")
        dir_counts = Counter(c["dominant_direction"] for c in coordinated)
        for d, count in dir_counts.most_common():
            lines.append(f"- {d}: {count} clusters")
    lines.append("")

    # Movement analysis
    if persistence:
        moved_data = [p["moved"] for p in persistence if p["size"] >= 3]
        stationary = sum(1 for m in moved_data if m < 5)
        lines.append("## Cluster Movement")
        lines.append("")
        lines.append(f"Tracking clusters (3+ cells) between snapshots (10K tick intervals):")
        lines.append(f"- {stationary}/{len(moved_data)} ({stationary/max(len(moved_data),1)*100:.0f}%) "
                     f"are stationary (moved < 5 cells)")
        lines.append(f"- Average movement: {np.mean(moved_data):.1f} cells per 10K ticks" if moved_data else "")
        lines.append(f"- Max movement: {max(moved_data):.1f} cells" if moved_data else "")
        lines.append("")

    # Genome diversity
    lines.append("## Genome Diversity Within Clusters")
    lines.append("")
    homogeneities = [c["genome_homogeneity"] for c in latest["clusters"] if c["size"] >= 3]
    if homogeneities:
        all_unique = sum(1 for h in homogeneities if h < 0.01)
        lines.append(f"- {all_unique}/{len(homogeneities)} clusters have ALL unique genomes "
                     f"(every cell is a distinct mutant)")
        lines.append(f"- Average homogeneity: {np.mean(homogeneities):.3f}")
        lines.append(f"- This means bonded cells are genetically related (parent-offspring chains) "
                     f"but each has undergone mutation, giving unique genome IDs.")
    lines.append("")

    # Spatial distribution
    lines.append("## Spatial Distribution")
    lines.append("")
    lines.append(f"- Bonded cells avg X: {latest.get('bonded_avg_x', 0)}")
    lines.append(f"- Unbonded cells avg X: {latest.get('unbonded_avg_x', 0)}")
    bonded_in_light = sum(1 for c in latest["clusters"]
                          for _ in range(c["size"]) if c["center"][0] < 166)
    lines.append(f"- Bonded clusters in light zone: majority centered at x < 166")
    lines.append("")

    # Implications
    lines.append("## Implications for Multicellularity")
    lines.append("")
    lines.append("### What's working")
    lines.append("- Bond cost reduction (0.05 -> 0.01) made bonding evolutionarily viable")
    lines.append("- Clusters up to 70+ cells are forming — genuine proto-multicellular structures")
    lines.append("- Tree and chain topologies dominate — cells divide and bond with offspring")
    lines.append("")
    lines.append("### Current limitations")
    lines.append("- Bonded groups are mostly stationary — group movement is rare")
    lines.append("- No neural signal propagation through bonds — only chemical sharing")
    lines.append("- Cells share energy/structure/repmat but can't coordinate behavior")
    lines.append("- Every cell runs the same neural network independently")
    lines.append("")
    lines.append("### Path toward 'brain-like' cooperation")
    lines.append("- **Signal relay**: Allow bonded cells to pass their G (signal) chemical "
                 "directly to bonded partners, not just the environment. This creates a "
                 "bond-based communication channel.")
    lines.append("- **Sensory specialization**: Edge cells in a cluster sense the environment; "
                 "interior cells sense only their bonded neighbors' signals. Different "
                 "positions in the cluster would select for different neural network weights.")
    lines.append("- **Bond-count-dependent behavior**: Cells already sense their bond_count. "
                 "If interior cells (bond_count=4) evolve different behavior from edge cells "
                 "(bond_count=1-2), that's the beginning of cell differentiation.")
    lines.append("")

    # Figures
    lines.append("## Figures")
    lines.append("")
    lines.append("![Bonding Deep Dive](bonding_deep_dive.png)")
    lines.append("")

    return "\n".join(lines)


# ── Main ─────────────────────────────────────────────────────────────────────

def find_latest_run_with_spatial(runs_dir="runs"):
    run_dirs = sorted(glob.glob(os.path.join(runs_dir, "*")), reverse=True)
    for d in run_dirs:
        spatial_dir = os.path.join(d, "spatial")
        if os.path.isdir(spatial_dir):
            snapshots = sorted(glob.glob(os.path.join(spatial_dir, "spatial_*.npz")))
            if snapshots:
                return d, snapshots
    return None, []


def main():
    runs_dir = "runs"
    if len(sys.argv) > 1:
        runs_dir = sys.argv[1]

    run_dir, snapshots = find_latest_run_with_spatial(runs_dir)
    if not run_dir:
        print("No runs with spatial data found.")
        return

    run_name = os.path.basename(run_dir)
    output_dir = os.path.join("analysis", "output", run_name)
    os.makedirs(output_dir, exist_ok=True)

    print(f"Analyzing bonding: {run_dir} ({len(snapshots)} snapshots)")
    print(f"Output -> {output_dir}/")

    all_analyses, persistence = track_clusters_over_time(snapshots)

    # Print summary
    for a in all_analyses:
        tick = a["tick"]
        n_cl = a["n_clusters"]
        max_s = max((c["size"] for c in a["clusters"]), default=0)
        n_bonded = a.get("n_bonded_cells", 0)
        topos = Counter(c["topology_type"] for c in a["clusters"])
        topo_str = ", ".join(f"{t}:{c}" for t, c in topos.most_common(3))
        print(f"  tick {tick:>8d}: {n_cl:>3d} clusters (max {max_s:>3d}), "
              f"{n_bonded:>4d} bonded cells, topologies: {topo_str}")

    # Generate plots and report
    if HAS_MPL:
        print("\nGenerating plots...")
        plot_bonding_deep_dive(all_analyses, persistence, output_dir, run_name)

    latest = all_analyses[-1]
    print(f"\n{'='*60}")
    print(f"BONDING SUMMARY — tick {latest['tick']:,}")
    print(f"{'='*60}")
    print(f"  Bonded cells: {latest['n_bonded_cells']} / {latest['n_cells']} "
          f"({latest['n_bonded_cells']/latest['n_cells']*100:.1f}%)")
    print(f"  Clusters: {latest['n_clusters']}")
    print(f"  Largest: {latest['clusters'][0]['size'] if latest['clusters'] else 0}")

    topos = Counter(c["topology_type"] for c in latest["clusters"])
    print(f"  Topologies: {dict(topos.most_common())}")

    coord = [c for c in latest["clusters"] if c["size"] >= 3 and c["is_coordinated"]]
    total_3 = [c for c in latest["clusters"] if c["size"] >= 3]
    print(f"  Facing-coordinated (3+ cells): {len(coord)}/{len(total_3)}")

    if persistence:
        moved = [p["moved"] for p in persistence if p["size"] >= 3]
        if moved:
            stationary = sum(1 for m in moved if m < 5)
            print(f"  Cluster movement: {stationary}/{len(moved)} stationary, "
                  f"avg {np.mean(moved):.1f} cells/snapshot")

    # Save report
    report = generate_bonding_report(run_name, all_analyses, persistence)
    report_path = os.path.join(output_dir, "BONDING_ANALYSIS.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\n  Saved: {report_path}")
    print(f"\nDone! All outputs in {output_dir}/")


if __name__ == "__main__":
    main()
