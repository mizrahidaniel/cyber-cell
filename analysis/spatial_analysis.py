"""Detect spatial structures: lines, clusters, and bonded chains in cell populations."""

import json
import os
import sys
import glob

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    from matplotlib.colors import LogNorm
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


# ── Union-Find for connected components ──────────────────────────────────────

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


# ── Structure detection algorithms ───────────────────────────────────────────

def detect_bonded_clusters(positions, bonds):
    """Find connected components of bonded cells. Returns list of clusters."""
    n = len(positions)
    if n == 0 or len(bonds) == 0:
        return []
    uf = UnionFind(n)
    for a, b in bonds:
        uf.union(a, b)
    return [c for c in uf.components() if len(c) >= 2]


def cluster_linearity(positions, cluster_indices):
    """Measure how linear a cluster is using PCA eigenvalue ratio."""
    pts = positions[cluster_indices].astype(np.float64)
    if len(pts) < 2:
        return 0.0, 0.0, 0.0
    centered = pts - pts.mean(axis=0)
    cov = np.cov(centered.T)
    if cov.shape != (2, 2):
        return 0.0, 0.0, 0.0
    eigenvalues = np.sort(np.linalg.eigvalsh(cov))[::-1]
    eigenvalues = np.maximum(eigenvalues, 0)
    length = np.sqrt(eigenvalues[0]) * 2
    width = np.sqrt(eigenvalues[1]) * 2
    total = eigenvalues.sum()
    if total < 1e-10:
        return 0.0, 0.0, 0.0
    linearity = float(eigenvalues[0] / total)
    return linearity, float(length), float(width)


def detect_grid_runs(positions, grid_size=500):
    """Detect consecutive runs of occupied cells in rows and columns."""
    occupied = set(map(tuple, positions))
    all_runs = []

    for y in range(grid_size):
        run_len = 0
        run_start = 0
        for x in range(grid_size):
            if (x, y) in occupied:
                if run_len == 0:
                    run_start = x
                run_len += 1
            else:
                if run_len >= 3:
                    all_runs.append({"y": y, "x_start": run_start, "x": -1,
                                     "y_start": -1, "length": run_len,
                                     "direction": "horizontal"})
                run_len = 0
        if run_len >= 3:
            all_runs.append({"y": y, "x_start": run_start, "x": -1,
                             "y_start": -1, "length": run_len,
                             "direction": "horizontal"})

    for x in range(grid_size):
        run_len = 0
        run_start = 0
        for y in range(grid_size):
            if (x, y) in occupied:
                if run_len == 0:
                    run_start = y
                run_len += 1
            else:
                if run_len >= 3:
                    all_runs.append({"x": x, "y_start": run_start, "y": -1,
                                     "x_start": -1, "length": run_len,
                                     "direction": "vertical"})
                run_len = 0
        if run_len >= 3:
            all_runs.append({"x": x, "y_start": run_start, "y": -1,
                             "x_start": -1, "length": run_len,
                             "direction": "vertical"})

    all_runs.sort(key=lambda r: r["length"], reverse=True)
    return all_runs


def compute_zone_stats(positions, grid_size=500):
    """Break down cell distribution by light zone."""
    xs = positions[:, 0]
    n = len(xs)
    light = int(np.sum(xs < 166))
    dim = int(np.sum((xs >= 166) & (xs < 333)))
    dark = int(np.sum(xs >= 333))
    return {
        "light_count": light, "dim_count": dim, "dark_count": dark,
        "light_pct": light / n * 100 if n else 0,
        "dim_pct": dim / n * 100 if n else 0,
        "dark_pct": dark / n * 100 if n else 0,
    }


def compute_column_density_profile(positions, grid_size=500):
    """Get per-column cell counts and identify density peaks."""
    xs = positions[:, 0]
    counts = np.bincount(xs, minlength=grid_size)

    # Find peaks (columns with significantly more cells than neighbors)
    peaks = []
    for x in range(grid_size):
        c = counts[x]
        if c < 5:
            continue
        left = counts[(x - 1) % grid_size]
        right = counts[(x + 1) % grid_size]
        if c > left and c > right:
            peaks.append({"x": x, "count": int(c)})
    peaks.sort(key=lambda p: p["count"], reverse=True)

    return counts, peaks


def detect_adjacency_clusters(positions, grid_size=500):
    """Find connected components of adjacent cells (touching, not bonded).

    Uses flood-fill on the occupancy grid.
    """
    occupied = {}
    for i, (x, y) in enumerate(positions):
        occupied[(int(x), int(y))] = i

    visited = set()
    clusters = []

    for pos in occupied:
        if pos in visited:
            continue
        # BFS
        cluster = []
        queue = [pos]
        while queue:
            p = queue.pop()
            if p in visited:
                continue
            if p not in occupied:
                continue
            visited.add(p)
            cluster.append(occupied[p])
            x, y = p
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx = (x + dx) % grid_size
                ny = (y + dy) % grid_size
                if (nx, ny) not in visited and (nx, ny) in occupied:
                    queue.append((nx, ny))
        if len(cluster) >= 2:
            clusters.append(cluster)

    clusters.sort(key=len, reverse=True)
    return clusters


# ── Analyze a single spatial snapshot ────────────────────────────────────────

def analyze_snapshot(path):
    """Analyze a single spatial snapshot file."""
    data = np.load(path)
    positions = data["positions"]
    bonds = data["bonds"]

    n_cells = len(positions)
    results = {"n_cells": n_cells, "n_bonds": len(bonds)}
    if n_cells == 0:
        return results

    # Zone breakdown
    results.update(compute_zone_stats(positions))

    # Column density
    col_counts, col_peaks = compute_column_density_profile(positions)
    results["col_counts"] = col_counts
    results["top_5_columns"] = col_peaks[:5]
    results["max_cells_in_col"] = int(col_counts.max())
    results["densest_col_x"] = int(col_counts.argmax())

    row_counts = np.bincount(positions[:, 1], minlength=500)
    results["row_counts"] = row_counts
    results["max_cells_in_row"] = int(row_counts.max())
    results["densest_row_y"] = int(row_counts.argmax())

    # Bonded cluster analysis
    bonded_clusters = detect_bonded_clusters(positions, bonds)
    bonded_cells = sum(len(c) for c in bonded_clusters)
    results["n_bonded_clusters"] = len(bonded_clusters)
    results["n_bonded_cells"] = bonded_cells

    if bonded_clusters:
        sizes = [len(c) for c in bonded_clusters]
        results["max_bonded_cluster"] = max(sizes)
        results["avg_bonded_cluster"] = float(np.mean(sizes))
        linear_clusters = []
        for c in bonded_clusters:
            if len(c) >= 3:
                lin, length, width = cluster_linearity(positions, c)
                linear_clusters.append({
                    "size": len(c), "linearity": lin,
                    "length": length, "width": width,
                    "center": positions[c].mean(axis=0).tolist(),
                })
        results["linear_clusters"] = sorted(
            linear_clusters, key=lambda x: x["linearity"], reverse=True)
    else:
        results["max_bonded_cluster"] = 0
        results["linear_clusters"] = []

    # Adjacency cluster analysis (touching cells, regardless of bonds)
    adj_clusters = detect_adjacency_clusters(positions)
    results["n_adj_clusters"] = len(adj_clusters)
    if adj_clusters:
        adj_sizes = [len(c) for c in adj_clusters]
        results["max_adj_cluster"] = max(adj_sizes)
        results["adj_cluster_sizes"] = adj_sizes[:20]

        # Linearity of top adjacency clusters
        adj_linear = []
        for c in adj_clusters[:10]:
            if len(c) >= 5:
                lin, length, width = cluster_linearity(positions, c)
                adj_linear.append({
                    "size": len(c), "linearity": lin,
                    "length": length, "width": width,
                    "center": positions[c].mean(axis=0).tolist(),
                })
        results["adj_linear_clusters"] = adj_linear
    else:
        results["max_adj_cluster"] = 0
        results["adj_cluster_sizes"] = []
        results["adj_linear_clusters"] = []

    # Grid runs
    runs = detect_grid_runs(positions)
    results["n_runs_3plus"] = len(runs)
    results["n_runs_5plus"] = len([r for r in runs if r["length"] >= 5])
    results["n_runs_10plus"] = len([r for r in runs if r["length"] >= 10])
    results["max_run_length"] = runs[0]["length"] if runs else 0
    results["top_10_runs"] = runs[:10]

    return results


# ── Deep dive plots ──────────────────────────────────────────────────────────

def plot_deep_dive(snapshots, all_results, output_dir, run_name):
    """Generate comprehensive spatial analysis plots."""
    if not HAS_MPL:
        print("  matplotlib not available, skipping plots")
        return

    # ── Figure 1: Latest snapshot spatial map with annotations ──
    latest_data = np.load(snapshots[-1])
    positions = latest_data["positions"]
    bonds = latest_data["bonds"]
    latest = all_results[-1]

    fig = plt.figure(figsize=(20, 16))
    fig.suptitle(f"Spatial Structure Deep Dive — {run_name}\n"
                 f"Tick {latest['tick']:,} | {latest['n_cells']:,} cells",
                 fontsize=14, fontweight="bold")
    gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)

    # Panel 1: Full spatial map with zone overlays
    ax = fig.add_subplot(gs[0, 0])
    ax.set_title("Cell positions + zones", fontsize=11)
    ax.axvspan(0, 166, alpha=0.15, color="yellow", label="Light")
    ax.axvspan(166, 333, alpha=0.15, color="orange", label="Dim")
    ax.axvspan(333, 500, alpha=0.15, color="gray", label="Dark")
    if len(positions) > 0:
        ax.scatter(positions[:, 0], positions[:, 1], s=0.2, alpha=0.3, c="blue")
        for a, b in bonds:
            ax.plot([positions[a, 0], positions[b, 0]],
                    [positions[a, 1], positions[b, 1]],
                    "r-", linewidth=0.8, alpha=0.8)
    ax.set_xlim(0, 500)
    ax.set_ylim(0, 500)
    ax.set_aspect("equal")
    ax.legend(fontsize=7, loc="upper right")

    # Panel 2: 2D density heatmap
    ax = fig.add_subplot(gs[0, 1])
    ax.set_title("Cell density heatmap", fontsize=11)
    if len(positions) > 0:
        heatmap, xedges, yedges = np.histogram2d(
            positions[:, 0], positions[:, 1], bins=50, range=[[0, 500], [0, 500]])
        im = ax.imshow(heatmap.T, origin="lower", extent=[0, 500, 0, 500],
                       cmap="hot", interpolation="nearest",
                       norm=LogNorm(vmin=max(1, heatmap[heatmap > 0].min()),
                                    vmax=heatmap.max()))
        plt.colorbar(im, ax=ax, label="Cells per bin")
    ax.axvline(166, color="yellow", alpha=0.5, linestyle="--")
    ax.axvline(333, color="red", alpha=0.5, linestyle="--")
    ax.set_aspect("equal")

    # Panel 3: Column density profile (cells per x-column)
    ax = fig.add_subplot(gs[0, 2])
    ax.set_title("Column density profile", fontsize=11)
    col_counts = latest["col_counts"]
    ax.fill_between(range(500), col_counts, alpha=0.6, color="steelblue")
    ax.plot(range(500), col_counts, linewidth=0.5, color="navy")
    ax.axvline(166, color="orange", linestyle="--", alpha=0.7, label="Dim boundary")
    ax.axvline(333, color="red", linestyle="--", alpha=0.7, label="Dark boundary")
    if latest["top_5_columns"]:
        for p in latest["top_5_columns"][:3]:
            ax.annotate(f"x={p['x']}\n{p['count']} cells",
                        xy=(p["x"], p["count"]),
                        xytext=(p["x"] + 30, p["count"]),
                        arrowprops=dict(arrowstyle="->", color="red"),
                        fontsize=7, color="red")
    ax.set_xlabel("X column")
    ax.set_ylabel("Cells in column")
    ax.legend(fontsize=7)

    # Panel 4: Left-edge zoom (x=0-30) where the lines form
    ax = fig.add_subplot(gs[1, 0])
    ax.set_title("Left-edge zoom (x=0-30)", fontsize=11)
    if len(positions) > 0:
        mask = positions[:, 0] < 30
        edge_pos = positions[mask]
        ax.scatter(edge_pos[:, 0], edge_pos[:, 1], s=2, alpha=0.5, c="lime")
        # Draw bonds in this region
        for a, b in bonds:
            if positions[a, 0] < 30 or positions[b, 0] < 30:
                ax.plot([positions[a, 0], positions[b, 0]],
                        [positions[a, 1], positions[b, 1]],
                        "r-", linewidth=1, alpha=0.8)
        ax.set_facecolor("black")
    ax.set_xlim(0, 30)
    ax.set_ylim(0, 500)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_aspect("equal")

    # Panel 5: Adjacency cluster size distribution
    ax = fig.add_subplot(gs[1, 1])
    ax.set_title("Adjacency cluster sizes (touching cells)", fontsize=11)
    adj_sizes = latest.get("adj_cluster_sizes", [])
    if adj_sizes:
        max_show = min(max(adj_sizes) + 2, 100)
        ax.hist(adj_sizes, bins=range(2, max_show), color="teal", edgecolor="black", alpha=0.8)
        ax.set_xlabel("Cluster size")
        ax.set_ylabel("Count")
        ax.text(0.95, 0.95,
                f"Total: {latest['n_adj_clusters']}\n"
                f"Max: {latest['max_adj_cluster']}\n"
                f"Top 5: {adj_sizes[:5]}",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=8, bbox=dict(boxstyle="round", fc="lightyellow"))
    else:
        ax.text(0.5, 0.5, "No adjacency clusters", transform=ax.transAxes,
                ha="center", va="center")

    # Panel 6: Adjacency cluster linearity vs size
    ax = fig.add_subplot(gs[1, 2])
    ax.set_title("Adjacency cluster shape analysis", fontsize=11)
    adj_lin = latest.get("adj_linear_clusters", [])
    if adj_lin:
        sizes_a = [c["size"] for c in adj_lin]
        lins = [c["linearity"] for c in adj_lin]
        ax.scatter(sizes_a, lins, s=40, c="orange", edgecolors="black", alpha=0.8)
        for c in adj_lin[:3]:
            ax.annotate(f"({c['center'][0]:.0f},{c['center'][1]:.0f})",
                        xy=(c["size"], c["linearity"]), fontsize=7,
                        xytext=(5, 5), textcoords="offset points")
        ax.set_xlabel("Cluster size")
        ax.set_ylabel("Linearity (1=line, 0.5=circle)")
        ax.set_ylim(0.4, 1.05)
        ax.axhline(0.9, color="red", linestyle="--", alpha=0.5, label="Linear threshold")
        ax.legend(fontsize=7)
    else:
        ax.text(0.5, 0.5, "No large clusters", transform=ax.transAxes,
                ha="center", va="center")

    # Panel 7: Structure evolution — max run length + max adj cluster
    ax = fig.add_subplot(gs[2, 0])
    ax.set_title("Line formation over time", fontsize=11)
    ticks = [r["tick"] for r in all_results]
    ax.plot(ticks, [r["max_run_length"] for r in all_results],
            "o-", ms=3, color="green", label="Max consecutive run")
    ax.plot(ticks, [r.get("max_adj_cluster", 0) for r in all_results],
            "s-", ms=3, color="purple", label="Max adjacency cluster")
    ax.set_xlabel("Tick")
    ax.set_ylabel("Size")
    ax.legend(fontsize=8)

    # Panel 8: Zone distribution over time
    ax = fig.add_subplot(gs[2, 1])
    ax.set_title("Zone distribution over time", fontsize=11)
    ax.stackplot(ticks,
                 [r.get("light_pct", 0) for r in all_results],
                 [r.get("dim_pct", 0) for r in all_results],
                 [r.get("dark_pct", 0) for r in all_results],
                 labels=["Light", "Dim", "Dark"],
                 colors=["#fff176", "#ffb74d", "#90a4ae"], alpha=0.8)
    ax.set_xlabel("Tick")
    ax.set_ylabel("% of population")
    ax.set_ylim(0, 100)
    ax.legend(fontsize=8, loc="center right")

    # Panel 9: Bonding evolution
    ax = fig.add_subplot(gs[2, 2])
    ax.set_title("Bonding evolution", fontsize=11)
    ax.plot(ticks, [r.get("n_bonded_clusters", 0) for r in all_results],
            "o-", ms=3, color="red", label="Bonded clusters")
    ax2 = ax.twinx()
    ax2.plot(ticks, [r.get("max_bonded_cluster", 0) for r in all_results],
             "s-", ms=3, color="blue", label="Max cluster size")
    ax.set_xlabel("Tick")
    ax.set_ylabel("Cluster count", color="red")
    ax2.set_ylabel("Max size", color="blue")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=7)

    out_path = os.path.join(output_dir, "spatial_deep_dive.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved: {out_path}")

    # ── Figure 2: Structure evolution time series ──
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Structure Evolution — {run_name}", fontsize=14, fontweight="bold")

    ax = axes[0, 0]
    ax.plot(ticks, [r.get("n_bonded_clusters", 0) for r in all_results], "o-", ms=3)
    ax.set_ylabel("Bonded clusters")
    ax.set_title("Bonded cluster count")

    ax = axes[0, 1]
    ax.plot(ticks, [r.get("max_bonded_cluster", 0) for r in all_results],
            "o-", ms=3, color="red")
    ax.set_ylabel("Max cluster size")
    ax.set_title("Largest bonded cluster")

    ax = axes[1, 0]
    ax.plot(ticks, [r["max_run_length"] for r in all_results], "o-", ms=3, color="green")
    ax.set_ylabel("Max run length")
    ax.set_title("Longest consecutive cell line")

    ax = axes[1, 1]
    ax.plot(ticks, [r["n_runs_5plus"] for r in all_results],
            "o-", ms=3, color="purple", label="runs >= 5")
    ax.plot(ticks, [r["n_runs_10plus"] for r in all_results],
            "o-", ms=3, color="orange", label="runs >= 10")
    ax.set_ylabel("Count")
    ax.set_title("Cell line counts")
    ax.legend()

    for ax in axes.flat:
        ax.set_xlabel("Tick")
        ax.grid(True, alpha=0.3)

    evo_path = os.path.join(output_dir, "structure_evolution.png")
    plt.savefig(evo_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {evo_path}")


# ── Report generation ────────────────────────────────────────────────────────

def generate_spatial_report(run_name, all_results):
    """Generate a markdown report on spatial structures."""
    latest = all_results[-1]
    tick = latest["tick"]

    lines = []
    lines.append(f"# Spatial Structure Analysis")
    lines.append(f"")
    lines.append(f"**Run:** `{run_name}`  ")
    lines.append(f"**Snapshot:** tick {tick:,}  ")
    lines.append(f"**Spatial snapshots analyzed:** {len(all_results)}  ")
    lines.append(f"")

    # Zone distribution
    lines.append(f"## Population Distribution")
    lines.append(f"")
    lines.append(f"| Zone | Cells | % |")
    lines.append(f"|------|-------|---|")
    lines.append(f"| Light (x < 166) | {latest.get('light_count', 0):,} | {latest.get('light_pct', 0):.1f}% |")
    lines.append(f"| Dim (166-333) | {latest.get('dim_count', 0):,} | {latest.get('dim_pct', 0):.1f}% |")
    lines.append(f"| Dark (x >= 333) | {latest.get('dark_count', 0):,} | {latest.get('dark_pct', 0):.1f}% |")
    lines.append(f"")

    # Zone evolution
    if len(all_results) > 1:
        first = all_results[0]
        lines.append(f"Zone distribution evolved from "
                     f"{first.get('light_pct', 0):.0f}% / {first.get('dim_pct', 0):.0f}% / {first.get('dark_pct', 0):.0f}% "
                     f"(light/dim/dark) at tick {first['tick']:,} to "
                     f"{latest.get('light_pct', 0):.0f}% / {latest.get('dim_pct', 0):.0f}% / {latest.get('dark_pct', 0):.0f}% "
                     f"by tick {tick:,}.")
        lines.append(f"")

    # Density hotspots
    lines.append(f"## Density Hotspots")
    lines.append(f"")
    lines.append(f"- Densest column: x={latest['densest_col_x']} ({latest['max_cells_in_col']} cells)")
    lines.append(f"- Densest row: y={latest['densest_row_y']} ({latest['max_cells_in_row']} cells)")
    if latest.get("top_5_columns"):
        lines.append(f"- Top 5 columns by cell count: " +
                     ", ".join(f"x={p['x']} ({p['count']})" for p in latest["top_5_columns"]))
    lines.append(f"")

    # Adjacency clusters
    lines.append(f"## Adjacency Clusters (touching cells)")
    lines.append(f"")
    lines.append(f"Total clusters (2+ cells): {latest.get('n_adj_clusters', 0)}  ")
    lines.append(f"Largest cluster: {latest.get('max_adj_cluster', 0)} cells  ")
    lines.append(f"")

    adj_lin = latest.get("adj_linear_clusters", [])
    if adj_lin:
        lines.append(f"| Rank | Size | Linearity | Shape | Center (x,y) |")
        lines.append(f"|------|------|-----------|-------|--------------|")
        for i, cl in enumerate(adj_lin[:10]):
            shape = "LINE" if cl["linearity"] > 0.9 else \
                    "elongated" if cl["linearity"] > 0.7 else "blob"
            lines.append(f"| {i+1} | {cl['size']} | {cl['linearity']:.3f} | {shape} | "
                         f"({cl['center'][0]:.0f}, {cl['center'][1]:.0f}) |")
        lines.append(f"")

    # Consecutive runs
    lines.append(f"## Consecutive Cell Runs (axis-aligned lines)")
    lines.append(f"")
    lines.append(f"| Threshold | Count |")
    lines.append(f"|-----------|-------|")
    lines.append(f"| >= 3 cells | {latest['n_runs_3plus']} |")
    lines.append(f"| >= 5 cells | {latest['n_runs_5plus']} |")
    lines.append(f"| >= 10 cells | {latest['n_runs_10plus']} |")
    lines.append(f"| Max length | {latest['max_run_length']} |")
    lines.append(f"")

    top_runs = latest.get("top_10_runs", [])
    if top_runs:
        lines.append(f"Top 10 longest runs:")
        lines.append(f"")
        lines.append(f"| Rank | Length | Direction | Location |")
        lines.append(f"|------|--------|-----------|----------|")
        for i, run in enumerate(top_runs):
            if run["direction"] == "horizontal":
                loc = f"row y={run['y']}, x={run['x_start']}"
            else:
                loc = f"col x={run['x']}, y={run['y_start']}"
            lines.append(f"| {i+1} | {run['length']} | {run['direction']} | {loc} |")
        lines.append(f"")

    # Bonding
    lines.append(f"## Bonded Clusters")
    lines.append(f"")
    lines.append(f"- Total bond pairs: {latest.get('n_bonds', 0)}")
    lines.append(f"- Bonded clusters: {latest.get('n_bonded_clusters', 0)}")
    lines.append(f"- Max bonded cluster: {latest.get('max_bonded_cluster', 0)}")
    lines.append(f"")

    # Figures
    lines.append(f"## Figures")
    lines.append(f"")
    lines.append(f"![Spatial Deep Dive](spatial_deep_dive.png)")
    lines.append(f"")
    lines.append(f"![Structure Evolution](structure_evolution.png)")
    lines.append(f"")

    return "\n".join(lines)


# ── Main ─────────────────────────────────────────────────────────────────────

def find_latest_run_with_spatial(runs_dir="runs"):
    """Find the most recent run that has spatial snapshot data."""
    run_dirs = sorted(glob.glob(os.path.join(runs_dir, "*")), reverse=True)
    for d in run_dirs:
        spatial_dir = os.path.join(d, "spatial")
        if os.path.isdir(spatial_dir):
            snapshots = sorted(glob.glob(os.path.join(spatial_dir, "spatial_*.npz")))
            if snapshots:
                return d, snapshots
    return None, []


def main():
    arg = sys.argv[1] if len(sys.argv) > 1 else "runs"

    # If arg points directly to a run dir (has spatial/ inside), use it
    spatial_dir = os.path.join(arg, "spatial")
    if os.path.isdir(spatial_dir):
        run_dir = arg
        snapshots = sorted(glob.glob(os.path.join(spatial_dir, "spatial_*.npz")))
    else:
        run_dir, snapshots = find_latest_run_with_spatial(arg)

    if not run_dir or not snapshots:
        print("No runs with spatial data found.")
        print("Run a simulation first — spatial snapshots are saved every "
              "SPATIAL_SNAPSHOT_INTERVAL ticks.")
        return

    run_name = os.path.basename(run_dir)
    output_dir = os.path.join("analysis", "output", run_name)
    os.makedirs(output_dir, exist_ok=True)

    print(f"Analyzing: {run_dir} ({len(snapshots)} spatial snapshots)")
    print(f"Output -> {output_dir}/")

    # Analyze each snapshot
    all_results = []
    for snap_path in snapshots:
        tick = int(os.path.basename(snap_path).split("_")[1].split(".")[0])
        results = analyze_snapshot(snap_path)
        results["tick"] = tick
        all_results.append(results)

        max_run = results["max_run_length"]
        max_adj = results.get("max_adj_cluster", 0)
        n_bond = results.get("n_bonded_clusters", 0)
        max_bond = results.get("max_bonded_cluster", 0)
        print(f"  tick {tick:>8d}: {results['n_cells']:>5d} cells | "
              f"adj clusters: {results['n_adj_clusters']:>3d} (max {max_adj:>4d}) | "
              f"bonded: {n_bond:>3d} (max {max_bond:>2d}) | "
              f"max run: {max_run:>3d} | "
              f"light/dim/dark: {results['light_pct']:.0f}/{results['dim_pct']:.0f}/{results['dark_pct']:.0f}%")

    # Generate deep dive plots
    if HAS_MPL and snapshots:
        print("\nGenerating plots...")
        plot_deep_dive(snapshots, all_results, output_dir, run_name)

    # Print summary
    if all_results:
        latest = all_results[-1]
        print(f"\n{'='*70}")
        print(f"SPATIAL STRUCTURE ANALYSIS — tick {latest['tick']:,}")
        print(f"{'='*70}")

        print(f"\n  POPULATION DISTRIBUTION")
        print(f"    Total cells: {latest['n_cells']:,}")
        print(f"    Light zone: {latest['light_count']:,} ({latest['light_pct']:.1f}%)")
        print(f"    Dim zone:   {latest['dim_count']:,} ({latest['dim_pct']:.1f}%)")
        print(f"    Dark zone:  {latest['dark_count']:,} ({latest['dark_pct']:.1f}%)")

        print(f"\n  DENSITY HOTSPOTS")
        print(f"    Densest column: x={latest['densest_col_x']} "
              f"({latest['max_cells_in_col']} cells)")
        print(f"    Densest row:    y={latest['densest_row_y']} "
              f"({latest['max_cells_in_row']} cells)")
        if latest.get("top_5_columns"):
            print(f"    Top 5 columns: {latest['top_5_columns']}")

        print(f"\n  ADJACENCY CLUSTERS (touching cells, any direction)")
        print(f"    Total clusters (2+ cells): {latest['n_adj_clusters']}")
        print(f"    Largest cluster: {latest['max_adj_cluster']} cells")
        if latest.get("adj_cluster_sizes"):
            print(f"    Top 10 sizes: {latest['adj_cluster_sizes'][:10]}")
        if latest.get("adj_linear_clusters"):
            print(f"    Shape analysis (clusters 5+ cells):")
            for i, cl in enumerate(latest["adj_linear_clusters"][:5]):
                shape = "LINE" if cl["linearity"] > 0.9 else \
                        "elongated" if cl["linearity"] > 0.7 else "blob"
                print(f"      {i+1}. size={cl['size']}, linearity={cl['linearity']:.3f} "
                      f"({shape}), len={cl['length']:.1f}, "
                      f"width={cl['width']:.1f}, center=({cl['center'][0]:.0f},{cl['center'][1]:.0f})")

        print(f"\n  CONSECUTIVE CELL RUNS (axis-aligned)")
        print(f"    Runs >= 3:  {latest['n_runs_3plus']}")
        print(f"    Runs >= 5:  {latest['n_runs_5plus']}")
        print(f"    Runs >= 10: {latest['n_runs_10plus']}")
        print(f"    Max length: {latest['max_run_length']}")
        if latest.get("top_10_runs"):
            print(f"    Top 10 runs:")
            for i, run in enumerate(latest["top_10_runs"]):
                if run["direction"] == "horizontal":
                    loc = f"row y={run['y']}, x={run['x_start']}"
                else:
                    loc = f"col x={run['x']}, y={run['y_start']}"
                print(f"      {i+1}. len={run['length']:>3d} {run['direction']:<10s} at {loc}")

        print(f"\n  BONDED CLUSTERS")
        print(f"    Total bond pairs: {latest['n_bonds']}")
        print(f"    Bonded clusters: {latest['n_bonded_clusters']}")
        print(f"    Max bonded cluster: {latest['max_bonded_cluster']}")

    # Save results as JSON
    summary_path = os.path.join(output_dir, "spatial_summary.json")
    serializable = []
    for r in all_results:
        s = {k: v for k, v in r.items()
             if not isinstance(v, np.ndarray)}
        serializable.append(s)
    with open(summary_path, "w") as f:
        json.dump(serializable, f, indent=2, default=str)
    print(f"\n  Saved: {summary_path}")

    # Generate spatial analysis report
    if all_results:
        report_path = os.path.join(output_dir, "SPATIAL_ANALYSIS.md")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(generate_spatial_report(run_name, all_results))
        print(f"  Saved: {report_path}")

    print(f"\nDone! All outputs in {output_dir}/")


if __name__ == "__main__":
    main()
