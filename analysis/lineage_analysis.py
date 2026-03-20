"""
CyberCell Lineage Analysis
===========================
Analyzes lineage.jsonl and genome weight snapshots to reconstruct
phylogenetic trees, detect selective sweeps, and track weight evolution.

Usage:
    python analysis/lineage_analysis.py                      # analyze latest run
    python analysis/lineage_analysis.py runs/20260319_123456 # specific run
"""

import json
import os
import sys
import glob
from pathlib import Path
from collections import defaultdict

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

# Output action names for weight evolution plots
ACTION_NAMES = [
    "move_forward", "turn_left", "turn_right", "eat", "emit_signal",
    "divide", "bond", "unbond", "attack", "repair",
]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_lineage(run_dir: str) -> list[dict]:
    """Load lineage.jsonl events."""
    path = Path(run_dir) / "lineage.jsonl"
    if not path.exists():
        return []
    events = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                events.append(json.loads(line))
    return events


def load_metrics(run_dir: str) -> list[dict]:
    """Load metrics.jsonl records."""
    path = Path(run_dir) / "metrics.jsonl"
    if not path.exists():
        return []
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_genome_snapshots(run_dir: str) -> list[tuple[int, dict]]:
    """Load genome weight snapshots sorted by tick."""
    genomes_dir = Path(run_dir) / "genomes"
    if not genomes_dir.exists():
        return []
    files = sorted(genomes_dir.glob("genomes_*.npz"))
    snapshots = []
    for f in files:
        tick = int(f.stem.split("_")[1])
        data = dict(np.load(f))
        snapshots.append((tick, data))
    return snapshots


# ---------------------------------------------------------------------------
# Phylogenetic tree construction
# ---------------------------------------------------------------------------

def build_tree(events: list[dict]) -> tuple[dict, dict]:
    """Build parent->children adjacency and child->parent lookup from events."""
    children = defaultdict(list)  # parent_gid -> [(child_gid, tick)]
    parent_of = {}               # child_gid -> (parent_gid, tick)
    for e in events:
        p, c, t = e["parent"], e["child"], e["tick"]
        children[p].append((c, t))
        parent_of[c] = (p, t)
    return dict(children), parent_of


def trace_ancestors(gid: int, parent_of: dict, max_depth: int = 500) -> list[int]:
    """Trace a genome back to its root ancestor."""
    chain = [gid]
    current = gid
    for _ in range(max_depth):
        if current not in parent_of:
            break
        current = parent_of[current][0]
        chain.append(current)
    return list(reversed(chain))


def find_surviving_genomes(genome_snapshots: list[tuple[int, dict]]) -> set[int]:
    """Get genome IDs that are alive in the latest snapshot."""
    if not genome_snapshots:
        return set()
    _, latest = genome_snapshots[-1]
    return set(latest["genome_ids"].tolist())


def find_top_lineages(events: list[dict], surviving: set[int],
                      parent_of: dict, top_n: int = 20) -> list[list[int]]:
    """Find the top N longest lineage chains ending at surviving genomes."""
    chains = []
    for gid in surviving:
        chain = trace_ancestors(gid, parent_of)
        chains.append(chain)
    chains.sort(key=len, reverse=True)
    return chains[:top_n]


# ---------------------------------------------------------------------------
# Selective sweep detection
# ---------------------------------------------------------------------------

def compute_root_diversity(events: list[dict], parent_of: dict,
                           tick_window: int = 5000) -> tuple[list[int], list[int]]:
    """Count distinct root ancestors of active genomes at each time window.

    A drop in root diversity indicates a selective sweep.
    """
    if not events:
        return [], []

    max_tick = max(e["tick"] for e in events)
    # Group events by window
    all_genomes_by_window = defaultdict(set)
    for e in events:
        window = e["tick"] // tick_window
        all_genomes_by_window[window].add(e["child"])
        all_genomes_by_window[window].add(e["parent"])

    ticks = []
    root_counts = []
    for window in sorted(all_genomes_by_window.keys()):
        roots = set()
        for gid in all_genomes_by_window[window]:
            chain = trace_ancestors(gid, parent_of, max_depth=100)
            roots.add(chain[0])
        ticks.append(window * tick_window)
        root_counts.append(len(roots))

    return ticks, root_counts


# ---------------------------------------------------------------------------
# Weight evolution analysis
# ---------------------------------------------------------------------------

def extract_output_biases(genome_snapshots: list[tuple[int, dict]],
                          lineage_chain: list[int]) -> tuple[list[int], np.ndarray]:
    """Extract output layer biases for a lineage chain across snapshots.

    Returns (ticks, biases) where biases is shape (n_snapshots, 10).
    """
    # Import layout offsets
    try:
        from config import W3_END, B3_END
    except ImportError:
        W3_END, B3_END = 1920, 1930

    chain_set = set(lineage_chain)
    ticks = []
    biases = []

    for tick, snap in genome_snapshots:
        gids = snap["genome_ids"]
        weights = snap["weights"]
        # Find intersection of chain genomes and snapshot genomes
        for i, gid in enumerate(gids):
            if int(gid) in chain_set:
                # Extract output biases (last layer)
                bias = weights[i, W3_END:B3_END]
                ticks.append(tick)
                biases.append(bias)
                break  # one representative per snapshot

    if not biases:
        return [], np.empty((0, 10))
    return ticks, np.array(biases)


# ---------------------------------------------------------------------------
# Mutation rate analysis
# ---------------------------------------------------------------------------

def compute_mutation_rates(events: list[dict],
                           window: int = 1000) -> tuple[list[int], list[float], list[int]]:
    """Compute mutations per tick (windowed) and total mutations per window."""
    if not events:
        return [], [], []

    max_tick = max(e["tick"] for e in events)
    n_windows = max_tick // window + 1
    counts = [0] * n_windows

    for e in events:
        w = e["tick"] // window
        counts[w] += 1

    ticks = [w * window for w in range(n_windows)]
    rates = [c / window for c in counts]
    return ticks, rates, counts


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_lineage_analysis(events, parent_of, children, surviving,
                          genome_snapshots, metrics, output_dir, run_name):
    """Generate comprehensive lineage analysis figure."""
    if not HAS_MPL:
        print("  matplotlib not available, skipping plots")
        return

    fig = plt.figure(figsize=(18, 14))
    fig.suptitle(f"Lineage Analysis — {run_name}\n"
                 f"{len(events):,} mutation events",
                 fontsize=14, fontweight="bold")
    gs = GridSpec(3, 2, figure=fig, hspace=0.4, wspace=0.3)

    # Panel 1: Lineage depth distribution (tree as branching diagram)
    ax = fig.add_subplot(gs[0, 0])
    ax.set_title("Top lineage chains (depth)", fontsize=11)
    top_chains = find_top_lineages(events, surviving, parent_of, top_n=15)
    if top_chains:
        for i, chain in enumerate(top_chains[:15]):
            birth_ticks = []
            for gid in chain:
                if gid in parent_of:
                    birth_ticks.append(parent_of[gid][1])
                else:
                    birth_ticks.append(0)
            ax.plot(birth_ticks, [i] * len(birth_ticks), ".-", ms=2,
                    linewidth=0.8, alpha=0.7)
        ax.set_xlabel("Tick (birth of genome)")
        ax.set_ylabel("Lineage rank (by depth)")
        ax.set_yticks(range(min(15, len(top_chains))))
        ax.set_yticklabels([f"depth={len(c)}" for c in top_chains[:15]], fontsize=7)
    else:
        ax.text(0.5, 0.5, "No lineage data", transform=ax.transAxes,
                ha="center", va="center")
    ax.grid(True, alpha=0.3)

    # Panel 2: Selective sweeps (root diversity over time)
    ax = fig.add_subplot(gs[0, 1])
    ax.set_title("Root ancestor diversity (selective sweeps)", fontsize=11)
    sweep_ticks, root_counts = compute_root_diversity(events, parent_of)
    if sweep_ticks:
        ax.plot([t/1000 for t in sweep_ticks], root_counts, "-", color="#e65100",
                linewidth=1.5)
        ax.set_xlabel("Ticks (thousands)")
        ax.set_ylabel("Distinct root ancestors")
        ax.fill_between([t/1000 for t in sweep_ticks], root_counts, alpha=0.2,
                        color="#e65100")
    else:
        ax.text(0.5, 0.5, "No lineage data", transform=ax.transAxes,
                ha="center", va="center")
    ax.grid(True, alpha=0.3)

    # Panel 3: Mutation rate over time
    ax = fig.add_subplot(gs[1, 0])
    ax.set_title("Mutation rate over time", fontsize=11)
    mut_ticks, mut_rates, mut_counts = compute_mutation_rates(events)
    if mut_ticks:
        ax.plot([t/1000 for t in mut_ticks], mut_rates, "-", color="#1565c0",
                linewidth=1)
        ax.set_xlabel("Ticks (thousands)")
        ax.set_ylabel("Mutations per tick")
        ax.fill_between([t/1000 for t in mut_ticks], mut_rates, alpha=0.15,
                        color="#1565c0")
    ax.grid(True, alpha=0.3)

    # Panel 4: Cumulative mutations
    ax = fig.add_subplot(gs[1, 1])
    ax.set_title("Cumulative mutations", fontsize=11)
    if mut_counts:
        cumulative = np.cumsum(mut_counts)
        ax.plot([t/1000 for t in mut_ticks], cumulative, "-", color="#2e7d32",
                linewidth=1.5)
        ax.set_xlabel("Ticks (thousands)")
        ax.set_ylabel("Total mutations")
        ax.fill_between([t/1000 for t in mut_ticks], cumulative, alpha=0.15,
                        color="#2e7d32")
    ax.grid(True, alpha=0.3)

    # Panel 5: Output bias evolution for longest lineage
    ax = fig.add_subplot(gs[2, 0])
    ax.set_title("Output bias evolution (longest lineage)", fontsize=11)
    if top_chains and genome_snapshots:
        longest = top_chains[0]
        bias_ticks, biases = extract_output_biases(genome_snapshots, longest)
        if len(bias_ticks) > 0:
            key_actions = [0, 3, 5, 8, 6]  # move, eat, divide, attack, bond
            colors = ["#1565c0", "#2e7d32", "#6a1b9a", "#e53935", "#ff8f00"]
            for idx, action_i in enumerate(key_actions):
                ax.plot([t/1000 for t in bias_ticks], biases[:, action_i],
                        "-o", ms=3, linewidth=1, color=colors[idx],
                        label=ACTION_NAMES[action_i])
            ax.set_xlabel("Ticks (thousands)")
            ax.set_ylabel("Output bias value")
            ax.legend(fontsize=7, loc="best")
        else:
            ax.text(0.5, 0.5, "No weight snapshots overlap lineage",
                    transform=ax.transAxes, ha="center", va="center")
    else:
        ax.text(0.5, 0.5, "No lineage or weight data",
                transform=ax.transAxes, ha="center", va="center")
    ax.grid(True, alpha=0.3)

    # Panel 6: Population vs mutation rate correlation
    ax = fig.add_subplot(gs[2, 1])
    ax.set_title("Population vs mutation rate", fontsize=11)
    if metrics and mut_ticks:
        met_ticks = [r["tick"] for r in metrics]
        met_pop = [r["population"] for r in metrics]
        ax.plot([t/1000 for t in met_ticks], met_pop, "-", color="#1565c0",
                linewidth=1, alpha=0.7, label="Population")
        ax2 = ax.twinx()
        ax2.plot([t/1000 for t in mut_ticks], mut_rates, "-", color="#e65100",
                 linewidth=1, alpha=0.7, label="Mut rate")
        ax.set_xlabel("Ticks (thousands)")
        ax.set_ylabel("Population", color="#1565c0")
        ax2.set_ylabel("Mutations/tick", color="#e65100")
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize=7)
    ax.grid(True, alpha=0.3)

    out_path = os.path.join(output_dir, "lineage_tree.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def generate_lineage_report(events, parent_of, children, surviving,
                            genome_snapshots, metrics, run_name) -> str:
    """Generate markdown report."""
    lines = []
    lines.append("# Lineage Analysis")
    lines.append("")
    lines.append(f"**Run:** `{run_name}`  ")
    lines.append(f"**Mutation events:** {len(events):,}  ")
    if events:
        lines.append(f"**Tick range:** {events[0]['tick']:,} - {events[-1]['tick']:,}  ")
    lines.append("")

    # Summary stats
    lines.append("## Mutation Summary")
    lines.append("")

    unique_parents = set(e["parent"] for e in events)
    unique_children = set(e["child"] for e in events)
    lines.append(f"| Metric | Value |")
    lines.append(f"|--------|-------|")
    lines.append(f"| Total mutation events | {len(events):,} |")
    lines.append(f"| Unique parent genomes | {len(unique_parents):,} |")
    lines.append(f"| Unique child genomes | {len(unique_children):,} |")
    lines.append(f"| Surviving genomes (latest snapshot) | {len(surviving):,} |")

    if events:
        ticks_span = events[-1]["tick"] - events[0]["tick"]
        if ticks_span > 0:
            lines.append(f"| Avg mutations/tick | {len(events)/ticks_span:.2f} |")
    lines.append("")

    # Top lineages
    top_chains = find_top_lineages(events, surviving, parent_of, top_n=10)
    if top_chains:
        lines.append("## Longest Surviving Lineages")
        lines.append("")
        lines.append("| Rank | Depth | Root genome | Tip genome |")
        lines.append("|------|-------|-------------|------------|")
        for i, chain in enumerate(top_chains[:10]):
            lines.append(f"| {i+1} | {len(chain)} | {chain[0]} | {chain[-1]} |")
        lines.append("")

    # Selective sweeps
    sweep_ticks, root_counts = compute_root_diversity(events, parent_of)
    if root_counts:
        lines.append("## Selective Sweep Indicators")
        lines.append("")
        lines.append(f"- Initial root diversity: {root_counts[0] if root_counts else 'N/A'}")
        lines.append(f"- Final root diversity: {root_counts[-1] if root_counts else 'N/A'}")
        min_roots = min(root_counts)
        min_idx = root_counts.index(min_roots)
        lines.append(f"- Minimum root diversity: {min_roots} at tick ~{sweep_ticks[min_idx]:,}")
        lines.append("")
        if root_counts[-1] < root_counts[0] * 0.5:
            lines.append("A significant selective sweep is indicated: root diversity "
                        "dropped by more than 50%, suggesting a dominant lineage displaced "
                        "many competing lineages.")
        else:
            lines.append("No strong selective sweep detected. Multiple independent "
                        "lineages persist.")
        lines.append("")

    # Mutation rate
    mut_ticks, mut_rates, mut_counts = compute_mutation_rates(events)
    if mut_rates:
        lines.append("## Mutation Dynamics")
        lines.append("")
        lines.append(f"| Metric | Value |")
        lines.append(f"|--------|-------|")
        lines.append(f"| Peak mutation rate | {max(mut_rates):.2f} per tick |")
        lines.append(f"| Final mutation rate | {mut_rates[-1]:.2f} per tick |")
        lines.append(f"| Total mutations | {sum(mut_counts):,} |")
        lines.append("")

    # Figures
    lines.append("## Figures")
    lines.append("")
    lines.append("![Lineage Analysis](lineage_tree.png)")
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def find_latest_run_with_lineage(runs_dir="runs"):
    """Find the most recent run that has lineage data."""
    run_dirs = sorted(glob.glob(os.path.join(runs_dir, "*")), reverse=True)
    for d in run_dirs:
        lineage_path = os.path.join(d, "lineage.jsonl")
        if os.path.isfile(lineage_path) and os.path.getsize(lineage_path) > 0:
            return d
    return None


def main():
    if len(sys.argv) > 1:
        run_dir = sys.argv[1]
    else:
        run_dir = find_latest_run_with_lineage()

    if not run_dir:
        print("No runs with lineage data found.")
        print("Run a simulation first — lineage events are logged to lineage.jsonl.")
        return

    run_name = os.path.basename(run_dir)
    output_dir = os.path.join("analysis", "output", run_name)
    os.makedirs(output_dir, exist_ok=True)

    print(f"Analyzing lineage: {run_dir}")
    print(f"Output -> {output_dir}/")

    # Load data
    print("  Loading lineage events...")
    events = load_lineage(run_dir)
    print(f"  {len(events):,} events")

    print("  Loading genome snapshots...")
    genome_snapshots = load_genome_snapshots(run_dir)
    print(f"  {len(genome_snapshots)} snapshots")

    print("  Loading metrics...")
    metrics = load_metrics(run_dir)
    print(f"  {len(metrics)} records")

    if not events:
        print("No lineage events found. Run simulation longer.")
        return

    # Build tree
    children, parent_of = build_tree(events)
    surviving = find_surviving_genomes(genome_snapshots)
    print(f"  {len(surviving)} surviving genomes in latest snapshot")

    # Summary stats
    top_chains = find_top_lineages(events, surviving, parent_of, top_n=5)
    if top_chains:
        print(f"\n  Top lineage depths: {[len(c) for c in top_chains]}")

    mut_ticks, mut_rates, mut_counts = compute_mutation_rates(events)
    if mut_rates:
        print(f"  Peak mutation rate: {max(mut_rates):.2f}/tick")
        print(f"  Total mutations: {sum(mut_counts):,}")

    # Plot
    if HAS_MPL:
        print("\nGenerating plots...")
        plot_lineage_analysis(events, parent_of, children, surviving,
                              genome_snapshots, metrics, output_dir, run_name)

    # Report
    print("Generating report...")
    report = generate_lineage_report(events, parent_of, children, surviving,
                                     genome_snapshots, metrics, run_name)
    report_path = os.path.join(output_dir, "LINEAGE_ANALYSIS.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"  Saved: {report_path}")

    print(f"\nDone! All outputs in {output_dir}/")


if __name__ == "__main__":
    main()
