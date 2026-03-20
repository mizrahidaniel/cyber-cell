"""Compare neural vs CRN genome runs with comprehensive visualization.

Reads metrics.jsonl and oee_metrics.jsonl from run directories and generates
side-by-side comparison plots and a summary report.

Usage:
    python analysis/compare_runs.py NEURAL_RUN_DIR CRN_RUN_DIR
    python analysis/compare_runs.py --latest  # auto-detect latest long runs
"""

import argparse
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


def load_metrics(run_dir: str) -> list[dict]:
    """Load metrics.jsonl from a run directory."""
    path = os.path.join(run_dir, "metrics.jsonl")
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def load_oee(run_dir: str) -> list[dict]:
    """Load oee_metrics.jsonl from a run directory."""
    path = os.path.join(run_dir, "oee_metrics.jsonl")
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def find_latest_runs(runs_dir: str = "runs", min_records: int = 20):
    """Find the two most recent long runs."""
    if not os.path.exists(runs_dir):
        return []
    dirs = sorted([os.path.join(runs_dir, d) for d in os.listdir(runs_dir)
                   if os.path.isdir(os.path.join(runs_dir, d))], reverse=True)
    long_runs = []
    for d in dirs:
        metrics = load_metrics(d)
        if len(metrics) >= min_records:
            long_runs.append(d)
        if len(long_runs) >= 2:
            break
    return long_runs


def extract_series(metrics: list[dict], key: str) -> tuple[np.ndarray, np.ndarray]:
    """Extract a time series from metrics records."""
    ticks = np.array([m["tick"] for m in metrics if key in m])
    vals = np.array([m[key] for m in metrics if key in m])
    return ticks, vals


def _load_crn_metrics(path: str) -> list[dict]:
    """Load crn_metrics.jsonl from a file path."""
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _plot_crn_comparison(path_a: str | None, path_b: str, output_dir: str):
    """Generate CRN-specific comparison figure (2x3)."""
    recs_a = _load_crn_metrics(path_a) if path_a else []
    recs_b = _load_crn_metrics(path_b)
    if not recs_a and not recs_b:
        return

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("CRN Genome Comparison", fontsize=14, fontweight="bold")

    panels = [
        ("hidden_mean", "Hidden Zone Activation"),
        (lambda r: r["bias_mean"][0], "Eat Bias"),
        ("active_reactions", "Active Reactions"),
        ("inverted_threshold_frac", "Inverted Thresholds"),
        (lambda r: r["bias_mean"][1], "Move Bias"),
        (lambda r: r["bias_mean"][3], "Attack Bias"),
    ]

    for idx, (key, title) in enumerate(panels):
        ax = axes[idx // 3, idx % 3]
        for recs, color, label in [(recs_a, "b", "Run A"), (recs_b, "r", "Run B")]:
            if not recs:
                continue
            ticks = np.array([r["tick"] for r in recs])
            if callable(key):
                vals = np.array([key(r) for r in recs])
            else:
                vals = np.array([r[key] for r in recs])
            ax.plot(ticks, vals, f"{color}-", linewidth=0.8, alpha=0.8,
                    label=label)
        ax.set_title(title)
        ax.set_xlabel("Tick")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "comparison_crn.png"), dpi=150)
    plt.close()


def detect_genome_type(run_dir: str) -> str:
    """Detect genome type from run directory contents."""
    crn_path = os.path.join(run_dir, "crn_metrics.jsonl")
    return "CRN" if os.path.exists(crn_path) else "Neural"


def plot_comparison(dir_a: str, dir_b: str, output_dir: str):
    """Generate comprehensive comparison plots."""
    os.makedirs(output_dir, exist_ok=True)

    label_a = detect_genome_type(dir_a)
    label_b = detect_genome_type(dir_b)

    n_metrics = load_metrics(dir_a)
    c_metrics = load_metrics(dir_b)
    n_oee = load_oee(dir_a)
    c_oee = load_oee(dir_b)

    if not n_metrics and not c_metrics:
        print("No metrics found in either directory!")
        return

    # ── Figure 1: Population & Energy ──
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f"{label_a} vs {label_b}: Population & Energy Dynamics",
                 fontsize=14, fontweight="bold")

    series_pairs = [
        ("population", "Population"),
        ("avg_energy", "Average Energy"),
        ("num_genomes", "Unique Genomes"),
        ("move_fraction", "Movement Fraction (chemotaxis indicator)"),
        ("bond_fraction", "Bond Fraction"),
        ("attack_fraction", "Attack Fraction (predation)"),
    ]

    for idx, (key, title) in enumerate(series_pairs):
        ax = axes[idx // 3, idx % 3]
        if n_metrics:
            t, v = extract_series(n_metrics, key)
            ax.plot(t, v, "b-", linewidth=0.8, alpha=0.8, label=label_a)
        if c_metrics:
            t, v = extract_series(c_metrics, key)
            ax.plot(t, v, "r-", linewidth=0.8, alpha=0.8, label=label_b)
        ax.set_title(title)
        ax.set_xlabel("Tick")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    ax.set_title("Attack Fraction (predation)")
    ax.set_xlabel("Tick")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "comparison_dynamics.png"), dpi=150)
    plt.close()

    # ── Figure 2: OEE Metrics ──
    if n_oee or c_oee:
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle(f"{label_a} vs {label_b}: Open-Ended Evolution Metrics",
                     fontsize=14, fontweight="bold")

        oee_keys = [
            ("entropy", "Shannon Entropy"),
            ("activity", "Bedau Evolutionary Activity"),
            ("modes_novelty", "MODES: Novelty"),
            ("modes_ecology", "MODES: Ecology (evenness)"),
            ("mutual_info", "Mutual Information (sense-action)"),
            ("bond_density", "Bond Density"),
        ]

        for idx, (key, title) in enumerate(oee_keys):
            ax = axes[idx // 3, idx % 3]
            if n_oee:
                t, v = extract_series(n_oee, key)
                if len(t) > 0:
                    ax.plot(t, v, "b-", linewidth=0.8, alpha=0.8, label=label_a)
            if c_oee:
                t, v = extract_series(c_oee, key)
                if len(t) > 0:
                    ax.plot(t, v, "r-", linewidth=0.8, alpha=0.8, label=label_b)
            ax.set_title(title)
            ax.set_xlabel("Tick")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "comparison_oee.png"), dpi=150)
        plt.close()

    # ── Figure 3: CRN comparison (if both have CRN metrics) ──
    a_crn_path = os.path.join(dir_a, "crn_metrics.jsonl")
    b_crn_path = os.path.join(dir_b, "crn_metrics.jsonl")
    if os.path.exists(a_crn_path) and os.path.exists(b_crn_path):
        _plot_crn_comparison(a_crn_path, b_crn_path, output_dir)
    elif os.path.exists(b_crn_path):
        _plot_crn_comparison(None, b_crn_path, output_dir)
    elif os.path.exists(a_crn_path):
        _plot_crn_comparison(None, a_crn_path, output_dir)

    # ── Figure 4: Environmental Pressure ──
    env_keys = [
        ("avg_waste_at_cells", "Avg Waste at Cells"),
        ("max_waste", "Peak Waste"),
        ("waste_gt_threshold_frac", "Cells Above Toxicity Threshold"),
        ("bright_pct", "Bright Zone %"),
        ("dim_pct", "Dim Zone %"),
        ("avg_local_density", "Avg Local Density"),
    ]
    # Check if any env keys exist in either run
    has_env = any(key in m for m in (n_metrics + c_metrics) for key, _ in env_keys)
    if has_env:
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle(f"{label_a} vs {label_b}: Environmental Pressure",
                     fontsize=14, fontweight="bold")
        for idx, (key, title) in enumerate(env_keys):
            ax = axes[idx // 3, idx % 3]
            if n_metrics:
                t, v = extract_series(n_metrics, key)
                if len(t) > 0:
                    ax.plot(t, v, "b-", linewidth=0.8, alpha=0.8, label=label_a)
            if c_metrics:
                t, v = extract_series(c_metrics, key)
                if len(t) > 0:
                    ax.plot(t, v, "r-", linewidth=0.8, alpha=0.8, label=label_b)
            ax.set_title(title)
            ax.set_xlabel("Tick")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "comparison_env.png"), dpi=150)
        plt.close()

    # ── Summary report ──
    report_lines = [f"# {label_a} vs {label_b} Comparison Report\n"]

    for label, metrics, oee in [(label_a, n_metrics, n_oee),
                                 (label_b, c_metrics, c_oee)]:
        if not metrics:
            continue
        report_lines.append(f"\n## {label} Genome\n")
        t, pop = extract_series(metrics, "population")
        report_lines.append(f"- **Duration:** {int(t[-1]):,} ticks ({len(metrics)} snapshots)")
        report_lines.append(f"- **Population:** min={int(pop.min())}, "
                           f"max={int(pop.max())}, final={int(pop[-1])}, "
                           f"mean={pop.mean():.0f}")
        _, energy = extract_series(metrics, "avg_energy")
        if len(energy) > 0:
            report_lines.append(f"- **Avg Energy:** {energy.mean():.1f} "
                               f"(range {energy.min():.1f}-{energy.max():.1f})")
        _, move = extract_series(metrics, "move_fraction")
        if len(move) > 0:
            report_lines.append(f"- **Movement:** {move.mean():.1%} of cells moving")
        _, attack = extract_series(metrics, "attack_fraction")
        if len(attack) > 0:
            report_lines.append(f"- **Attack:** {attack.mean():.1%} of cells attacking")
        _, bond = extract_series(metrics, "bond_fraction")
        if len(bond) > 0:
            report_lines.append(f"- **Bonding:** {bond.mean():.1%} of cells bonded")

        if oee:
            _, ent = extract_series(oee, "entropy")
            _, act = extract_series(oee, "activity")
            _, nov = extract_series(oee, "modes_novelty")
            _, eco = extract_series(oee, "modes_ecology")
            _, mi = extract_series(oee, "mutual_info")
            if len(ent) > 0:
                report_lines.append(f"- **Shannon Entropy:** {ent.mean():.2f}")
            if len(act) > 0:
                report_lines.append(f"- **Evolutionary Activity:** {act.mean():.4f}")
            if len(nov) > 0:
                report_lines.append(f"- **Novelty Rate:** {nov.mean():.3f}")
            if len(eco) > 0:
                report_lines.append(f"- **Ecology Evenness:** {eco.mean():.3f}")
            if len(mi) > 0:
                report_lines.append(f"- **Mutual Information:** {mi.mean():.4f}")

        # Waste and zone stats
        _, waste = extract_series(metrics, "avg_waste_at_cells")
        if len(waste) > 0:
            report_lines.append(f"- **Avg Waste at Cells:** {waste.mean():.4f}")
        _, wgt = extract_series(metrics, "waste_gt_threshold_frac")
        if len(wgt) > 0:
            report_lines.append(f"- **Cells Above Toxicity:** {wgt.mean():.1%}")
        _, bright = extract_series(metrics, "bright_pct")
        if len(bright) > 0:
            report_lines.append(f"- **Bright Zone %:** {bright.mean():.1%}")
        _, dim = extract_series(metrics, "dim_pct")
        if len(dim) > 0:
            report_lines.append(f"- **Dim Zone %:** {dim.mean():.1%}")

    report = "\n".join(report_lines)
    report_path = os.path.join(output_dir, "comparison_report.md")
    with open(report_path, "w") as f:
        f.write(report)

    print(f"\nPlots saved to {output_dir}/")
    print(f"Report saved to {report_path}")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare neural vs CRN runs")
    parser.add_argument("dirs", nargs="*", help="Run directories to compare")
    parser.add_argument("--latest", action="store_true",
                       help="Auto-detect latest long runs")
    args = parser.parse_args()

    if args.latest:
        runs = find_latest_runs()
        if len(runs) < 2:
            print(f"Found only {len(runs)} long runs. Need at least 2.")
            sys.exit(1)
        dirs = runs[:2]
    elif len(args.dirs) >= 2:
        dirs = args.dirs[:2]
    else:
        print("Usage: python analysis/compare_runs.py DIR1 DIR2")
        print("   or: python analysis/compare_runs.py --latest")
        sys.exit(1)

    name_a = os.path.basename(dirs[0].rstrip("/\\"))
    name_b = os.path.basename(dirs[1].rstrip("/\\"))
    output_dir = os.path.join("analysis", "output",
                              f"comparison_{name_a}_vs_{name_b}")

    print(f"Comparing:\n  {dirs[0]}\n  {dirs[1]}")
    report = plot_comparison(dirs[0], dirs[1], output_dir)
    if report:
        print(report)
