"""CTRNN genome diagnostics -- 9-panel visualization + markdown report.

Reads ctrnn_metrics.jsonl from a simulation run and generates:
- ctrnn_evolution.png (9-panel figure)
- CTRNN_ANALYSIS.md (markdown report)

Usage:
    python analysis/ctrnn_analysis.py                      # latest run
    python analysis/ctrnn_analysis.py runs/20260319_123456 # specific run
"""

import glob
import json
import os
import sys
from pathlib import Path

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

ZONE_COLORS = {"sensory": "#4CAF50", "hidden": "#FF9800", "action": "#2196F3"}


def load_ctrnn_metrics(run_dir: str) -> list[dict]:
    path = Path(run_dir) / "ctrnn_metrics.jsonl"
    if not path.exists():
        return []
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def find_latest_ctrnn_run(runs_dir="runs"):
    run_dirs = sorted(glob.glob(os.path.join(runs_dir, "*")), reverse=True)
    for d in run_dirs:
        if os.path.isfile(os.path.join(d, "ctrnn_metrics.jsonl")):
            return d
    return None


def records_to_arrays(records):
    """Convert list of dicts to dict of numpy arrays."""
    if not records:
        return {}
    keys = records[0].keys()
    arrays = {}
    for k in keys:
        vals = [r.get(k, 0) for r in records]
        if isinstance(vals[0], list):
            arrays[k] = np.array(vals)
        else:
            arrays[k] = np.array(vals, dtype=float)
    return arrays


def plot_ctrnn_evolution(arrays, output_dir):
    """Generate 9-panel CTRNN diagnostics figure."""
    if not HAS_MPL:
        print("  matplotlib not available, skipping plot")
        return

    os.makedirs(output_dir, exist_ok=True)
    ticks = arrays.get("tick", np.arange(len(next(iter(arrays.values())))))

    fig = plt.figure(figsize=(18, 14))
    gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)
    fig.suptitle("CTRNN Evolution Diagnostics", fontsize=14, fontweight="bold")

    # 1. Zone activation means
    ax = fig.add_subplot(gs[0, 0])
    for key, color, label in [
        ("sensory_mean", ZONE_COLORS["sensory"], "Sensory (0-7)"),
        ("hidden_mean", ZONE_COLORS["hidden"], "Hidden (8-11)"),
        ("action_mean", ZONE_COLORS["action"], "Action (12-15)"),
    ]:
        if key in arrays:
            ax.plot(ticks, arrays[key], color=color, label=label, linewidth=0.8)
    ax.set_title("Zone Activation Means")
    ax.set_xlabel("Tick")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # 2. Per-neuron activation heatmap
    ax = fig.add_subplot(gs[0, 1])
    if "neuron_means" in arrays:
        data = arrays["neuron_means"]
        if data.ndim == 2 and data.shape[0] > 1:
            ax.imshow(data.T, aspect="auto", cmap="RdYlBu_r",
                      extent=[float(ticks[0]), float(ticks[-1]), 15.5, -0.5])
            ax.set_ylabel("Neuron Index")
            ax.axhline(7.5, color="white", linewidth=1, linestyle="--")
            ax.axhline(11.5, color="white", linewidth=1, linestyle="--")
    ax.set_title("Neuron Activations (heatmap)")
    ax.set_xlabel("Tick")

    # 3. Tau evolution
    ax = fig.add_subplot(gs[0, 2])
    if "tau_mean" in arrays:
        data = arrays["tau_mean"]
        if data.ndim == 2:
            # Sensory avg
            ax.plot(ticks, data[:, :8].mean(axis=1),
                    color=ZONE_COLORS["sensory"], label="Sensory", linewidth=0.8)
            # Hidden avg
            ax.plot(ticks, data[:, 8:12].mean(axis=1),
                    color=ZONE_COLORS["hidden"], label="Hidden", linewidth=0.8)
            # Action avg
            ax.plot(ticks, data[:, 12:].mean(axis=1),
                    color=ZONE_COLORS["action"], label="Action", linewidth=0.8)
    ax.set_title("Time Constants (tau)")
    ax.set_xlabel("Tick")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # 4. Bias evolution
    ax = fig.add_subplot(gs[1, 0])
    if "bias_mean" in arrays:
        data = arrays["bias_mean"]
        if data.ndim == 2:
            for zone, sl, color in [
                ("Sensory", slice(0, 8), ZONE_COLORS["sensory"]),
                ("Hidden", slice(8, 12), ZONE_COLORS["hidden"]),
                ("Action", slice(12, 16), ZONE_COLORS["action"]),
            ]:
                ax.plot(ticks, data[:, sl].mean(axis=1),
                        color=color, label=zone, linewidth=0.8)
    ax.set_title("Neuron Biases")
    ax.set_xlabel("Tick")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # 5. Amplitude evolution
    ax = fig.add_subplot(gs[1, 1])
    if "amp_mean" in arrays:
        data = arrays["amp_mean"]
        if data.ndim == 2:
            for zone, sl, color in [
                ("Sensory", slice(0, 8), ZONE_COLORS["sensory"]),
                ("Hidden", slice(8, 12), ZONE_COLORS["hidden"]),
                ("Action", slice(12, 16), ZONE_COLORS["action"]),
            ]:
                ax.plot(ticks, data[:, sl].mean(axis=1),
                        color=color, label=zone, linewidth=0.8)
    ax.set_title("Amplitudes (A)")
    ax.set_xlabel("Tick")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # 6. Action biases
    ax = fig.add_subplot(gs[1, 2])
    if "action_biases" in arrays:
        data = arrays["action_biases"]
        if data.ndim == 2:
            labels = ["eat", "move", "divide", "attack"]
            colors = ["#4CAF50", "#2196F3", "#FF9800", "#F44336"]
            for a in range(min(4, data.shape[1])):
                ax.plot(ticks, data[:, a], color=colors[a],
                        label=labels[a], linewidth=0.8)
    ax.set_title("Action Biases")
    ax.set_xlabel("Tick")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # 7. Neuron std (activity diversity)
    ax = fig.add_subplot(gs[2, 0])
    if "neuron_stds" in arrays:
        data = arrays["neuron_stds"]
        if data.ndim == 2:
            ax.plot(ticks, data[:, :8].mean(axis=1),
                    color=ZONE_COLORS["sensory"], label="Sensory", linewidth=0.8)
            ax.plot(ticks, data[:, 8:12].mean(axis=1),
                    color=ZONE_COLORS["hidden"], label="Hidden", linewidth=0.8)
            ax.plot(ticks, data[:, 12:].mean(axis=1),
                    color=ZONE_COLORS["action"], label="Action", linewidth=0.8)
    ax.set_title("Activation Diversity (std)")
    ax.set_xlabel("Tick")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # 8. Active genomes
    ax = fig.add_subplot(gs[2, 1])
    if "num_active_genomes" in arrays:
        ax.plot(ticks, arrays["num_active_genomes"],
                color="#9C27B0", linewidth=0.8)
    ax.set_title("Active Genomes")
    ax.set_xlabel("Tick")
    ax.grid(True, alpha=0.3)

    # 9. Action neuron activations (mean per action)
    ax = fig.add_subplot(gs[2, 2])
    if "neuron_means" in arrays:
        data = arrays["neuron_means"]
        if data.ndim == 2 and data.shape[1] >= 16:
            labels = ["eat (12)", "move (13)", "divide (14)", "attack (15)"]
            colors = ["#4CAF50", "#2196F3", "#FF9800", "#F44336"]
            for a in range(4):
                ax.plot(ticks, data[:, 12 + a], color=colors[a],
                        label=labels[a], linewidth=0.8)
    ax.set_title("Action Neuron Activations")
    ax.set_xlabel("Tick")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    path = os.path.join(output_dir, "ctrnn_evolution.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Plot saved: {path}")


def generate_report(records, arrays, output_dir):
    """Generate markdown summary report."""
    os.makedirs(output_dir, exist_ok=True)

    n = len(records)
    if n == 0:
        return

    first, last = records[0], records[-1]
    tick_range = f"{first.get('tick', 0):,} - {last.get('tick', 0):,}"

    lines = [
        f"# CTRNN Analysis Report",
        f"",
        f"**Ticks:** {tick_range} ({n} snapshots)",
        f"**Active genomes:** {last.get('num_active_genomes', 'N/A')}",
        f"",
        f"## Zone Activations (final)",
        f"| Zone | Mean |",
        f"|------|------|",
        f"| Sensory | {last.get('sensory_mean', 0):.4f} |",
        f"| Hidden | {last.get('hidden_mean', 0):.4f} |",
        f"| Action | {last.get('action_mean', 0):.4f} |",
        f"",
    ]

    # Tau summary
    if "tau_mean" in arrays:
        tau_data = arrays["tau_mean"]
        if tau_data.ndim == 2 and len(tau_data) > 0:
            final_tau = tau_data[-1]
            lines.extend([
                f"## Time Constants (final)",
                f"| Zone | Mean tau |",
                f"|------|---------|",
                f"| Sensory | {final_tau[:8].mean():.3f} |",
                f"| Hidden | {final_tau[8:12].mean():.3f} |",
                f"| Action | {final_tau[12:].mean():.3f} |",
                f"",
            ])

    # Action biases
    if "action_biases" in arrays:
        ab = arrays["action_biases"]
        if ab.ndim == 2 and len(ab) > 0:
            final_ab = ab[-1]
            labels = ["eat", "move", "divide", "attack"]
            lines.extend([
                f"## Action Biases (final)",
                f"| Action | Bias |",
                f"|--------|------|",
            ])
            for a in range(min(4, len(final_ab))):
                lines.append(f"| {labels[a]} | {final_ab[a]:.4f} |")
            lines.append("")

    path = os.path.join(output_dir, "CTRNN_ANALYSIS.md")
    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"  Report saved: {path}")


def main():
    if len(sys.argv) > 1:
        run_dir = sys.argv[1]
    else:
        run_dir = find_latest_ctrnn_run()

    if not run_dir:
        print("No CTRNN run found.")
        sys.exit(1)

    run_name = os.path.basename(run_dir)
    output_dir = os.path.join("analysis", "output", run_name)

    print(f"CTRNN Analysis: {run_name}")
    records = load_ctrnn_metrics(run_dir)
    if not records:
        print("  No CTRNN metrics found.")
        sys.exit(1)

    print(f"  Loaded {len(records)} snapshots")
    arrays = records_to_arrays(records)
    plot_ctrnn_evolution(arrays, output_dir)
    generate_report(records, arrays, output_dir)


if __name__ == "__main__":
    main()
