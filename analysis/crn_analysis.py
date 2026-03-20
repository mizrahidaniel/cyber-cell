"""CRN genome deep diagnostics -- 9-panel visualization + markdown report.

Reads crn_metrics.jsonl from a simulation run and generates:
- crn_evolution.png (9-panel figure)
- CRN_ANALYSIS.md (markdown report)

Usage:
    python analysis/crn_analysis.py                      # latest run
    python analysis/crn_analysis.py runs/20260319_123456 # specific run
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


def load_crn_metrics(run_dir: str) -> list[dict]:
    path = Path(run_dir) / "crn_metrics.jsonl"
    if not path.exists():
        return []
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def find_latest_crn_run(runs_dir="runs"):
    run_dirs = sorted(glob.glob(os.path.join(runs_dir, "*")), reverse=True)
    for d in run_dirs:
        if os.path.isfile(os.path.join(d, "crn_metrics.jsonl")):
            return d
    return None


def _draw_reaction_graph(ax, reactions):
    """Draw reaction network as circular node layout."""
    NC = 16
    angles = np.linspace(0, 2 * np.pi, NC, endpoint=False)
    positions = {c: (np.cos(a), np.sin(a)) for c, a in enumerate(angles)}

    for c in range(NC):
        x, y = positions[c]
        color = ZONE_COLORS["sensory"] if c < 8 else (
            ZONE_COLORS["hidden"] if c < 12 else ZONE_COLORS["action"])
        ax.scatter(x, y, s=200, c=color, edgecolors="black",
                   linewidths=0.5, zorder=3)
        ax.text(x, y, str(c), ha="center", va="center", fontsize=6, zorder=4)

    for rxn in reactions:
        rate = rxn.get("rate", 0)
        if abs(rate) < 0.001:
            continue
        for inp in set([rxn["input_a"], rxn["input_b"]]):
            out = rxn["output"]
            if inp == out:
                continue
            x0, y0 = positions[inp]
            x1, y1 = positions[out]
            color = "#e53935" if rate < 0 else "#1565c0"
            ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                        arrowprops=dict(arrowstyle="->", color=color,
                                        alpha=min(1.0, abs(rate) * 2), lw=0.8))
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect("equal")
    ax.axis("off")


def plot_crn_analysis(records: list[dict], output_dir: str, run_name: str):
    if not HAS_MPL or not records:
        return

    ticks = np.array([r["tick"] for r in records]) / 1000
    fig = plt.figure(figsize=(20, 16))
    fig.suptitle(f"CRN Genome Deep Diagnostics — {run_name}\n"
                 f"{len(records)} snapshots, {int(ticks[-1]*1000):,} ticks",
                 fontsize=14, fontweight="bold")
    gs = GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

    # Panel 1: Zone activation (stacked area)
    ax = fig.add_subplot(gs[0, 0])
    s_m = [r["sensory_mean"] for r in records]
    h_m = [r["hidden_mean"] for r in records]
    a_m = [r["action_mean"] for r in records]
    ax.stackplot(ticks, s_m, h_m, a_m,
                 labels=["Sensory (0-7)", "Hidden (8-11)", "Action (12-15)"],
                 colors=[ZONE_COLORS["sensory"], ZONE_COLORS["hidden"],
                         ZONE_COLORS["action"]], alpha=0.7)
    ax.set_title("Zone Activation Over Time")
    ax.set_xlabel("Ticks (thousands)")
    ax.set_ylabel("Mean concentration")
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(True, alpha=0.3)

    # Panel 2: Individual chemical trajectories
    ax = fig.add_subplot(gs[0, 1])
    for c in range(16):
        vals = [r["chem_means"][c] for r in records]
        if c < 8:
            color, alpha = ZONE_COLORS["sensory"], 0.4 + 0.06 * c
        elif c < 12:
            color, alpha = ZONE_COLORS["hidden"], 0.5 + 0.12 * (c - 8)
        else:
            color, alpha = ZONE_COLORS["action"], 0.5 + 0.12 * (c - 12)
        label = f"C{c}" if c in (0, 8, 12) else None
        ax.plot(ticks, vals, color=color, alpha=alpha, linewidth=0.8, label=label)
    ax.set_title("Chemical Trajectories (16 chemicals)")
    ax.set_xlabel("Ticks (thousands)")
    ax.set_ylabel("Mean concentration")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # Panel 3: Action bias evolution
    ax = fig.add_subplot(gs[0, 2])
    bias_names = ["eat", "move", "divide", "attack"]
    bias_colors = ["#2e7d32", "#1565c0", "#6a1b9a", "#e53935"]
    for a in range(4):
        vals = [r["bias_mean"][a] for r in records]
        ax.plot(ticks, vals, color=bias_colors[a], linewidth=1.5,
                label=bias_names[a])
    ax.set_title("Action Bias Evolution")
    ax.set_xlabel("Ticks (thousands)")
    ax.set_ylabel("Bias value")
    ax.legend(fontsize=8)
    ax.axhline(0, color="gray", linestyle="--", alpha=0.3)
    ax.grid(True, alpha=0.3)

    # Panel 4: Hidden decay rates
    ax = fig.add_subplot(gs[1, 0])
    for h in range(4):
        vals = [r["decay_mean"][h] for r in records]
        ax.plot(ticks, vals, linewidth=1.5, label=f"Hidden {8+h}")
    ax.set_title("Hidden Decay Rate Evolution")
    ax.set_xlabel("Ticks (thousands)")
    ax.set_ylabel("Decay rate")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 5: Inverted threshold fraction
    ax = fig.add_subplot(gs[1, 1])
    inv = [r["inverted_threshold_frac"] for r in records]
    ax.plot(ticks, inv, color="#e65100", linewidth=1.5)
    ax.fill_between(ticks, inv, alpha=0.2, color="#e65100")
    ax.set_title("Inverted Threshold Fraction (NOT-gates)")
    ax.set_xlabel("Ticks (thousands)")
    ax.set_ylabel("Fraction negative")
    ax.grid(True, alpha=0.3)

    # Panel 6: Active reaction count
    ax = fig.add_subplot(gs[1, 2])
    active = [r["active_reactions"] for r in records]
    ax.plot(ticks, active, color="#1565c0", linewidth=1.5)
    ax.set_title("Active Reactions (mean per genome)")
    ax.set_xlabel("Ticks (thousands)")
    ax.set_ylabel("Count (of 16)")
    ax.set_ylim(0, 16.5)
    ax.grid(True, alpha=0.3)

    # Panel 7: Reaction network graph (dominant genome, final)
    ax = fig.add_subplot(gs[2, 0])
    final = records[-1]
    _draw_reaction_graph(ax, final.get("dominant_reactions", []))
    ax.set_title(f"Dominant Genome Reactions (gid={final.get('dominant_gid', '?')})")

    # Panel 8: Zone flow heatmap
    ax = fig.add_subplot(gs[2, 1])
    flow = np.array(final.get("zone_flow", [[0]*3]*3))
    im = ax.imshow(flow, cmap="YlOrRd", aspect="equal")
    zone_labels = ["Sensory", "Hidden", "Action"]
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(zone_labels)
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(zone_labels)
    ax.set_xlabel("Target zone")
    ax.set_ylabel("Source zone")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, str(flow[i, j]), ha="center", va="center",
                    fontsize=14, fontweight="bold")
    ax.set_title("Reaction Zone Flow")
    plt.colorbar(im, ax=ax, shrink=0.8)

    # Panel 9: Chemical distribution (final snapshot)
    ax = fig.add_subplot(gs[2, 2])
    means = final["chem_means"]
    stds = final["chem_stds"]
    colors = ([ZONE_COLORS["sensory"]] * 8 +
              [ZONE_COLORS["hidden"]] * 4 +
              [ZONE_COLORS["action"]] * 4)
    ax.bar(range(16), means, yerr=stds, color=colors, edgecolor="black",
           linewidth=0.5, alpha=0.7, capsize=2)
    ax.set_title("Chemical Distribution (final snapshot)")
    ax.set_xlabel("Chemical index")
    ax.set_ylabel("Mean +/- std")
    ax.set_xticks(range(16))
    ax.grid(True, alpha=0.3, axis="y")

    path = os.path.join(output_dir, "crn_evolution.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved: {path}")


def generate_crn_report(records: list[dict], run_name: str) -> str:
    """Generate CRN analysis markdown report."""
    if not records:
        return "# CRN Analysis\n\nNo CRN metrics data found.\n"

    final = records[-1]
    lines = [
        "# CRN Genome Deep Analysis",
        "",
        f"**Run:** `{run_name}`  ",
        f"**Snapshots:** {len(records)}  ",
        f"**Duration:** {final['tick']:,} ticks  ",
        f"**Active genomes:** {final['num_active_genomes']}  ",
        "",
        "## Chemical Summary",
        "",
        "| Zone | Chemicals | Mean | Description |",
        "|------|-----------|------|-------------|",
        f"| Sensory | 0-7 | {final['sensory_mean']:.3f} | Environment inputs |",
        f"| Hidden | 8-11 | {final['hidden_mean']:.3f} | Internal memory/gates |",
        f"| Action | 12-15 | {final['action_mean']:.3f} | Action triggers |",
        "",
        "## Action Biases (population-weighted)",
        "",
        "| Action | Bias Value | Interpretation |",
        "|--------|-----------|----------------|",
    ]
    bias_names = ["eat", "move", "divide", "attack"]
    for i, name in enumerate(bias_names):
        val = final["bias_mean"][i]
        strength = "strong" if abs(val) > 0.3 else (
            "moderate" if abs(val) > 0.1 else "weak")
        direction = "positive" if val > 0 else "negative"
        lines.append(f"| {name} | {val:+.3f} | {strength} {direction} |")

    lines += [
        "",
        "## Hidden Decay Rates",
        "",
        "Low decay = long memory. High decay = short-term reactivity.",
        "",
        "| Chemical | Decay Rate | Memory Half-Life |",
        "|----------|-----------|-----------------|",
    ]
    for h in range(4):
        decay = final["decay_mean"][h]
        hl = int(0.693 / max(0.001, decay)) if decay > 0.001 else ">1000"
        lines.append(f"| Hidden {8+h} | {decay:.4f} | {hl} ticks |")

    lines += [
        "",
        "## Computational Sophistication",
        "",
        f"- **Active reactions:** {final['active_reactions']:.1f} of 16",
        f"- **Inverted thresholds (NOT-gates):** "
        f"{final['inverted_threshold_frac']:.1%}",
        f"- **Dominant genome:** gid={final['dominant_gid']} "
        f"({final['dominant_count']} cells)",
        "",
    ]

    flow = final.get("zone_flow", [[0]*3]*3)
    lines += [
        "## Reaction Zone Flow (dominant genome)",
        "",
        "| Source \\ Target | Sensory | Hidden | Action |",
        "|---------------|---------|--------|--------|",
    ]
    zone_names = ["Sensory", "Hidden", "Action"]
    for i, name in enumerate(zone_names):
        row = " | ".join(str(flow[i][j]) for j in range(3))
        lines.append(f"| {name} | {row} |")

    # Auto findings
    lines += ["", "## Key Findings", ""]
    if final["hidden_mean"] > 0.01:
        lines.append(
            "- Hidden chemicals are active — CRN is using internal state")
    else:
        lines.append(
            "- Hidden chemicals are near zero — CRN is purely reactive")

    inv_frac = final["inverted_threshold_frac"]
    if inv_frac > 0.1:
        lines.append(
            f"- {inv_frac:.0%} inverted thresholds indicate "
            f"NOT-gate logic has evolved")

    ar = final["active_reactions"]
    if ar > 10:
        lines.append(f"- Using {ar:.0f}/16 reactions — complex network")
    elif ar < 5:
        lines.append(f"- Only {ar:.0f}/16 reactions active — minimal network")

    flow_arr = np.array(flow)
    s2h, h2a = flow_arr[0, 1], flow_arr[1, 2]
    if s2h > 0 and h2a > 0:
        lines.append(
            f"- Sensory->Hidden->Action pathway exists "
            f"({s2h}+{h2a} reactions)")
    elif flow_arr[0, 2] > 0:
        lines.append(
            f"- Direct Sensory->Action pathway "
            f"({flow_arr[0, 2]} reactions), no hidden processing")

    lines += ["", "## Figures", "", "![CRN Evolution](crn_evolution.png)", ""]
    return "\n".join(lines)


def main():
    if len(sys.argv) > 1:
        run_dir = sys.argv[1]
    else:
        run_dir = find_latest_crn_run()

    if not run_dir:
        print("No CRN runs found. Run with --genome crn first.")
        return

    run_name = os.path.basename(run_dir)
    output_dir = os.path.join("analysis", "output", run_name)
    os.makedirs(output_dir, exist_ok=True)

    print(f"Analyzing CRN run: {run_dir}")
    records = load_crn_metrics(run_dir)
    print(f"  {len(records)} CRN snapshots")

    if not records:
        print("  No CRN metrics data found.")
        return

    if HAS_MPL:
        print("Generating CRN plots...")
        plot_crn_analysis(records, output_dir, run_name)

    print("Generating CRN report...")
    report = generate_crn_report(records, run_name)
    report_path = os.path.join(output_dir, "CRN_ANALYSIS.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"  Saved: {report_path}")

    print(f"\nDone! All outputs in {output_dir}/")


if __name__ == "__main__":
    main()
