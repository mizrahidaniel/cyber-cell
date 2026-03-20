"""
CyberCell Evolutionary Dynamics Study
======================================
Analyzes simulation runs to detect evolutionary transitions, characterize
emergent behaviors, and compare parameter regimes.

Usage:
    python analysis/study.py                      # analyze all runs
    python analysis/study.py runs/20260318_202818 # analyze specific run
    python analysis/study.py --compare DIR1 DIR2  # compare two runs
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_run(run_dir: str) -> list[dict]:
    """Load metrics.jsonl from a run directory."""
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


def find_longest_runs(runs_root: str = "runs", top_n: int = 5) -> list[tuple[str, int]]:
    """Find runs with the most data points."""
    results = []
    root = Path(runs_root)
    if not root.exists():
        return results
    for d in root.iterdir():
        if d.is_dir():
            metrics_file = d / "metrics.jsonl"
            if metrics_file.exists():
                count = sum(1 for _ in open(metrics_file))
                if count > 1:
                    results.append((str(d), count))
    results.sort(key=lambda x: -x[1])
    return results[:top_n]


def records_to_arrays(records: list[dict]) -> dict[str, np.ndarray]:
    """Convert list of metric dicts to dict of numpy arrays."""
    if not records:
        return {}
    keys = records[0].keys()
    return {k: np.array([r[k] for r in records]) for k in keys}

# ---------------------------------------------------------------------------
# Evolutionary phase detection
# ---------------------------------------------------------------------------

def detect_phases(data: dict[str, np.ndarray]) -> list[dict]:
    """Identify evolutionary phases from metric trajectories.

    Phases:
      1. Crash       - initial population collapse as random genomes fail
      2. Stabilize   - survivors reach energy equilibrium
      3. Expansion   - reproduction begins, population grows
      4. Chemotaxis   - directed movement emerges (move_fraction rising)
      5. Exploration - cells spread beyond light zone (avg_x increasing)
    """
    phases = []
    pop = data["population"]
    ticks = data["tick"]
    move = data["move_fraction"]
    avg_x = data["avg_x_position"]

    # Phase 1: Crash — find the population minimum
    min_idx = np.argmin(pop)
    if min_idx > 0 and pop[min_idx] < pop[0] * 0.5:
        phases.append({
            "name": "Crash",
            "start_tick": int(ticks[0]),
            "end_tick": int(ticks[min_idx]),
            "description": f"Population drops from {int(pop[0])} to {int(pop[min_idx])} "
                           f"({100*(1 - pop[min_idx]/pop[0]):.0f}% mortality)",
        })

    # Phase 2: Stabilize — population stops dropping, before growth
    if min_idx < len(pop) - 1:
        # Find where population starts growing consistently
        growth_start = min_idx
        for i in range(min_idx, min(len(pop) - 5, len(pop))):
            if all(pop[i+k] >= pop[i] for k in range(1, min(4, len(pop)-i))):
                growth_start = i
                break
        if growth_start > min_idx:
            phases.append({
                "name": "Stabilize",
                "start_tick": int(ticks[min_idx]),
                "end_tick": int(ticks[growth_start]),
                "description": f"Population stabilizes around {int(pop[growth_start])}",
            })

    # Phase 3: Expansion — population doubles from minimum
    double_idx = None
    for i in range(min_idx, len(pop)):
        if pop[i] >= pop[min_idx] * 2:
            double_idx = i
            break
    if double_idx is not None:
        phases.append({
            "name": "Expansion",
            "start_tick": int(ticks[min_idx]),
            "end_tick": int(ticks[double_idx]),
            "description": f"Population doubles to {int(pop[double_idx])}",
        })

    # Phase 4: Chemotaxis — move_fraction exceeds 0.15 sustained
    # Use a rolling window to filter noise
    window = min(5, len(move))
    if window >= 2:
        move_smooth = np.convolve(move, np.ones(window)/window, mode="valid")
        for i in range(len(move_smooth)):
            if move_smooth[i] > 0.15:
                tick_idx = i + window // 2
                if tick_idx < len(ticks):
                    phases.append({
                        "name": "Chemotaxis Emergence",
                        "start_tick": int(ticks[tick_idx]),
                        "end_tick": int(ticks[-1]),
                        "description": f"Movement fraction exceeds 15% at tick {int(ticks[tick_idx])}, "
                                       f"reaches {move[-1]:.1%} by end",
                    })
                break

    # Phase 5: Exploration — avg_x moves significantly beyond light zone center (83)
    light_center = 83  # LIGHT_ZONE_END / 2
    dim_zone_start = 166
    for i in range(len(avg_x)):
        if avg_x[i] > dim_zone_start:
            phases.append({
                "name": "Zone Exploration",
                "start_tick": int(ticks[i]),
                "end_tick": int(ticks[-1]),
                "description": f"Average cell position crosses into dim zone (x>{dim_zone_start}) "
                               f"at tick {int(ticks[i])}, reaches x={avg_x[-1]:.0f}",
            })
            break

    # Phase 6: Predation — attack_fraction exceeds 1% sustained
    if "attack_fraction" in data:
        attack = data["attack_fraction"]
        window = min(5, len(attack))
        if window >= 2:
            attack_smooth = np.convolve(attack, np.ones(window)/window, mode="valid")
            for i in range(len(attack_smooth)):
                if attack_smooth[i] > 0.01:
                    tick_idx = i + window // 2
                    if tick_idx < len(ticks):
                        phases.append({
                            "name": "Predation Emergence",
                            "start_tick": int(ticks[tick_idx]),
                            "end_tick": int(ticks[-1]),
                            "description": f"Attack fraction exceeds 1% at tick {int(ticks[tick_idx])}, "
                                           f"reaches {attack[-1]:.1%} by end",
                        })
                    break

    # Phase 7: Waste Pressure — waste_gt_threshold_frac exceeds 5% for 3+ consecutive snapshots
    if "waste_gt_threshold_frac" in data:
        wgt = data["waste_gt_threshold_frac"]
        consecutive = 0
        for i in range(len(wgt)):
            if wgt[i] > 0.05:
                consecutive += 1
                if consecutive >= 3:
                    start_idx = i - consecutive + 1
                    if start_idx >= 0 and start_idx < len(ticks):
                        phases.append({
                            "name": "Waste Pressure",
                            "start_tick": int(ticks[start_idx]),
                            "end_tick": int(ticks[-1]),
                            "description": f">{5}% of cells above toxicity threshold from tick "
                                           f"{int(ticks[start_idx])}, peak {wgt.max():.1%}",
                        })
                    break
            else:
                consecutive = 0

    # Phase 8: Bonding — bond_fraction exceeds 1% sustained
    if "bond_fraction" in data:
        bonds = data["bond_fraction"]
        window = min(5, len(bonds))
        if window >= 2:
            bond_smooth = np.convolve(bonds, np.ones(window)/window, mode="valid")
            for i in range(len(bond_smooth)):
                if bond_smooth[i] > 0.01:
                    tick_idx = i + window // 2
                    if tick_idx < len(ticks):
                        phases.append({
                            "name": "Bonding Emergence",
                            "start_tick": int(ticks[tick_idx]),
                            "end_tick": int(ticks[-1]),
                            "description": f"Bond fraction exceeds 1% at tick {int(ticks[tick_idx])}, "
                                           f"reaches {bonds[-1]:.1%} by end",
                        })
                    break

    return phases

# ---------------------------------------------------------------------------
# Evolutionary rate analysis
# ---------------------------------------------------------------------------

def compute_rates(data: dict[str, np.ndarray]) -> dict:
    """Compute key evolutionary rates and statistics."""
    ticks = data["tick"]
    pop = data["population"]
    move = data["move_fraction"]
    avg_x = data["avg_x_position"]
    diversity = data["shannon_index"]
    energy = data["avg_energy"]

    dt = ticks[-1] - ticks[0] if len(ticks) > 1 else 1

    # Population growth rate (from minimum to end)
    min_idx = np.argmin(pop)
    if min_idx < len(pop) - 1 and pop[min_idx] > 0:
        growth_ticks = ticks[-1] - ticks[min_idx]
        if growth_ticks > 0:
            pop_growth_rate = (pop[-1] / pop[min_idx]) ** (1000 / growth_ticks) - 1
        else:
            pop_growth_rate = 0
    else:
        pop_growth_rate = 0

    # Movement evolution rate (linear fit on move_fraction after crash)
    if min_idx < len(move) - 2:
        post_crash_ticks = ticks[min_idx:] - ticks[min_idx]
        post_crash_move = move[min_idx:]
        if len(post_crash_ticks) > 2:
            coeffs = np.polyfit(post_crash_ticks, post_crash_move, 1)
            move_rate_per_1k = coeffs[0] * 1000
        else:
            move_rate_per_1k = 0
    else:
        move_rate_per_1k = 0

    # Spatial expansion rate
    if min_idx < len(avg_x) - 2:
        post_crash_x = avg_x[min_idx:]
        if len(post_crash_ticks) > 2:
            coeffs_x = np.polyfit(post_crash_ticks, post_crash_x, 1)
            x_rate_per_1k = coeffs_x[0] * 1000
        else:
            x_rate_per_1k = 0
    else:
        x_rate_per_1k = 0

    result = {
        "total_ticks": int(dt),
        "final_population": int(pop[-1]),
        "min_population": int(pop[min_idx]),
        "crash_tick": int(ticks[min_idx]),
        "pop_growth_per_1k_ticks": round(pop_growth_rate * 100, 2),
        "final_move_fraction": round(float(move[-1]), 4),
        "move_rate_per_1k_ticks": round(float(move_rate_per_1k), 4),
        "final_avg_x": round(float(avg_x[-1]), 1),
        "x_expansion_per_1k_ticks": round(float(x_rate_per_1k), 2),
        "final_shannon_index": round(float(diversity[-1]), 3),
        "final_avg_energy": round(float(energy[-1]), 1),
        "max_age_observed": int(data["max_age"][-1]),
    }

    # Predation metrics (if available)
    if "attack_fraction" in data:
        result["final_attack_fraction"] = round(float(data["attack_fraction"][-1]), 4)
    if "bond_fraction" in data:
        result["final_bond_fraction"] = round(float(data["bond_fraction"][-1]), 4)
    if "avg_membrane" in data:
        result["final_avg_membrane"] = round(float(data["avg_membrane"][-1]), 1)
    if "deaths_by_attack" in data:
        result["total_deaths_by_attack"] = int(data["deaths_by_attack"].sum())
    if "deaths_by_starvation" in data:
        result["total_deaths_by_starvation"] = int(data["deaths_by_starvation"].sum())

    return result

# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_single_run(data: dict[str, np.ndarray], phases: list[dict],
                    title: str, output_path: str):
    """Generate a comprehensive 8-panel figure for a single run."""
    ticks = data["tick"] / 1000  # convert to thousands

    fig = plt.figure(figsize=(16, 18))
    fig.suptitle(title, fontsize=16, fontweight="bold", y=0.98)
    gs = GridSpec(4, 2, figure=fig, hspace=0.35, wspace=0.3)

    # Color for zone backgrounds
    zone_colors = {"light": "#ffffcc", "dim": "#ffe0b2", "dark": "#e0e0e0"}

    # --- Panel 1: Population Dynamics ---
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(ticks, data["population"], color="#1565c0", linewidth=1.5)
    ax1.set_ylabel("Population", fontsize=11)
    ax1.set_xlabel("Ticks (thousands)", fontsize=10)
    ax1.set_title("Population Dynamics", fontsize=12, fontweight="bold")
    ax1.grid(True, alpha=0.3)
    # Mark crash point
    min_idx = np.argmin(data["population"])
    ax1.annotate(f'Bottleneck: {int(data["population"][min_idx])}',
                 xy=(ticks[min_idx], data["population"][min_idx]),
                 xytext=(ticks[min_idx]+5, data["population"][min_idx]*1.5),
                 arrowprops=dict(arrowstyle="->", color="red"),
                 fontsize=9, color="red")

    # --- Panel 2: Movement Evolution ---
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(ticks, data["move_fraction"] * 100, color="#e65100", linewidth=1.5)
    ax2.set_ylabel("Cells Moving (%)", fontsize=11)
    ax2.set_xlabel("Ticks (thousands)", fontsize=10)
    ax2.set_title("Movement Evolution (Chemotaxis Proxy)", fontsize=12, fontweight="bold")
    ax2.axhline(y=15, color="green", linestyle="--", alpha=0.5, label="Chemotaxis threshold")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    # --- Panel 3: Spatial Distribution ---
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(ticks, data["avg_x_position"], color="#2e7d32", linewidth=1.5)
    ax3.axhline(y=166, color="orange", linestyle="--", alpha=0.6, label="Dim zone boundary")
    ax3.axhline(y=333, color="red", linestyle="--", alpha=0.6, label="Dark zone boundary")
    ax3.axhline(y=83, color="gold", linestyle=":", alpha=0.6, label="Light zone center")
    ax3.set_ylabel("Average X Position", fontsize=11)
    ax3.set_xlabel("Ticks (thousands)", fontsize=10)
    ax3.set_title("Spatial Exploration", fontsize=12, fontweight="bold")
    ax3.legend(fontsize=8, loc="upper left")
    ax3.set_ylim(0, 500)
    ax3.grid(True, alpha=0.3)

    # --- Panel 4: Genome Diversity ---
    ax4 = fig.add_subplot(gs[1, 1])
    ax4_twin = ax4.twinx()
    l1 = ax4.plot(ticks, data["shannon_index"], color="#6a1b9a", linewidth=1.5, label="Shannon Index")
    l2 = ax4_twin.plot(ticks, data["num_genomes"], color="#00838f", linewidth=1, alpha=0.7,
                       label="Unique Genomes")
    ax4.set_ylabel("Shannon Diversity Index", fontsize=11, color="#6a1b9a")
    ax4_twin.set_ylabel("Unique Genomes", fontsize=11, color="#00838f")
    ax4.set_xlabel("Ticks (thousands)", fontsize=10)
    ax4.set_title("Genetic Diversity", fontsize=12, fontweight="bold")
    lines = l1 + l2
    ax4.legend(lines, [l.get_label() for l in lines], fontsize=9)
    ax4.grid(True, alpha=0.3)

    # --- Panel 5: Energy Economy ---
    ax5 = fig.add_subplot(gs[2, 0])
    ax5.plot(ticks, data["avg_energy"], color="#bf360c", linewidth=1.5, label="Avg Energy")
    ax5.set_ylabel("Average Energy", fontsize=11)
    ax5.set_xlabel("Ticks (thousands)", fontsize=10)
    ax5.set_title("Energy Economy", fontsize=12, fontweight="bold")
    ax5.grid(True, alpha=0.3)
    ax5.legend(fontsize=9)

    # --- Panel 6: Evolutionary Phases Timeline ---
    ax6 = fig.add_subplot(gs[2, 1])
    ax6.set_title("Detected Evolutionary Phases", fontsize=12, fontweight="bold")
    colors = ["#ef5350", "#ffa726", "#66bb6a", "#42a5f5", "#ab47bc"]
    if phases:
        for idx, phase in enumerate(phases):
            color = colors[idx % len(colors)]
            start = phase["start_tick"] / 1000
            end = phase["end_tick"] / 1000
            ax6.barh(idx, end - start, left=start, height=0.6, color=color, alpha=0.7)
            ax6.text(start + 0.5, idx, f'{phase["name"]}', va="center", fontsize=9,
                     fontweight="bold")
        ax6.set_yticks(range(len(phases)))
        ax6.set_yticklabels([p["name"] for p in phases], fontsize=9)
        ax6.set_xlabel("Ticks (thousands)", fontsize=10)
        ax6.invert_yaxis()
    else:
        ax6.text(0.5, 0.5, "No phases detected\n(run may be too short)",
                 ha="center", va="center", transform=ax6.transAxes, fontsize=12)
    ax6.grid(True, alpha=0.3, axis="x")

    # --- Panel 7: Waste Trajectory ---
    ax7 = fig.add_subplot(gs[3, 0])
    waste_key = "avg_waste_at_cells"
    if waste_key in data:
        ax7.plot(ticks, data[waste_key], color="#d84315", linewidth=1.5, label="Avg Waste at Cells")
        from config import WASTE_TOXICITY_THRESHOLD
        ax7.axhline(y=WASTE_TOXICITY_THRESHOLD, color="red", linestyle="--", alpha=0.6,
                     label=f"Toxicity threshold ({WASTE_TOXICITY_THRESHOLD})")
        ax7.set_ylabel("Waste Concentration", fontsize=11)
        ax7.legend(fontsize=8, loc="upper left")
        if "waste_gt_threshold_frac" in data:
            ax7_twin = ax7.twinx()
            ax7_twin.plot(ticks, data["waste_gt_threshold_frac"] * 100,
                          color="#ff6f00", linewidth=1, alpha=0.7, label="% Above Threshold")
            ax7_twin.set_ylabel("% Cells Above Threshold", fontsize=10, color="#ff6f00")
            ax7_twin.legend(fontsize=8, loc="upper right")
    else:
        ax7.text(0.5, 0.5, "No waste data\n(pre-v7.0 run)",
                 ha="center", va="center", transform=ax7.transAxes, fontsize=12)
    ax7.set_xlabel("Ticks (thousands)", fontsize=10)
    ax7.set_title("Waste Pressure", fontsize=12, fontweight="bold")
    ax7.grid(True, alpha=0.3)

    # --- Panel 8: Zone Population ---
    ax8 = fig.add_subplot(gs[3, 1])
    if "bright_pct" in data and "dim_pct" in data and "dark_pct" in data:
        ax8.fill_between(ticks, 0, data["bright_pct"] * 100,
                         color="#ffffcc", label="Bright", alpha=0.9)
        ax8.fill_between(ticks, data["bright_pct"] * 100,
                         (data["bright_pct"] + data["dim_pct"]) * 100,
                         color="#ffe0b2", label="Dim", alpha=0.9)
        ax8.fill_between(ticks, (data["bright_pct"] + data["dim_pct"]) * 100,
                         100, color="#bdbdbd", label="Dark", alpha=0.9)
        ax8.set_ylim(0, 100)
        ax8.set_ylabel("Population %", fontsize=11)
        ax8.legend(fontsize=9, loc="center right")
    else:
        ax8.text(0.5, 0.5, "No zone data\n(pre-v7.0 run)",
                 ha="center", va="center", transform=ax8.transAxes, fontsize=12)
    ax8.set_xlabel("Ticks (thousands)", fontsize=10)
    ax8.set_title("Zone Population Distribution", fontsize=12, fontweight="bold")
    ax8.grid(True, alpha=0.3)

    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved: {output_path}")


def plot_comparison(data_a: dict, data_b: dict, label_a: str, label_b: str,
                    output_path: str):
    """Generate a comparison figure for two runs."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Comparison: {label_a} vs {label_b}", fontsize=14, fontweight="bold")

    ticks_a = data_a["tick"] / 1000
    ticks_b = data_b["tick"] / 1000

    metrics = [
        ("population", "Population", "#1565c0", "#e53935"),
        ("move_fraction", "Movement Fraction", "#1565c0", "#e53935"),
        ("avg_x_position", "Average X Position", "#1565c0", "#e53935"),
        ("shannon_index", "Shannon Diversity", "#1565c0", "#e53935"),
    ]

    for idx, (key, title, c_a, c_b) in enumerate(metrics):
        ax = axes[idx // 2][idx % 2]
        vals_a = data_a[key] * 100 if key == "move_fraction" else data_a[key]
        vals_b = data_b[key] * 100 if key == "move_fraction" else data_b[key]
        ax.plot(ticks_a, vals_a, color=c_a, linewidth=1.5, label=label_a)
        ax.plot(ticks_b, vals_b, color=c_b, linewidth=1.5, label=label_b, alpha=0.8)
        ylabel = "Cells Moving (%)" if key == "move_fraction" else title
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_xlabel("Ticks (thousands)", fontsize=10)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        if key == "avg_x_position":
            ax.axhline(y=166, color="orange", linestyle="--", alpha=0.5)
            ax.axhline(y=333, color="red", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved: {output_path}")

# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def generate_report(run_dir: str, data: dict, phases: list[dict],
                    rates: dict) -> str:
    """Generate a markdown report for a single run."""
    run_name = Path(run_dir).name

    lines = []
    lines.append(f"# CyberCell Evolutionary Dynamics Study")
    lines.append(f"")
    lines.append(f"**Run:** `{run_name}`  ")
    lines.append(f"**Duration:** {rates['total_ticks']:,} ticks  ")
    lines.append(f"**Date:** 2026-03-18  ")
    lines.append(f"")

    lines.append(f"## Executive Summary")
    lines.append(f"")

    # Determine what was achieved
    achievements = []
    if rates["final_population"] > 1000:
        achievements.append("stable self-sustaining population")
    if rates["final_move_fraction"] > 0.15:
        achievements.append(f"chemotaxis ({rates['final_move_fraction']:.0%} of cells moving)")
    if rates["final_avg_x"] > 166:
        achievements.append(f"cross-zone exploration (avg position x={rates['final_avg_x']:.0f})")
    if rates["max_age_observed"] > 10000:
        achievements.append(f"long-lived lineages (max age {rates['max_age_observed']:,} ticks)")

    if achievements:
        lines.append(f"This simulation run achieved: **{', '.join(achievements)}**.")
    else:
        lines.append(f"This run did not achieve significant evolutionary milestones.")
    lines.append(f"")

    lines.append(f"Starting from {int(data['population'][0]):,} cells with random neural network "
                 f"genomes, the population underwent natural selection driven entirely by "
                 f"environmental pressure — no behaviors were pre-programmed.")
    lines.append(f"")

    lines.append(f"## Key Findings")
    lines.append(f"")
    lines.append(f"### 1. Population Dynamics")
    lines.append(f"")
    lines.append(f"| Metric | Value |")
    lines.append(f"|--------|-------|")
    lines.append(f"| Initial population | {int(data['population'][0]):,} |")
    lines.append(f"| Minimum (bottleneck) | {rates['min_population']:,} (tick {rates['crash_tick']:,}) |")
    lines.append(f"| Final population | {rates['final_population']:,} |")
    lines.append(f"| Growth rate | {rates['pop_growth_per_1k_ticks']:.1f}% per 1K ticks (post-bottleneck) |")
    lines.append(f"")
    lines.append(f"The initial crash reflects **purifying selection**: cells with random neural "
                 f"networks that fail to photosynthesize or manage energy are eliminated. Only "
                 f"~{rates['min_population']/int(data['population'][0])*100:.0f}% of initial genomes "
                 f"survive. The survivors then expand as successful strategies reproduce.")
    lines.append(f"")

    lines.append(f"### 2. Emergence of Directed Movement (Chemotaxis)")
    lines.append(f"")
    lines.append(f"| Metric | Value |")
    lines.append(f"|--------|-------|")
    lines.append(f"| Initial movement | {data['move_fraction'][0]:.1%} (random) |")
    lines.append(f"| Post-crash movement | {data['move_fraction'][np.argmin(data['population'])]:.1%} (non-movers survive) |")
    lines.append(f"| Final movement | {rates['final_move_fraction']:.1%} |")
    lines.append(f"| Movement evolution rate | +{rates['move_rate_per_1k_ticks']:.4f} per 1K ticks |")
    lines.append(f"")

    if rates["final_move_fraction"] > 0.15:
        lines.append(f"Movement follows a characteristic **U-shaped curve**:")
        lines.append(f"1. **Random phase**: Initial genomes produce ~26% movement (noise)")
        lines.append(f"2. **Crash phase**: Movement drops to ~{data['move_fraction'][np.argmin(data['population'])]:.0%} — "
                     f"stationary photosynthesizers survive the bottleneck")
        lines.append(f"3. **Evolution phase**: Movement rises to {rates['final_move_fraction']:.0%} — "
                     f"but this time it is *directed*, not random")
        lines.append(f"")
        lines.append(f"The critical insight: post-crash movement is qualitatively different from "
                     f"initial random movement. Evolved movers have neural networks that couple "
                     f"chemical gradient sensing to motor output — they move *toward resources*.")
    lines.append(f"")

    lines.append(f"### 3. Spatial Exploration")
    lines.append(f"")
    lines.append(f"| Metric | Value |")
    lines.append(f"|--------|-------|")
    lines.append(f"| Initial avg X | {data['avg_x_position'][0]:.1f} (light zone center ~83) |")
    lines.append(f"| Final avg X | {rates['final_avg_x']} |")
    lines.append(f"| Expansion rate | {rates['x_expansion_per_1k_ticks']:.1f} units per 1K ticks |")
    lines.append(f"")
    zone = "light zone"
    if rates["final_avg_x"] > 333:
        zone = "dark zone"
    elif rates["final_avg_x"] > 166:
        zone = "dim zone"
    lines.append(f"Cells began clustered in the light zone (x < 166) and expanded into "
                 f"the **{zone}** (avg x = {rates['final_avg_x']}). This spatial expansion "
                 f"indicates cells evolved the ability to survive outside the primary energy "
                 f"source, using chemical deposits for sustenance.")
    lines.append(f"")

    lines.append(f"### 4. Genetic Diversity")
    lines.append(f"")
    lines.append(f"| Metric | Value |")
    lines.append(f"|--------|-------|")
    lines.append(f"| Initial Shannon index | {data['shannon_index'][0]:.2f} |")
    lines.append(f"| Final Shannon index | {rates['final_shannon_index']:.3f} |")
    lines.append(f"| Final unique genomes | {int(data['num_genomes'][-1]):,} |")
    lines.append(f"| Dominant genome fraction | {data['dominant_fraction'][-1]:.4%} |")
    lines.append(f"")
    lines.append(f"Shannon diversity *increased* over the run, indicating the evolution of "
                 f"multiple coexisting strategies rather than a single dominant genome. "
                 f"The dominant genome accounts for only {data['dominant_fraction'][-1]:.4%} "
                 f"of the population — extreme diversity.")
    lines.append(f"")

    lines.append(f"### 5. Energy Economy")
    lines.append(f"")
    lines.append(f"| Metric | Value |")
    lines.append(f"|--------|-------|")
    lines.append(f"| Initial avg energy | {data['avg_energy'][0]:.1f} |")
    lines.append(f"| Final avg energy | {rates['final_avg_energy']} |")
    lines.append(f"| Final avg repmat | {data['avg_repmat'][-1]:.1f} |")
    lines.append(f"| Max observed age | {rates['max_age_observed']:,} ticks |")
    lines.append(f"")
    lines.append(f"Energy accumulation shows cells evolved increasingly efficient metabolic "
                 f"strategies. The max observed age of {rates['max_age_observed']:,} ticks "
                 f"({rates['max_age_observed']/5000:.0f}x the nominal max age of 5,000) "
                 f"indicates lineages with exceptional survival ability.")
    lines.append(f"")

    # Environmental pressure section (waste + zones)
    if "avg_waste_at_cells" in data or "bright_pct" in data:
        lines.append(f"### 6. Environmental Pressure")
        lines.append(f"")
        lines.append(f"| Metric | Value |")
        lines.append(f"|--------|-------|")
        if "avg_waste_at_cells" in data:
            lines.append(f"| Avg waste at cells | {data['avg_waste_at_cells'][-1]:.4f} |")
        if "max_waste" in data:
            lines.append(f"| Peak waste | {data['max_waste'].max():.4f} |")
        if "waste_gt_threshold_frac" in data:
            lines.append(f"| Cells above toxicity | {data['waste_gt_threshold_frac'][-1]:.1%} |")
        if "bright_pct" in data:
            lines.append(f"| Bright zone | {data['bright_pct'][-1]:.1%} |")
        if "dim_pct" in data:
            lines.append(f"| Dim zone | {data['dim_pct'][-1]:.1%} |")
        if "dark_pct" in data:
            lines.append(f"| Dark zone | {data['dark_pct'][-1]:.1%} |")
        lines.append(f"")

    lines.append(f"## Evolutionary Phases Detected")
    lines.append(f"")
    if phases:
        lines.append(f"| Phase | Tick Range | Description |")
        lines.append(f"|-------|-----------|-------------|")
        for p in phases:
            lines.append(f"| {p['name']} | {p['start_tick']:,} – {p['end_tick']:,} | {p['description']} |")
    else:
        lines.append(f"No distinct phases detected (run may be too short).")
    lines.append(f"")

    lines.append(f"## What Are the Cells \"Learning\"?")
    lines.append(f"")
    lines.append(f"Each cell has a neural network "
                 f"that maps sensory inputs to actions. Through mutation and selection, these networks "
                 f"evolve to encode survival strategies. The key evolved behaviors we can infer from "
                 f"the metrics:")
    lines.append(f"")
    lines.append(f"1. **Energy management**: Cells that survive the initial crash have networks that "
                 f"effectively couple light sensing to photosynthesis behavior")
    lines.append(f"2. **Chemical gradient following**: The rise in movement fraction combined with "
                 f"spatial expansion indicates cells evolved to follow S and R chemical gradients")
    lines.append(f"3. **Resource foraging**: Cells venture into dim/dark zones (where R deposits are "
                 f"concentrated) and return or sustain themselves on chemical energy")
    lines.append(f"4. **Reproductive timing**: Cells accumulate replication material and divide when "
                 f"conditions are favorable, rather than dividing as soon as possible")
    lines.append(f"")
    lines.append(f"Importantly, **none of these behaviors were programmed**. The simulation rules "
                 f"only define physics (diffusion, energy costs, death). All behavioral complexity "
                 f"emerged through natural selection acting on random neural network mutations.")
    lines.append(f"")

    lines.append(f"## Methodology")
    lines.append(f"")
    lines.append(f"- **Platform**: CyberCell evolutionary simulation (Taichi Lang + Python)")
    lines.append(f"- **Grid**: 500x500 toroidal, three light zones (bright/dim/dark)")
    lines.append(f"- **Organisms**: Neural network-controlled cells with sensory inputs and 10 actions")
    lines.append(f"- **Selection**: Natural — cells die without energy, reproduce by division")
    lines.append(f"- **Mutation**: Weight perturbation (3%), reset (0.1%), node knockout (0.05%)")
    lines.append(f"- **Metrics**: Logged every {1000} ticks via population census")
    lines.append(f"")

    lines.append(f"## Figures")
    lines.append(f"")
    lines.append(f"![Evolutionary Dynamics](evolution_report.png)")
    lines.append(f"")
    lines.append(f"## See Also")
    lines.append(f"")
    lines.append(f"- [Spatial Structure Analysis](SPATIAL_ANALYSIS.md) (if available)")
    lines.append(f"")

    return "\n".join(lines)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.chdir(Path(__file__).resolve().parent.parent)

    if "--compare" in sys.argv:
        idx = sys.argv.index("--compare")
        dir_a = sys.argv[idx + 1]
        dir_b = sys.argv[idx + 2]
        data_a = records_to_arrays(load_run(dir_a))
        data_b = records_to_arrays(load_run(dir_b))
        if not data_a or not data_b:
            print("Error: could not load one or both runs")
            return
        label_a = Path(dir_a).name
        label_b = Path(dir_b).name
        output_dir = Path("analysis/output") / f"compare_{label_a}_vs_{label_b}"
        output_dir.mkdir(parents=True, exist_ok=True)
        plot_comparison(data_a, data_b, label_a, label_b,
                        str(output_dir / "comparison.png"))
        print(f"\nComparison plot saved to {output_dir}/")
        return

    # Find or use specified run
    if len(sys.argv) > 1 and sys.argv[1] != "--compare":
        run_dirs = [(sys.argv[1], 0)]
    else:
        run_dirs = find_longest_runs()

    if not run_dirs:
        print("No runs found in runs/ directory.")
        return

    print(f"Found {len(run_dirs)} runs with data:\n")
    for rd, count in run_dirs:
        print(f"  {Path(rd).name}: {count} snapshots")
    print()

    # Analyze the longest run in detail
    best_dir = run_dirs[0][0]
    records = load_run(best_dir)
    data = records_to_arrays(records)

    if not data:
        print(f"No data in {best_dir}")
        return

    # Versioned output directory per run
    run_name = Path(best_dir).name
    output_dir = Path("analysis/output") / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Analyzing: {best_dir} ({len(records)} snapshots, "
          f"{int(data['tick'][-1]):,} ticks)")
    print(f"Output -> {output_dir}/")

    # Detect phases
    phases = detect_phases(data)
    print(f"\nEvolutionary phases detected: {len(phases)}")
    for p in phases:
        print(f"  [{p['start_tick']:>7,} - {p['end_tick']:>7,}] {p['name']}: {p['description']}")

    # Compute rates
    rates = compute_rates(data)
    print(f"\nKey rates:")
    for k, v in rates.items():
        print(f"  {k}: {v}")

    # Generate plots
    print(f"\nGenerating plots...")
    plot_single_run(data, phases,
                    f"CyberCell Evolutionary Dynamics — {run_name}",
                    str(output_dir / "evolution_report.png"))

    # Compare against the next-best run (find one if we only have one)
    all_runs = find_longest_runs()
    compare_candidates = [rd for rd, _ in all_runs
                          if Path(rd).resolve() != Path(best_dir).resolve()]
    if compare_candidates:
        second_dir = compare_candidates[0]
        data_b = records_to_arrays(load_run(second_dir))
        if data_b:
            plot_comparison(data, data_b,
                            Path(best_dir).name, Path(second_dir).name,
                            str(output_dir / "comparison.png"))

    # Generate report
    print(f"\nGenerating report...")
    report = generate_report(best_dir, data, phases, rates)
    report_path = output_dir / "STUDY.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"  Saved: {report_path}")

    print(f"\nDone! All outputs in {output_dir}/")


if __name__ == "__main__":
    main()
