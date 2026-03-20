"""
CyberCell Burst Snapshot Analysis
==================================
Analyzes burst snapshots (20 consecutive frames) to detect frame-by-frame
cell movement, division events, and death events.

Usage:
    python analysis/burst_analysis.py                      # analyze latest run
    python analysis/burst_analysis.py runs/20260319_123456 # specific run
"""

import json
import os
import sys
import glob
from pathlib import Path

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    from matplotlib.colors import Normalize
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_burst(burst_dir: str) -> list[dict]:
    """Load all frames from a burst directory."""
    frames = []
    frame_files = sorted(glob.glob(os.path.join(burst_dir, "frame_*.npz")))
    for f in frame_files:
        data = dict(np.load(f))
        frames.append(data)
    return frames


def find_all_bursts(run_dir: str) -> list[tuple[int, str]]:
    """Find all burst directories in a run, sorted by tick."""
    burst_root = os.path.join(run_dir, "burst")
    if not os.path.isdir(burst_root):
        return []
    results = []
    for d in sorted(os.listdir(burst_root)):
        if d.startswith("burst_"):
            tick = int(d.split("_")[1])
            results.append((tick, os.path.join(burst_root, d)))
    return results


# ---------------------------------------------------------------------------
# Frame-to-frame analysis
# ---------------------------------------------------------------------------

def build_position_map(positions: np.ndarray, genome_ids: np.ndarray) -> dict:
    """Build (x, y) -> index mapping for fast lookup."""
    pos_map = {}
    for i in range(len(positions)):
        key = (int(positions[i, 0]), int(positions[i, 1]))
        pos_map[key] = i
    return pos_map


def analyze_frame_pair(frame_a: dict, frame_b: dict) -> dict:
    """Compute delta between two consecutive frames.

    Returns movement vectors, new cells (divisions), and disappeared cells (deaths).
    """
    pos_a = frame_a["positions"]
    pos_b = frame_b["positions"]
    gids_a = frame_a["genome_ids"]
    gids_b = frame_b["genome_ids"]

    # Build position sets
    set_a = set(map(tuple, pos_a.tolist()))
    set_b = set(map(tuple, pos_b.tolist()))

    # Cells that appeared (potential divisions/births)
    appeared = set_b - set_a
    # Cells that disappeared (potential deaths)
    disappeared = set_a - set_b
    # Cells present in both frames (potential movers or stationary)
    stable = set_a & set_b

    # For appeared cells, check if they're adjacent to a cell in frame_a
    # (suggesting division)
    divisions = []
    for pos in appeared:
        x, y = pos
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            neighbor = ((x + dx) % 500, (y + dy) % 500)
            if neighbor in set_a:
                divisions.append({"new_pos": pos, "parent_pos": neighbor})
                break

    return {
        "n_cells_a": len(pos_a),
        "n_cells_b": len(pos_b),
        "n_appeared": len(appeared),
        "n_disappeared": len(disappeared),
        "n_stable": len(stable),
        "n_divisions": len(divisions),
        "appeared": list(appeared),
        "disappeared": list(disappeared),
        "divisions": divisions,
    }


def analyze_burst(frames: list[dict]) -> list[dict]:
    """Analyze all consecutive frame pairs in a burst."""
    deltas = []
    for i in range(len(frames) - 1):
        delta = analyze_frame_pair(frames[i], frames[i + 1])
        delta["frame_pair"] = (i, i + 1)
        deltas.append(delta)
    return deltas


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_filmstrip(frames: list[dict], deltas: list[dict],
                   burst_tick: int, output_dir: str, run_name: str):
    """Generate a filmstrip of burst frames with movement overlay."""
    if not HAS_MPL:
        print("  matplotlib not available, skipping plots")
        return

    n_frames = len(frames)
    cols = min(5, n_frames)
    rows = (n_frames + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    fig.suptitle(f"Burst Filmstrip — {run_name}\n"
                 f"Starting tick {burst_tick:,} ({n_frames} frames)",
                 fontsize=13, fontweight="bold")

    if rows == 1:
        axes = [axes] if cols == 1 else list(axes)
    else:
        axes = [ax for row in axes for ax in row]

    for i, frame in enumerate(frames):
        ax = axes[i]
        pos = frame["positions"]
        energies = frame.get("energies", np.ones(len(pos)))

        ax.set_facecolor("black")
        if len(pos) > 0:
            # Color by energy (green = high, red = low)
            norm_e = np.clip(energies / 100.0, 0, 1)
            colors = plt.cm.RdYlGn(norm_e)
            ax.scatter(pos[:, 0], pos[:, 1], s=0.3, c=colors, alpha=0.6)

            # Overlay division events from this frame's delta
            if i < len(deltas):
                delta = deltas[i]
                for div in delta["divisions"]:
                    nx, ny = div["new_pos"]
                    ax.plot(nx, ny, "c*", ms=4, alpha=0.8)
                # Mark disappeared cells
                for dx, dy in delta["disappeared"]:
                    ax.plot(dx, dy, "rx", ms=3, alpha=0.6)

        ax.set_xlim(0, 500)
        ax.set_ylim(0, 500)
        ax.set_aspect("equal")
        pop = len(pos)
        title = f"Frame {i} (n={pop})"
        if i < len(deltas):
            title += f"\n+{deltas[i]['n_appeared']}/-{deltas[i]['n_disappeared']}"
        ax.set_title(title, fontsize=8)
        ax.tick_params(labelsize=6)

    # Hide unused axes
    for i in range(n_frames, len(axes)):
        axes[i].set_visible(False)

    out_path = os.path.join(output_dir, f"burst_filmstrip_{burst_tick:08d}.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="black")
    plt.close()
    print(f"  Saved: {out_path}")


def plot_movement_trails(frames: list[dict], deltas: list[dict],
                         burst_tick: int, output_dir: str, run_name: str):
    """Plot aggregate movement trails across all burst frames."""
    if not HAS_MPL:
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f"Burst Movement Analysis — tick {burst_tick:,}",
                 fontsize=13, fontweight="bold")

    # Panel 1: All divisions (new cells marked)
    ax = axes[0]
    ax.set_title("Division events", fontsize=11)
    ax.set_facecolor("black")
    # Show first frame as background
    if frames:
        pos = frames[0]["positions"]
        if len(pos) > 0:
            ax.scatter(pos[:, 0], pos[:, 1], s=0.1, c="gray", alpha=0.2)
    # Mark all divisions across burst
    all_divs = []
    for delta in deltas:
        for div in delta["divisions"]:
            all_divs.append(div["new_pos"])
    if all_divs:
        div_arr = np.array(all_divs)
        ax.scatter(div_arr[:, 0], div_arr[:, 1], s=8, c="cyan", marker="*",
                   alpha=0.8, label=f"{len(all_divs)} divisions")
        ax.legend(fontsize=8, facecolor="black", labelcolor="white")
    ax.set_xlim(0, 500)
    ax.set_ylim(0, 500)
    ax.set_aspect("equal")

    # Panel 2: All death locations
    ax = axes[1]
    ax.set_title("Death events", fontsize=11)
    ax.set_facecolor("black")
    if frames:
        pos = frames[0]["positions"]
        if len(pos) > 0:
            ax.scatter(pos[:, 0], pos[:, 1], s=0.1, c="gray", alpha=0.2)
    all_deaths = []
    for delta in deltas:
        all_deaths.extend(delta["disappeared"])
    if all_deaths:
        death_arr = np.array(all_deaths)
        ax.scatter(death_arr[:, 0], death_arr[:, 1], s=5, c="red", marker="x",
                   alpha=0.6, label=f"{len(all_deaths)} deaths")
        ax.legend(fontsize=8, facecolor="black", labelcolor="white")
    ax.set_xlim(0, 500)
    ax.set_ylim(0, 500)
    ax.set_aspect("equal")

    # Panel 3: Population change over frames
    ax = axes[2]
    ax.set_title("Population across frames", fontsize=11)
    pops = [len(f["positions"]) for f in frames]
    ax.plot(range(len(pops)), pops, "o-", color="#42a5f5", ms=4)
    ax.set_xlabel("Frame")
    ax.set_ylabel("Population")
    ax.grid(True, alpha=0.3)

    # Add stats text
    if deltas:
        total_appeared = sum(d["n_appeared"] for d in deltas)
        total_disappeared = sum(d["n_disappeared"] for d in deltas)
        total_divs = sum(d["n_divisions"] for d in deltas)
        ax.text(0.02, 0.98,
                f"Total appeared: {total_appeared}\n"
                f"Total disappeared: {total_disappeared}\n"
                f"Detected divisions: {total_divs}\n"
                f"Net change: {pops[-1] - pops[0]:+d}",
                transform=ax.transAxes, va="top", fontsize=8,
                bbox=dict(boxstyle="round", fc="lightyellow"))

    out_path = os.path.join(output_dir, f"burst_movement_{burst_tick:08d}.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def generate_burst_report(bursts_analyzed: list[dict], run_name: str) -> str:
    """Generate markdown report for all burst analyses."""
    lines = []
    lines.append("# Burst Snapshot Analysis")
    lines.append("")
    lines.append(f"**Run:** `{run_name}`  ")
    lines.append(f"**Bursts analyzed:** {len(bursts_analyzed)}  ")
    lines.append("")

    for burst_info in bursts_analyzed:
        tick = burst_info["tick"]
        deltas = burst_info["deltas"]
        n_frames = burst_info["n_frames"]

        lines.append(f"## Burst at tick {tick:,}")
        lines.append("")
        lines.append(f"**Frames:** {n_frames}  ")

        if deltas:
            total_appeared = sum(d["n_appeared"] for d in deltas)
            total_disappeared = sum(d["n_disappeared"] for d in deltas)
            total_divs = sum(d["n_divisions"] for d in deltas)
            avg_pop = (deltas[0]["n_cells_a"] + deltas[-1]["n_cells_b"]) / 2

            lines.append("")
            lines.append(f"| Metric | Value |")
            lines.append(f"|--------|-------|")
            lines.append(f"| Avg population | {avg_pop:.0f} |")
            lines.append(f"| Total cells appeared | {total_appeared} |")
            lines.append(f"| Total cells disappeared | {total_disappeared} |")
            lines.append(f"| Detected divisions | {total_divs} |")
            lines.append(f"| Net population change | {deltas[-1]['n_cells_b'] - deltas[0]['n_cells_a']:+d} |")
            lines.append(f"| Avg turnover per frame | {(total_appeared + total_disappeared) / len(deltas) / 2:.1f} |")
            lines.append("")

            # Per-frame breakdown
            lines.append("### Frame-by-frame")
            lines.append("")
            lines.append("| Pair | Pop A | Pop B | Appeared | Disappeared | Divisions |")
            lines.append("|------|-------|-------|----------|-------------|-----------|")
            for d in deltas:
                pair = f"{d['frame_pair'][0]}->{d['frame_pair'][1]}"
                lines.append(f"| {pair} | {d['n_cells_a']} | {d['n_cells_b']} | "
                             f"{d['n_appeared']} | {d['n_disappeared']} | {d['n_divisions']} |")
            lines.append("")

        lines.append(f"![Filmstrip](burst_filmstrip_{tick:08d}.png)")
        lines.append("")
        lines.append(f"![Movement](burst_movement_{tick:08d}.png)")
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def find_latest_run_with_burst(runs_dir="runs"):
    """Find the most recent run that has burst data."""
    run_dirs = sorted(glob.glob(os.path.join(runs_dir, "*")), reverse=True)
    for d in run_dirs:
        burst_dir = os.path.join(d, "burst")
        if os.path.isdir(burst_dir):
            subdirs = [x for x in os.listdir(burst_dir) if x.startswith("burst_")]
            if subdirs:
                return d
    return None


def main():
    if len(sys.argv) > 1:
        run_dir = sys.argv[1]
    else:
        run_dir = find_latest_run_with_burst()

    if not run_dir:
        print("No runs with burst data found.")
        print("Run a simulation first — burst snapshots are captured every "
              "BURST_SNAPSHOT_INTERVAL ticks.")
        return

    run_name = os.path.basename(run_dir)
    output_dir = os.path.join("analysis", "output", run_name)
    os.makedirs(output_dir, exist_ok=True)

    print(f"Analyzing bursts: {run_dir}")
    print(f"Output -> {output_dir}/")

    bursts = find_all_bursts(run_dir)
    print(f"  Found {len(bursts)} burst captures")

    bursts_analyzed = []

    for tick, burst_dir in bursts:
        print(f"\n  Burst at tick {tick:,}:")
        frames = load_burst(burst_dir)
        print(f"    {len(frames)} frames loaded")

        if len(frames) < 2:
            print("    Skipping (need at least 2 frames)")
            continue

        deltas = analyze_burst(frames)

        total_appeared = sum(d["n_appeared"] for d in deltas)
        total_disappeared = sum(d["n_disappeared"] for d in deltas)
        total_divs = sum(d["n_divisions"] for d in deltas)
        print(f"    Appeared: {total_appeared}, Disappeared: {total_disappeared}, "
              f"Divisions: {total_divs}")

        burst_info = {
            "tick": tick,
            "n_frames": len(frames),
            "deltas": deltas,
        }
        bursts_analyzed.append(burst_info)

        # Generate plots
        if HAS_MPL:
            plot_filmstrip(frames, deltas, tick, output_dir, run_name)
            plot_movement_trails(frames, deltas, tick, output_dir, run_name)

    # Generate report
    if bursts_analyzed:
        print("\nGenerating report...")
        report = generate_burst_report(bursts_analyzed, run_name)
        report_path = os.path.join(output_dir, "BURST_ANALYSIS.md")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"  Saved: {report_path}")
    else:
        print("\nNo burst data to analyze.")

    print(f"\nDone! All outputs in {output_dir}/")


if __name__ == "__main__":
    main()
