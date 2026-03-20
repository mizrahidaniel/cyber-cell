"""Unified analysis CLI -- runs all applicable analyses on a simulation run.

Detects available data files and runs applicable analyses in order:
1. study  2. crn_analysis  3. lineage  4. spatial  5. bonding  6. burst

Usage:
    python analysis/run_all.py                      # latest run
    python analysis/run_all.py runs/20260319_123456 # specific run
"""

import glob
import os
import subprocess
import sys


def find_latest_run(runs_dir="runs"):
    """Find the most recent run directory with metrics data."""
    if not os.path.exists(runs_dir):
        return None
    dirs = sorted(glob.glob(os.path.join(runs_dir, "*")), reverse=True)
    for d in dirs:
        if os.path.isfile(os.path.join(d, "metrics.jsonl")):
            return d
    return None


def _run_script(name: str, script: str, run_dir: str) -> bool:
    """Run an analysis script as a subprocess."""
    try:
        result = subprocess.run(
            [sys.executable, script, run_dir],
            capture_output=True, text=True, timeout=120,
        )
        if result.returncode == 0:
            for line in result.stdout.strip().splitlines()[-3:]:
                print(f"  {line}")
            return True
        else:
            err = result.stderr.strip().splitlines()
            print(f"  Failed: {err[-1] if err else 'unknown error'}")
            return False
    except subprocess.TimeoutExpired:
        print(f"  Timed out after 120s")
        return False
    except Exception as e:
        print(f"  Error: {e}")
        return False


def main():
    if len(sys.argv) > 1:
        run_dir = sys.argv[1]
    else:
        run_dir = find_latest_run()

    if not run_dir or not os.path.isdir(run_dir):
        print("No runs found. Run a simulation first.")
        sys.exit(1)

    run_name = os.path.basename(run_dir)
    print(f"{'='*60}")
    print(f"  Unified Analysis: {run_name}")
    print(f"  Run directory: {run_dir}")
    print(f"{'='*60}\n")

    # Detect available data
    has = {
        "metrics": os.path.isfile(os.path.join(run_dir, "metrics.jsonl")),
        "crn": os.path.isfile(os.path.join(run_dir, "crn_metrics.jsonl")),
        "lineage": os.path.isfile(os.path.join(run_dir, "lineage.jsonl")),
        "spatial": os.path.isdir(os.path.join(run_dir, "spatial")),
        "burst": bool(glob.glob(os.path.join(run_dir, "burst", "burst_*"))),
    }

    print("Available data:")
    for name, flag in has.items():
        print(f"  {'[x]' if flag else '[ ]'} {name}")
    print()

    # Analysis pipeline: (name, script_path, required_data_key)
    scripts = [
        ("Study",    "analysis/study.py",            "metrics"),
        ("CRN",      "analysis/crn_analysis.py",     "crn"),
        ("Lineage",  "analysis/lineage_analysis.py",  "lineage"),
        ("Spatial",  "analysis/spatial_analysis.py",  "spatial"),
        ("Bonding",  "analysis/bonding_analysis.py",  "spatial"),
        ("Burst",    "analysis/burst_analysis.py",    "burst"),
    ]

    completed = 0
    for i, (name, script, req) in enumerate(scripts, 1):
        if not has.get(req, False):
            print(f"--- [{i}/{len(scripts)}] {name} Analysis --- "
                  f"skipped (no {req} data)\n")
            continue

        if not os.path.isfile(script):
            print(f"--- [{i}/{len(scripts)}] {name} Analysis --- "
                  f"skipped ({script} not found)\n")
            continue

        print(f"--- [{i}/{len(scripts)}] {name} Analysis ---")
        if _run_script(name, script, run_dir):
            completed += 1
        print()

    print(f"{'='*60}")
    print(f"  Completed {completed}/{len(scripts)} analyses")
    print(f"  Output: analysis/output/{run_name}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
