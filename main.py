"""CyberCell: Evolutionary Intelligence Simulation — Entry Point."""

import argparse
import json
import os
import subprocess
import sys
import time

import taichi as ti

from config import BACKEND, HEADLESS, RANDOM_SEED

CACHE_FILE = os.path.join(os.path.dirname(__file__), ".backend_cache.json")

# Standalone benchmark script run as a subprocess per backend.
_BENCH_SCRIPT = r'''
import sys, time, os
os.environ["TI_LOG_LEVEL"] = "warn"
import taichi as ti
backend = sys.argv[1]
arch = {"cpu": ti.cpu, "cuda": ti.cuda, "metal": ti.metal, "vulkan": ti.vulkan}[backend]
try:
    ti.init(arch=arch, random_seed=42)
except Exception:
    print("FAIL", flush=True)
    sys.exit(0)
from simulation.engine import SimulationEngine
engine = SimulationEngine(headless=True, auto_switch=False)
engine.init()
warmup = 50
bench = 500
for _ in range(warmup):
    engine.step()
ti.sync()
start = time.perf_counter()
for _ in range(bench):
    engine.step()
ti.sync()
elapsed = time.perf_counter() - start
tps = bench / elapsed
print(f"OK {tps:.1f}", flush=True)
'''


def _benchmark_backend(backend: str) -> float | None:
    """Run a short benchmark of the given backend in a subprocess. Returns ticks/sec or None."""
    try:
        result = subprocess.run(
            [sys.executable, "-c", _BENCH_SCRIPT, backend],
            capture_output=True, text=True, timeout=120,
            cwd=os.path.dirname(__file__) or ".",
        )
        for line in result.stdout.strip().splitlines():
            if line.startswith("OK "):
                return float(line.split()[1])
    except (subprocess.TimeoutExpired, Exception):
        pass
    return None


def auto_select_backend(force_rebench: bool = False) -> str:
    """Pick the fastest backend, using a cached result if available."""
    if not force_rebench and os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE) as f:
                data = json.load(f)
            print(f"[auto] Using cached backend: {data['winner']} "
                  f"({data['winner_tps']:.0f} ticks/s vs {data.get('results', {})})")
            return data["winner"]
        except (json.JSONDecodeError, KeyError):
            pass

    candidates = ["cpu"]
    # Only benchmark CUDA/Metal if likely available
    if sys.platform == "win32" or os.path.exists("/usr/bin/nvidia-smi"):
        candidates.append("cuda")
    if sys.platform == "darwin":
        candidates.append("metal")

    print(f"[auto] Benchmarking backends: {candidates} (one-time, ~30s per backend)...")
    results = {}
    for backend in candidates:
        print(f"[auto]   {backend}: ", end="", flush=True)
        tps = _benchmark_backend(backend)
        if tps is not None:
            results[backend] = tps
            print(f"{tps:.1f} ticks/sec")
        else:
            print("unavailable")

    if not results:
        print("[auto] No backends succeeded, falling back to cpu")
        return "cpu"

    winner = max(results, key=results.get)

    # Cache the result
    try:
        with open(CACHE_FILE, "w") as f:
            json.dump({"winner": winner, "winner_tps": results[winner],
                        "results": results}, f)
    except OSError:
        pass

    print(f"[auto] Selected: {winner} ({results[winner]:.1f} ticks/sec)")
    return winner


def parse_args():
    parser = argparse.ArgumentParser(description="CyberCell evolutionary simulation")
    parser.add_argument("--backend", default=BACKEND,
                        choices=["auto", "cpu", "metal", "cuda"],
                        help="Taichi compute backend (default: auto)")
    parser.add_argument("--rebenchmark", action="store_true",
                        help="Force re-run of auto backend benchmark")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from a checkpoint file")
    parser.add_argument("--no-auto-switch", action="store_true",
                        help="Disable automatic backend switching during simulation")
    parser.add_argument("--headless", action="store_true", default=HEADLESS,
                        help="Run without visualization")
    parser.add_argument("--ticks", type=int, default=0,
                        help="Number of ticks to run (0 = unlimited)")
    parser.add_argument("--genome", default=None, choices=["neural", "crn"],
                        help="Genome type (default: use config.py setting)")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED,
                        help="Random seed for reproducibility")
    parser.add_argument("--log-interval", type=int, default=1000,
                        help="Ticks between console progress prints (headless mode)")
    return parser.parse_args()


def get_arch(backend: str):
    return {"cpu": ti.cpu, "metal": ti.metal, "cuda": ti.cuda}[backend]


def main():
    args = parse_args()

    if args.genome:
        import config
        config.GENOME_TYPE = args.genome

    backend = args.backend
    if backend == "auto":
        backend = auto_select_backend(force_rebench=args.rebenchmark)

    ti.init(arch=get_arch(backend), random_seed=args.seed)

    from simulation.engine import SimulationEngine

    engine = SimulationEngine(
        headless=args.headless,
        log_interval=args.log_interval,
        backend=backend,
        auto_switch=not args.no_auto_switch,
    )

    if args.resume:
        engine.resume(args.resume)
    else:
        engine.init()

    engine.run(max_ticks=args.ticks)


if __name__ == "__main__":
    main()
