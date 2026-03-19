"""CyberCell: Evolutionary Intelligence Simulation — Entry Point."""

import argparse
import taichi as ti

from config import BACKEND, HEADLESS, RANDOM_SEED


def parse_args():
    parser = argparse.ArgumentParser(description="CyberCell evolutionary simulation")
    parser.add_argument("--backend", default=BACKEND,
                        choices=["cpu", "metal", "cuda"],
                        help="Taichi compute backend")
    parser.add_argument("--headless", action="store_true", default=HEADLESS,
                        help="Run without visualization")
    parser.add_argument("--ticks", type=int, default=0,
                        help="Number of ticks to run (0 = unlimited)")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED,
                        help="Random seed for reproducibility")
    parser.add_argument("--log-interval", type=int, default=1000,
                        help="Ticks between console progress prints (headless mode)")
    return parser.parse_args()


def get_arch(backend: str):
    return {"cpu": ti.cpu, "metal": ti.metal, "cuda": ti.cuda}[backend]


def main():
    args = parse_args()
    ti.init(arch=get_arch(args.backend), random_seed=args.seed)

    from simulation.engine import SimulationEngine

    engine = SimulationEngine(headless=args.headless, log_interval=args.log_interval)
    engine.init()
    engine.run(max_ticks=args.ticks)


if __name__ == "__main__":
    main()
