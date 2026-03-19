"""Periodic simulation snapshots to disk for offline analysis."""

import json
import os
import time

from analysis.metrics import get_population_stats, get_genome_diversity, get_movement_stats


class SimulationLogger:
    def __init__(self, run_dir: str = "runs"):
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.log_dir = os.path.join(run_dir, timestamp)
        os.makedirs(self.log_dir, exist_ok=True)
        self.log_path = os.path.join(self.log_dir, "metrics.jsonl")
        self._file = open(self.log_path, "w")

    def snapshot(self, tick: int):
        """Record a snapshot of simulation metrics."""
        pop_stats = get_population_stats()
        div_stats = get_genome_diversity()
        move_stats = get_movement_stats()

        record = {
            "tick": tick,
            **pop_stats,
            **div_stats,
            **move_stats,
        }

        self._file.write(json.dumps(record) + "\n")
        self._file.flush()

    def close(self):
        self._file.close()
