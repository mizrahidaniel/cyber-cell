"""Main simulation engine — orchestrates the tick loop."""

import json
import os
import subprocess
import sys

import numpy as np

from config import GENOME_GC_INTERVAL, RANDOM_SEED, SNAPSHOT_INTERVAL, DEPOSIT_RELOCATE_INTERVAL

from world.grid import compute_light, init_grid
from world.chemistry import (
    diffuse_all, replenish_deposits, swap_buffers, init_chemistry,
    get_env_S, get_env_R, get_env_G,
    get_current_buffer, set_current_buffer, relocate_deposits,
)
from cell.cell_state import init_cell_state, cell_count
from cell.lifecycle import photosynthesis, eat_passive, apply_metabolism, check_death
from cell.genome import (
    init_genome_table, evaluate_all_networks, process_mutations,
    garbage_collect_genomes, genome_count, get_mutation_events,
)
from cell.sensing import compute_sensory_inputs
from cell.actions import (
    clear_intentions, process_turns,
    process_movement_phase1, process_movement_phase2,
    process_eat, process_emit_signal, process_repair, process_attack,
    process_divide_phase1, process_divide_phase2,
    process_bond_signal_output,
)
from cell.bonding import (
    process_bond, process_unbond, process_bond_sharing,
    process_bonded_movement, process_bond_strength_update,
    process_bond_signal_relay,
)
from simulation.spawner import seed_cells

# Auto-switch thresholds (benchmarked: CUDA wins up to ~10K, CPU wins above ~15K)
_SWITCH_CHECK_INTERVAL = 500      # ticks between population checks
_CUDA_TO_CPU_THRESHOLD = 12000    # population above this → prefer CPU (GPU transfer bottleneck)
_CPU_TO_CUDA_THRESHOLD = 8000     # population below this → prefer CUDA
_SWITCH_CONFIRM_CHECKS = 3        # consecutive checks before triggering switch
_CHECKPOINT_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                ".switch_checkpoint.npz")
_CACHE_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                           ".backend_cache.json")


class SimulationEngine:
    def __init__(self, headless: bool = False, log_interval: int = 1000,
                 backend: str = "cpu", auto_switch: bool = True):
        self.headless = headless
        self.log_interval = log_interval
        self.tick_count = 0
        self.mutation_rng = np.random.default_rng(RANDOM_SEED + 100)
        self.renderer = None
        self.logger = None
        self.backend = backend
        self.auto_switch = auto_switch and backend in ("cpu", "cuda")
        self._switch_streak = 0  # consecutive checks wanting a switch

    def init(self):
        """Initialize all simulation subsystems from scratch."""
        init_grid()
        init_chemistry()
        init_cell_state()
        init_genome_table()
        seed_cells()

        from analysis.logger import SimulationLogger
        self.logger = SimulationLogger()

        if not self.headless:
            from visualization.renderer import Renderer
            self.renderer = Renderer()

    def resume(self, checkpoint_path: str):
        """Resume simulation from a checkpoint file."""
        from simulation.checkpoint import load_checkpoint

        # Fields are allocated at import time, just load data into them
        meta = load_checkpoint(checkpoint_path)
        self.tick_count = meta["tick"]
        set_current_buffer(meta["current_buffer"])

        # Restore RNG state
        rng_state = meta["mutation_rng_state"]
        if rng_state is not None:
            self.mutation_rng = np.random.default_rng()
            self.mutation_rng.bit_generator.state = rng_state.item()

        from analysis.logger import SimulationLogger
        self.logger = SimulationLogger()

        if not self.headless:
            from visualization.renderer import Renderer
            self.renderer = Renderer()

        pop = cell_count[None]
        print(f"[resume] Restored from tick {self.tick_count}, population {pop}, backend {self.backend}")

    def _desired_backend(self) -> str | None:
        """Check if we should switch backends based on current population.

        CUDA wins up to ~10K cells (GPU parallelism helps).
        CPU wins above ~15K cells (GPU<->CPU transfer for mutations dominates).
        Hysteresis band 8K-12K prevents flip-flopping.
        """
        if not self.auto_switch:
            return None

        pop = cell_count[None]

        if self.backend == "cuda" and pop >= _CUDA_TO_CPU_THRESHOLD:
            return "cpu"
        elif self.backend == "cpu" and pop <= _CPU_TO_CUDA_THRESHOLD:
            return "cuda"
        return None

    def _do_backend_switch(self, new_backend: str):
        """Save state and re-launch with the new backend."""
        from simulation.checkpoint import save_checkpoint

        pop = cell_count[None]
        print(f"[auto-switch] Population {pop}: switching {self.backend} -> {new_backend}")

        # Save checkpoint
        rng_state = self.mutation_rng.bit_generator.state
        save_checkpoint(_CHECKPOINT_FILE, self.tick_count,
                        get_current_buffer(), rng_state)

        # Update the cached backend preference
        try:
            with open(_CACHE_FILE, "w") as f:
                json.dump({"winner": new_backend, "winner_tps": 0,
                           "results": {}, "auto_switched": True}, f)
        except OSError:
            pass

        # Build args to relaunch
        args = [sys.executable, "-u", os.path.join(os.path.dirname(os.path.dirname(__file__)), "main.py"),
                "--backend", new_backend,
                "--resume", _CHECKPOINT_FILE]
        if self.headless:
            args.append("--headless")

        # Close GUI before restart
        if self.renderer:
            self.renderer.close()
            self.renderer = None

        print(f"[auto-switch] Restarting...")
        os.execv(sys.executable, args)

    def step(self):
        """Execute one simulation tick."""
        # 1. Update environment
        compute_light(self.tick_count)
        diffuse_all()
        replenish_deposits()

        env_S = get_env_S()
        env_R = get_env_R()
        env_G = get_env_G()

        # 2. Passive processes (always on, not gated by neural net)
        photosynthesis(env_S, env_R)
        eat_passive(env_S, env_R)

        # 3. Sense -> Think -> Act
        compute_sensory_inputs(env_S, env_R, env_G)
        evaluate_all_networks()

        clear_intentions()
        process_turns()
        process_movement_phase1()
        process_movement_phase2()
        process_bonded_movement()
        process_eat(env_S, env_R)
        process_emit_signal(env_G)
        process_repair()
        process_attack()
        process_bond()
        process_unbond()
        process_bond_strength_update()
        process_bond_sharing()
        process_bond_signal_output()
        process_bond_signal_relay()

        # 4. Division
        process_divide_phase1()
        process_divide_phase2()
        process_mutations(self.mutation_rng, self.tick_count)

        # Log lineage events
        if self.logger:
            events = get_mutation_events()
            if events:
                self.logger.log_lineage_events(events)

        # 5. Metabolism and death
        apply_metabolism()
        check_death(env_S, env_R, env_G)

        # 6. Swap diffusion buffers
        swap_buffers()

        # 7. Periodic deposit relocation
        if self.tick_count > 0 and self.tick_count % DEPOSIT_RELOCATE_INTERVAL == 0:
            relocate_deposits()

        # 8. Periodic genome garbage collection
        if self.tick_count > 0 and self.tick_count % GENOME_GC_INTERVAL == 0:
            garbage_collect_genomes()

        # 9. Periodic logging
        if self.logger and self.tick_count % SNAPSHOT_INTERVAL == 0:
            self.logger.snapshot(self.tick_count)

        # 9b. Burst snapshots (checked every tick, lightweight)
        if self.logger:
            self.logger.check_burst_snapshot(self.tick_count)

        # 10. Check for backend auto-switch
        if (self.auto_switch and self.tick_count > 0
                and self.tick_count % _SWITCH_CHECK_INTERVAL == 0):
            desired = self._desired_backend()
            if desired:
                self._switch_streak += 1
                if self._switch_streak >= _SWITCH_CONFIRM_CHECKS:
                    self._do_backend_switch(desired)
            else:
                self._switch_streak = 0

        self.tick_count += 1

    def _print_progress(self):
        """Print progress stats to console during headless runs."""
        pop = cell_count[None]
        genomes = genome_count[None]
        print(f"[tick {self.tick_count:>8d}] pop={pop:>5d}  genomes={genomes:>5d}  backend={self.backend}")

    def run(self, max_ticks: int = 0):
        """Main loop with optional visualization."""
        if self.headless:
            if max_ticks > 0:
                for _ in range(max_ticks):
                    self.step()
                    if self.tick_count % self.log_interval == 0:
                        self._print_progress()
            else:
                while True:
                    self.step()
                    if self.tick_count % self.log_interval == 0:
                        self._print_progress()
        else:
            running = True
            while running:
                if not self.renderer.handle_input():
                    break

                if not self.renderer.paused:
                    for _ in range(self.renderer.ticks_per_frame):
                        self.step()

                env_S = get_env_S()
                env_R = get_env_R()
                env_G = get_env_G()
                self.renderer.render(self.tick_count, env_S, env_R, env_G)

                if max_ticks > 0 and self.tick_count >= max_ticks:
                    break

            if self.renderer:
                self.renderer.close()
