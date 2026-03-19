"""Main simulation engine — orchestrates the tick loop."""

import numpy as np

from config import GENOME_GC_INTERVAL, RANDOM_SEED, SNAPSHOT_INTERVAL

from world.grid import compute_light, init_grid
from world.chemistry import (
    diffuse_all, replenish_deposits, swap_buffers, init_chemistry,
    get_env_S, get_env_R, get_env_G,
)
from cell.cell_state import init_cell_state, cell_count
from cell.lifecycle import photosynthesis, eat_passive, apply_metabolism, check_death
from cell.genome import (
    init_genome_table, evaluate_all_networks, process_mutations,
    garbage_collect_genomes, genome_count,
)
from cell.sensing import compute_sensory_inputs
from cell.actions import (
    clear_intentions, process_turns,
    process_movement_phase1, process_movement_phase2,
    process_eat, process_emit_signal, process_repair, process_attack,
    process_divide_phase1, process_divide_phase2,
)
from simulation.spawner import seed_cells


class SimulationEngine:
    def __init__(self, headless: bool = False, log_interval: int = 1000):
        self.headless = headless
        self.log_interval = log_interval
        self.tick_count = 0
        self.mutation_rng = np.random.default_rng(RANDOM_SEED + 100)
        self.renderer = None
        self.logger = None

    def init(self):
        """Initialize all simulation subsystems."""
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
        process_eat(env_S, env_R)
        process_emit_signal(env_G)
        process_repair()
        process_attack()

        # 4. Division
        process_divide_phase1()
        process_divide_phase2()
        process_mutations(self.mutation_rng)

        # 5. Metabolism and death
        apply_metabolism()
        check_death(env_S, env_R, env_G)

        # 6. Swap diffusion buffers
        swap_buffers()

        # 7. Periodic genome garbage collection
        if self.tick_count > 0 and self.tick_count % GENOME_GC_INTERVAL == 0:
            garbage_collect_genomes()

        # 8. Periodic logging
        if self.logger and self.tick_count % SNAPSHOT_INTERVAL == 0:
            self.logger.snapshot(self.tick_count)

        self.tick_count += 1

    def _print_progress(self):
        """Print progress stats to console during headless runs."""
        pop = cell_count[None]
        genomes = genome_count[None]
        print(f"[tick {self.tick_count:>8d}] pop={pop:>5d}  genomes={genomes:>5d}")

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
                    if self.renderer.fast_forward:
                        # Skip rendering for 10 ticks
                        for _ in range(10):
                            self.step()
                    else:
                        self.step()

                env_S = get_env_S()
                env_R = get_env_R()
                env_G = get_env_G()
                self.renderer.render(self.tick_count, env_S, env_R, env_G)

                if max_ticks > 0 and self.tick_count >= max_ticks:
                    break

            if self.renderer:
                self.renderer.close()
