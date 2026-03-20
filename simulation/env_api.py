"""Environment Parameter API for runtime modification.

Provides a clean Python API for external processes (future LLM-guided
environment design) to monitor and modify the simulation without touching
simulation internals.
"""

import config
from cell.cell_state import cell_alive, cell_energy, cell_count, cell_bonds
from cell.genome import genome_count
from world.chemistry import get_env_S, get_env_R, get_env_G


class EnvironmentAPI:
    """API for monitoring and modifying the simulation environment at runtime."""

    def __init__(self, engine):
        self.engine = engine

    def get_metrics(self) -> dict:
        """Return current simulation metrics."""
        pop = cell_count[None]
        genomes = genome_count[None]

        # Compute average energy and bond count
        alive_np = cell_alive.to_numpy()
        energy_np = cell_energy.to_numpy()
        bonds_np = cell_bonds.to_numpy()

        alive_mask = alive_np == 1
        alive_count = alive_mask.sum()

        avg_energy = 0.0
        avg_bonds = 0.0
        if alive_count > 0:
            avg_energy = float(energy_np[alive_mask].mean())
            bond_counts = (bonds_np[alive_mask] >= 0).sum(axis=1)
            avg_bonds = float(bond_counts.mean())

        return {
            "tick": self.engine.tick_count,
            "population": pop,
            "genomes": genomes,
            "avg_energy": avg_energy,
            "avg_bonds": avg_bonds,
        }

    def set_parameter(self, name: str, value: float) -> bool:
        """Modify a config parameter at runtime.

        Returns True if the parameter exists and was set, False otherwise.
        Note: parameters captured at Taichi compile time (field shapes, etc.)
        cannot be changed at runtime. This affects runtime-readable constants
        like PHOTOSYNTHESIS_RATE, BASAL_METABOLISM, etc.
        """
        if hasattr(config, name):
            setattr(config, name, value)
            return True
        return False

    def get_parameter(self, name: str):
        """Get current value of a config parameter."""
        return getattr(config, name, None)

    def add_deposit(self, x: int, y: int, chemical: str, amount: float):
        """Place a chemical deposit at (x, y)."""
        if chemical == "S":
            env = get_env_S()
        elif chemical == "R":
            env = get_env_R()
        elif chemical == "G":
            env = get_env_G()
        else:
            raise ValueError(f"Unknown chemical: {chemical}")
        env[x, y] += amount

    def trigger_event(self, event_type: str, **kwargs):
        """Trigger an environmental event.

        Supported events:
        - 'resource_boom': flood area with chemicals
            kwargs: x, y, radius, chemical, amount
        """
        if event_type == "resource_boom":
            x = kwargs.get("x", 250)
            y = kwargs.get("y", 250)
            radius = kwargs.get("radius", 20)
            chemical = kwargs.get("chemical", "S")
            amount = kwargs.get("amount", 10.0)
            if chemical == "S":
                env = get_env_S()
            elif chemical == "R":
                env = get_env_R()
            else:
                return
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    if dx * dx + dy * dy <= radius * radius:
                        px = (x + dx) % config.GRID_WIDTH
                        py = (y + dy) % config.GRID_HEIGHT
                        env[px, py] += amount

    def get_population_snapshot(self) -> dict:
        """Return detailed population data for analysis."""
        import numpy as np

        alive_np = cell_alive.to_numpy()
        energy_np = cell_energy.to_numpy()
        alive_mask = alive_np == 1

        return {
            "alive_count": int(alive_mask.sum()),
            "energy_mean": float(energy_np[alive_mask].mean()) if alive_mask.any() else 0.0,
            "energy_std": float(energy_np[alive_mask].std()) if alive_mask.any() else 0.0,
            "energy_min": float(energy_np[alive_mask].min()) if alive_mask.any() else 0.0,
            "energy_max": float(energy_np[alive_mask].max()) if alive_mask.any() else 0.0,
        }
