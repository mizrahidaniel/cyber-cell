"""Open-Ended Evolution (OEE) metrics for measuring evolutionary progress.

Implements:
- Bedau's evolutionary activity (genome frequency change rate)
- MODES metrics: Change, Novelty, Complexity, Ecology (Dolson et al.)
- Shannon entropy of genome distribution
- Mutual information between sensory inputs and action outputs
- Bond network information metrics

These metrics detect whether evolution is still producing novelty or has
plateaued. A plateau for >50,000 ticks signals that the environment needs
a qualitative change.
"""

import numpy as np
from collections import defaultdict

from config import MAX_CELLS, MAX_GENOMES, NUM_INPUTS, NUM_OUTPUTS
from cell.cell_state import cell_alive, cell_genome_id, cell_bonds
from cell.genome import genome_ref_count, sensory_inputs, action_outputs


class OEEMetrics:
    """Track and compute open-ended evolution metrics over time."""

    def __init__(self, window_size: int = 50):
        self.window_size = window_size
        # History of genome frequency distributions
        self._freq_history = []
        # Set of all genomes ever seen
        self._all_genomes_seen = set()
        # History of metric values for plateau detection
        self._metric_history = []

    def compute_all(self, tick: int) -> dict:
        """Compute all OEE metrics for the current tick."""
        ref_counts = genome_ref_count.to_numpy()
        alive_np = cell_alive.to_numpy()
        genome_ids = cell_genome_id.to_numpy()

        alive_mask = alive_np == 1
        alive_count = alive_mask.sum()

        if alive_count == 0:
            return self._empty_metrics(tick)

        # Current genome frequency distribution
        live_genomes = genome_ids[alive_mask]
        unique, counts = np.unique(live_genomes, return_counts=True)
        freq = dict(zip(unique.tolist(), counts.tolist()))
        self._freq_history.append(freq)
        if len(self._freq_history) > self.window_size:
            self._freq_history.pop(0)

        # Track all genomes ever seen
        for g in unique:
            self._all_genomes_seen.add(int(g))

        metrics = {
            "tick": tick,
            "population": int(alive_count),
            "unique_genomes": len(unique),
            "total_genomes_seen": len(self._all_genomes_seen),
        }

        # Bedau evolutionary activity
        metrics["activity"] = self._bedau_activity(freq, alive_count)

        # MODES metrics
        modes = self._modes_metrics(freq, unique, alive_count)
        metrics.update(modes)

        # Shannon entropy
        metrics["entropy"] = self._shannon_entropy(counts, alive_count)

        # Mutual information (sensory→action)
        metrics["mutual_info"] = self._mutual_information(alive_mask)

        # Bond network metrics
        metrics["bond_density"] = self._bond_density(alive_mask)

        # Plateau detection
        self._metric_history.append(metrics)
        if len(self._metric_history) > self.window_size:
            self._metric_history.pop(0)
        metrics["plateaued"] = bool(self._detect_plateau())

        return metrics

    def _bedau_activity(self, freq: dict, pop: int) -> float:
        """Bedau's evolutionary activity: rate of change in genome frequencies."""
        if len(self._freq_history) < 2:
            return 0.0

        prev_freq = self._freq_history[-2]
        activity = 0.0
        all_genomes = set(freq.keys()) | set(prev_freq.keys())

        for g in all_genomes:
            f_now = freq.get(g, 0) / max(1, pop)
            f_prev = prev_freq.get(g, 0) / max(1, pop)
            activity += abs(f_now - f_prev)

        return float(activity)

    def _modes_metrics(self, freq: dict, unique: np.ndarray,
                       alive_count: int) -> dict:
        """MODES: Change, Novelty, Complexity, Ecology."""
        change = 0.0
        novelty = 0.0

        if len(self._freq_history) >= 2:
            prev_freq = self._freq_history[-2]
            # Change: fraction of population that changed genome
            prev_genomes = set(prev_freq.keys())
            curr_genomes = set(freq.keys())
            new_genomes = curr_genomes - prev_genomes
            lost_genomes = prev_genomes - curr_genomes

            change = (len(new_genomes) + len(lost_genomes)) / max(1, len(curr_genomes | prev_genomes))
            novelty = len(new_genomes) / max(1, len(curr_genomes))

        # Complexity proxy: active reactions for CRN, unique genomes for neural
        from config import GENOME_TYPE
        if GENOME_TYPE == "crn":
            from cell.crn_genome import crn_weights
            from config import MAX_REACTIONS, CRN_PARAMS_PER_REACTION
            w = crn_weights.to_numpy()
            active_total = 0
            for gid in unique:
                active_total += sum(
                    1 for r in range(MAX_REACTIONS)
                    if abs(w[int(gid), r * CRN_PARAMS_PER_REACTION + 5]) > 0.001)
            complexity = float(active_total / max(1, len(unique)))
        else:
            complexity = float(len(unique))

        # Ecology: evenness of genome distribution (1 = perfectly even)
        counts = np.array(list(freq.values()), dtype=np.float64)
        if len(counts) > 1:
            entropy = -np.sum((counts / counts.sum()) * np.log2(counts / counts.sum() + 1e-10))
            max_entropy = np.log2(len(counts))
            ecology = float(entropy / max(1e-10, max_entropy))
        else:
            ecology = 0.0

        return {
            "modes_change": float(change),
            "modes_novelty": float(novelty),
            "modes_complexity": complexity,
            "modes_ecology": ecology,
        }

    def _shannon_entropy(self, counts: np.ndarray, pop: int) -> float:
        """Shannon entropy of genome frequency distribution."""
        if pop == 0 or len(counts) == 0:
            return 0.0
        probs = counts.astype(np.float64) / pop
        probs = probs[probs > 0]
        return float(-np.sum(probs * np.log2(probs)))

    def _mutual_information(self, alive_mask: np.ndarray,
                            sample_size: int = 1000) -> float:
        """Mutual information between sensory inputs and action outputs.

        High MI = organisms are using their senses to guide behavior.
        Low MI = random or fixed behavior regardless of input.
        """
        alive_indices = np.where(alive_mask)[0]
        if len(alive_indices) < 10:
            return 0.0

        # Sample a subset for efficiency
        n = min(sample_size, len(alive_indices))
        sample = np.random.choice(alive_indices, size=n, replace=False)

        sens_np = sensory_inputs.to_numpy()
        act_np = action_outputs.to_numpy()

        sens_sample = sens_np[sample, :NUM_INPUTS]
        act_sample = act_np[sample, :NUM_OUTPUTS]

        # Discretize into bins
        n_bins = 5
        sens_binned = np.digitize(sens_sample, np.linspace(0, 1, n_bins + 1)[:-1])
        act_binned = (act_sample > 0.5).astype(int)

        # Estimate MI using joint and marginal histograms
        mi_total = 0.0
        n_pairs = min(3, NUM_INPUTS)  # sample a few input-output pairs
        for s_idx in range(n_pairs):
            for a_idx in range(min(3, NUM_OUTPUTS)):
                s = sens_binned[:, s_idx]
                a = act_binned[:, a_idx]
                # Joint distribution
                joint = np.zeros((n_bins + 1, 2))
                for k in range(n):
                    joint[s[k], a[k]] += 1
                joint /= max(1, n)
                # Marginals
                p_s = joint.sum(axis=1)
                p_a = joint.sum(axis=0)
                # MI
                for si in range(n_bins + 1):
                    for ai in range(2):
                        if joint[si, ai] > 0 and p_s[si] > 0 and p_a[ai] > 0:
                            mi_total += joint[si, ai] * np.log2(
                                joint[si, ai] / (p_s[si] * p_a[ai]))

        return float(mi_total / max(1, n_pairs * min(3, NUM_OUTPUTS)))

    def _bond_density(self, alive_mask: np.ndarray) -> float:
        """Average bond count per alive cell."""
        alive_indices = np.where(alive_mask)[0]
        if len(alive_indices) == 0:
            return 0.0
        bonds_np = cell_bonds.to_numpy()
        bond_counts = (bonds_np[alive_indices] >= 0).sum(axis=1)
        return float(bond_counts.mean())

    def _detect_plateau(self, threshold_ticks: int = 50) -> bool:
        """Detect if evolution has stalled (metrics flat for too long)."""
        if len(self._metric_history) < threshold_ticks:
            return False
        recent = self._metric_history[-threshold_ticks:]
        activities = [m.get("activity", 0) for m in recent]
        novelties = [m.get("modes_novelty", 0) for m in recent]
        # Plateau if both activity and novelty are near zero
        return (np.mean(activities) < 0.01 and np.mean(novelties) < 0.01)

    def _empty_metrics(self, tick: int) -> dict:
        return {
            "tick": tick, "population": 0, "unique_genomes": 0,
            "total_genomes_seen": len(self._all_genomes_seen),
            "activity": 0.0, "modes_change": 0.0, "modes_novelty": 0.0,
            "modes_complexity": 0.0, "modes_ecology": 0.0, "entropy": 0.0,
            "mutual_info": 0.0, "bond_density": 0.0, "plateaued": False,
        }
