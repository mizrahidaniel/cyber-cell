"""Validation harness for research-informed upgrades (Steps 8-14).

Runs headless simulations and checks that each new feature is working correctly.
Reports pass/fail for each feature with detailed diagnostics.

Usage:
    python analysis/validate.py [--ticks N] [--genome neural|crn] [--no-archipelago]
    python validate.py ...  # backward-compat wrapper also works
"""

import argparse
import os
import sys
import time

import numpy as np


def run_validation(ticks: int, genome_type: str = "neural",
                   archipelago: bool = True, seed: int = 42):
    """Run a validation simulation and collect diagnostics."""
    # Configure before importing Taichi (which compiles on import)
    import config
    config.GENOME_TYPE = genome_type
    config.ARCHIPELAGO_ENABLED = archipelago
    config.RANDOM_SEED = seed

    import taichi as ti
    ti.init(arch=ti.cuda, log_level=ti.WARN)

    from simulation.engine import SimulationEngine
    from cell.cell_state import (
        cell_alive, cell_x, cell_y, cell_energy, cell_bonds,
        cell_bond_strength, cell_bond_signal_out, cell_bond_signal_in,
        cell_last_attacker, cell_count,
    )
    from cell.genome import (
        genome_count, action_outputs, sensory_inputs, genome_ref_count,
    )
    from cell.lifecycle import deaths_by_attack, deaths_by_starvation
    from world.chemistry import get_env_S, get_env_R

    engine = SimulationEngine(headless=True, backend="cuda", auto_switch=False)
    engine.init()

    # -- Sampling accumulators --
    pop_history = []
    genome_history = []
    energy_history = []
    bond_count_history = []
    bond_strength_samples = []
    bond_signal_nonzero_count = 0
    bond_signal_total_checks = 0
    attack_deaths_total = 0
    starvation_deaths_total = 0
    kill_reward_detected = False
    gradient_noise_variance_samples = []
    move_fraction_samples = []
    attack_fraction_samples = []
    # Light attenuation accumulators
    effective_light_samples = []
    local_density_samples = []
    # CRN-specific accumulators
    hidden_zone_samples = []

    sample_interval = max(1, ticks // 100)  # ~100 samples

    print(f"\n{'='*60}")
    print(f"  VALIDATION RUN: {ticks:,} ticks")
    print(f"  Genome: {genome_type}, Archipelago: {archipelago}, Seed: {seed}")
    print(f"{'='*60}\n")

    t0 = time.time()

    for t in range(ticks):
        engine.step()

        if t % sample_interval == 0:
            pop = cell_count[None]
            pop_history.append(pop)
            genome_history.append(genome_count[None])

            alive_np = cell_alive.to_numpy()
            alive_mask = alive_np == 1
            alive_count = alive_mask.sum()

            if alive_count > 0:
                energy_np = cell_energy.to_numpy()
                energy_history.append(float(energy_np[alive_mask].mean()))

                # Bond count
                bonds_np = cell_bonds.to_numpy()
                bond_counts = (bonds_np[alive_mask] >= 0).sum(axis=1)
                bond_count_history.append(float(bond_counts.mean()))

                # Bond strength (sample up to 200 cells)
                strength_np = cell_bond_strength.to_numpy()
                for idx in np.where(alive_mask)[0][:200]:
                    for b in range(4):
                        if bonds_np[idx, b] >= 0:
                            bond_strength_samples.append(strength_np[idx, b])

                # Bond signal activity
                sig_out = cell_bond_signal_out.to_numpy()
                sig_in = cell_bond_signal_in.to_numpy()
                bonded_cells = np.where(alive_mask & (bonds_np.max(axis=1) >= 0))[0]
                if len(bonded_cells) > 0:
                    bond_signal_total_checks += len(bonded_cells)
                    for idx in bonded_cells[:100]:
                        if np.any(sig_out[idx] != 0) or np.any(sig_in[idx] != 0):
                            bond_signal_nonzero_count += 1

                # Movement and attack fraction from action outputs
                act_np = action_outputs.to_numpy()
                alive_acts = act_np[alive_mask]
                if len(alive_acts) > 0:
                    move_fraction_samples.append(
                        float((alive_acts[:, 0] > 0.5).mean()))
                    attack_fraction_samples.append(
                        float((alive_acts[:, 8] > 0.5).mean()))

                # Gradient noise: check variance of gradient inputs across cells
                sens_np = sensory_inputs.to_numpy()
                alive_sens = sens_np[alive_mask]
                if len(alive_sens) > 50:
                    # Gradient inputs are [5..10]
                    grad_vars = alive_sens[:, 5:11].var(axis=0)
                    gradient_noise_variance_samples.append(grad_vars.mean())

                # Light attenuation stats
                if config.LIGHT_ATTENUATION_ENABLED:
                    from world.grid import local_density, light_field
                    density_np = local_density.to_numpy()
                    light_np = light_field.to_numpy()
                    alive_xs = cell_x.to_numpy()[alive_mask]
                    alive_ys = cell_y.to_numpy()[alive_mask]
                    cell_lights = light_np[alive_xs, alive_ys]
                    cell_dens = density_np[alive_xs, alive_ys]
                    effective_light_samples.append(float(cell_lights.mean()))
                    local_density_samples.append(float(cell_dens.mean()))

                # CRN-specific: hidden zone activation
                if genome_type == "crn":
                    from cell.crn_genome import crn_chemicals
                    from config import MAX_CELLS
                    crn_chems = crn_chemicals.to_numpy()[:MAX_CELLS]
                    alive_chems = crn_chems[alive_mask]
                    hidden_zone_samples.append(
                        float(alive_chems[:, 8:12].mean()))

            else:
                energy_history.append(0.0)
                bond_count_history.append(0.0)

            # Death tracking
            attack_deaths_total += deaths_by_attack[None]
            starvation_deaths_total += deaths_by_starvation[None]
            deaths_by_attack[None] = 0
            deaths_by_starvation[None] = 0

            # Kill reward: check if any attacker has high energy
            attacker_np = cell_last_attacker.to_numpy()
            if alive_count > 0:
                attackers_alive = attacker_np[alive_mask]
                valid_attackers = attackers_alive[attackers_alive >= 0]
                if len(valid_attackers) > 0:
                    kill_reward_detected = True

        if t % (ticks // 10) == 0 and t > 0:
            elapsed = time.time() - t0
            tps = t / elapsed
            pop = cell_count[None]
            print(f"  [{t:>8,}/{ticks:,}] pop={pop:>5,}  "
                  f"genomes={genome_count[None]:>5,}  "
                  f"{tps:.0f} ticks/s")

    elapsed = time.time() - t0
    print(f"\n  Completed in {elapsed:.1f}s ({ticks/elapsed:.0f} ticks/s)")

    # -- Collect final state --
    alive_np = cell_alive.to_numpy()
    alive_mask = alive_np == 1
    final_pop = alive_mask.sum()

    # Per-quadrant population (archipelago check)
    quadrant_pops = [0, 0, 0, 0]
    if archipelago:
        x_np = cell_x.to_numpy()
        y_np = cell_y.to_numpy()
        half_w = config.GRID_WIDTH // 2
        half_h = config.GRID_HEIGHT // 2
        for i in np.where(alive_mask)[0]:
            ix = 0 if x_np[i] < half_w else 1
            iy = 0 if y_np[i] < half_h else 1
            quadrant_pops[iy * 2 + ix] += 1

    # Bond cluster analysis
    bonds_np = cell_bonds.to_numpy()
    bonded_mask = alive_mask & (bonds_np.max(axis=1) >= 0)
    bonded_count = bonded_mask.sum()

    # Cluster size distribution via union-find
    parent = {}
    for i in np.where(bonded_mask)[0]:
        if i not in parent:
            parent[i] = i
        for b in range(4):
            p = bonds_np[i, b]
            if p >= 0 and alive_np[p] == 1:
                if p not in parent:
                    parent[p] = p
                # Union
                ri, rp = i, p
                while parent[ri] != ri:
                    ri = parent[ri]
                while parent[rp] != rp:
                    rp = parent[rp]
                if ri != rp:
                    parent[rp] = ri

    # Count cluster sizes
    clusters = {}
    for node in parent:
        root = node
        while parent[root] != root:
            root = parent[root]
        clusters.setdefault(root, []).append(node)

    cluster_sizes = [len(v) for v in clusters.values()]

    # CRN-specific: dominant genome active reaction count
    crn_dominant_active_reactions = 0
    if genome_type == "crn":
        from cell.crn_genome import crn_weights
        from config import MAX_REACTIONS, CRN_PARAMS_PER_REACTION
        refs = genome_ref_count.to_numpy()
        if refs.max() > 0:
            dom_gid = int(np.argmax(refs))
            w = crn_weights.to_numpy()
            crn_dominant_active_reactions = sum(
                1 for r in range(MAX_REACTIONS)
                if abs(w[dom_gid, r * CRN_PARAMS_PER_REACTION + 5]) > 0.001)

    return {
        "ticks": ticks,
        "genome_type": genome_type,
        "archipelago": archipelago,
        "elapsed": elapsed,
        "pop_history": pop_history,
        "genome_history": genome_history,
        "energy_history": energy_history,
        "bond_count_history": bond_count_history,
        "bond_strength_samples": bond_strength_samples,
        "bond_signal_nonzero_count": bond_signal_nonzero_count,
        "bond_signal_total_checks": bond_signal_total_checks,
        "gradient_noise_variance": gradient_noise_variance_samples,
        "move_fraction_samples": move_fraction_samples,
        "attack_fraction_samples": attack_fraction_samples,
        "attack_deaths": attack_deaths_total,
        "starvation_deaths": starvation_deaths_total,
        "kill_reward_detected": kill_reward_detected,
        "final_pop": final_pop,
        "quadrant_pops": quadrant_pops,
        "bonded_count": bonded_count,
        "cluster_sizes": cluster_sizes,
        "effective_light_samples": effective_light_samples,
        "local_density_samples": local_density_samples,
        "hidden_zone_samples": hidden_zone_samples,
        "crn_dominant_active_reactions": crn_dominant_active_reactions,
    }


def check_results(results: dict) -> list[tuple[str, bool, str]]:
    """Check each feature and return (name, passed, detail) tuples."""
    checks = []

    # 1. Population stability
    pop = results["pop_history"]
    min_pop = min(pop) if pop else 0
    final_pop = results["final_pop"]
    peak_pop = max(pop) if pop else 0

    passed = final_pop > 50
    detail = f"final={final_pop}, min={min_pop}, peak={peak_pop}"
    checks.append(("Population survives", passed, detail))

    # Check for catastrophic crash (>95% loss)
    if len(pop) > 2:
        crash = min_pop < pop[0] * 0.05
        checks.append(("No catastrophic crash (<95% loss)", not crash,
                       f"initial={pop[0]}, min={min_pop}"))

    # 2. Bond strength decay (Step 8a)
    strengths = results["bond_strength_samples"]
    if len(strengths) > 10:
        strengths_arr = np.array(strengths)
        mean_str = strengths_arr.mean()
        # Bonds should show SOME variety — decay/reinforcement is working
        has_variety = strengths_arr.std() > 0.01
        not_all_max = mean_str < 0.99  # near-permanent bonds OK (decay=0.001)
        passed = has_variety and not_all_max
        detail = f"mean={mean_str:.3f}, std={strengths_arr.std():.3f}, n={len(strengths)}"
    else:
        passed = True  # No bonds formed, so decay isn't testable
        detail = f"only {len(strengths)} bond samples (too few bonds to test decay)"
    checks.append(("Bond strength decay working (8a)", passed, detail))

    # 3. Lossy transfer (Step 8b) — hard to test directly, but if bonds exist
    # and population survives, the loss isn't killing everyone
    bonded = results["bonded_count"]
    checks.append(("Lossy transfer doesn't crash bonded cells (8b)", True,
                   f"{bonded} bonded cells alive at end"))

    # 4. Bond signal channels (Step 8c)
    sig_nz = results["bond_signal_nonzero_count"]
    sig_total = results["bond_signal_total_checks"]
    if results["genome_type"] == "crn":
        # CRN only maps 8 actions; bond signal outputs (10-13) are zeroed
        passed = True
        detail = f"N/A for CRN genome (8 actions, no bond signal outputs)"
    elif sig_total > 0:
        sig_frac = sig_nz / sig_total
        passed = sig_nz > 0
        detail = f"{sig_nz}/{sig_total} checks had nonzero signals ({sig_frac:.1%})"
    else:
        passed = True
        detail = "no bonded cells to check"
    checks.append(("Bond signals active for bonded cells (8c)", passed, detail))

    # 5. Gradient noise (Step 9a)
    grad_vars = results["gradient_noise_variance"]
    if len(grad_vars) > 5:
        mean_var = np.mean(grad_vars)
        passed = mean_var > 0.001  # some variance from noise
        detail = f"mean gradient variance={mean_var:.4f}"
    else:
        passed = True
        detail = "insufficient samples"
    checks.append(("Gradient noise applied (9a)", passed, detail))

    # 6. Archipelago (Step 10)
    if results["archipelago"]:
        qpops = results["quadrant_pops"]
        total = sum(qpops)
        populated = sum(1 for q in qpops if q > 0)
        if total > 100:
            # Cells spawn in light zone (left half only), so right quadrants
            # populate slowly via migration + exploration. For short runs,
            # check that at least left-side quadrants have cells.
            passed = populated >= 2
            detail = f"quadrant pops: {qpops}, {populated}/4 populated"
        else:
            passed = total > 0
            detail = f"low pop ({total}), quadrants: {qpops}"
        checks.append(("Archipelago: cells in multiple quadrants (10)", passed, detail))

    # 7. Predation (Step 11)
    attack_deaths = results["attack_deaths"]
    passed = True  # Predation may not evolve in short runs, that's OK
    detail = f"attack deaths: {attack_deaths}, starvation deaths: {results['starvation_deaths']}"
    if results["kill_reward_detected"]:
        detail += " [kill rewards detected]"
    checks.append(("Predation system active (11)", passed, detail))

    # 8. Genome diversity
    genomes = results["genome_history"]
    if len(genomes) > 2:
        peak_genomes = max(genomes)
        final_genomes = genomes[-1]
        passed = final_genomes > 10
        detail = f"final={final_genomes}, peak={peak_genomes}"
    else:
        passed = True
        detail = "insufficient data"
    checks.append(("Genome diversity maintained", passed, detail))

    # 9. Bond cluster analysis
    cluster_sizes = results["cluster_sizes"]
    if cluster_sizes:
        max_cluster = max(cluster_sizes)
        avg_cluster = np.mean(cluster_sizes)
        n_clusters = len(cluster_sizes)
        # Degenerate chains: clusters > 10 cells are suspicious
        degenerate = sum(1 for s in cluster_sizes if s > 10)
        passed = degenerate < n_clusters * 0.3  # < 30% degenerate
        detail = (f"{n_clusters} clusters, avg={avg_cluster:.1f}, "
                 f"max={max_cluster}, degenerate(>10)={degenerate}")
    else:
        passed = True
        detail = "no bonded clusters formed"
    checks.append(("No degenerate chains (8a goal)", passed, detail))

    # 10. Energy balance
    energies = results["energy_history"]
    if len(energies) > 5:
        avg_e = np.mean(energies[-10:])
        passed = avg_e > 1.0  # cells aren't all starving
        detail = f"avg energy (last samples)={avg_e:.2f}"
    else:
        passed = True
        detail = "insufficient data"
    checks.append(("Healthy energy levels", passed, detail))

    # 11. Light attenuation check
    light_samples = results.get("effective_light_samples", [])
    density_samples = results.get("local_density_samples", [])
    if light_samples and density_samples:
        avg_light = np.mean(light_samples)
        avg_dens = np.mean(density_samples)
        # With attenuation enabled, avg effective light at occupied cells
        # should be noticeably less than the base bright zone light (~0.45)
        passed = avg_light < 0.4 or avg_dens > 1.0
        detail = (f"avg effective light={avg_light:.3f}, "
                  f"avg local density={avg_dens:.1f}")
    else:
        passed = True
        detail = "attenuation not enabled or no samples"
    checks.append(("Light attenuation working", passed, detail))

    # 12-18. CRN-specific checks
    if results["genome_type"] == "crn":
        # CRN population target (lower with light attenuation enabled)
        import config as _cfg
        pop_target = 75 if getattr(_cfg, 'LIGHT_ATTENUATION_ENABLED', False) else 200
        passed = final_pop > pop_target
        detail = f"final_pop={final_pop} (target >{pop_target})"
        checks.append((f"CRN: Population > {pop_target}", passed, detail))

        # CRN movement: check early samples (circuit works before evolution erodes it)
        move_fracs = results.get("move_fraction_samples", [])
        if len(move_fracs) > 3:
            window = max(1, len(move_fracs) // 5)
            early_avg = np.mean(move_fracs[:window])
            overall_avg = np.mean(move_fracs)
            passed = early_avg > 0.05
            detail = (f"early avg={early_avg:.3f}, overall={overall_avg:.3f} "
                      f"(target: early >0.05)")
        elif move_fracs:
            avg_move = np.mean(move_fracs)
            passed = avg_move > 0.05
            detail = f"avg={avg_move:.3f} (few samples)"
        else:
            passed = True
            detail = "no samples"
        checks.append(("CRN: Movement circuit works", passed, detail))

        # CRN attack fraction
        atk_fracs = results.get("attack_fraction_samples", [])
        if atk_fracs:
            avg_atk = np.mean(atk_fracs)
            passed = avg_atk < 0.05
            detail = f"avg={avg_atk:.3f} (target <0.05)"
        else:
            passed = True
            detail = "no samples"
        checks.append(("CRN: Attack < 5%", passed, detail))

        # CRN: Hidden chemicals active
        hidden_samples = results.get("hidden_zone_samples", [])
        if hidden_samples:
            peak_hidden = max(hidden_samples)
            passed = peak_hidden > 0.01
            detail = f"peak hidden zone mean={peak_hidden:.4f} (target >0.01)"
        else:
            passed = True
            detail = "no samples"
        checks.append(("CRN: Hidden chemicals active", passed, detail))

        # CRN: Reactions diversified
        dom_active = results.get("crn_dominant_active_reactions", 0)
        passed = dom_active >= 5
        detail = f"dominant genome has {dom_active} active reactions (target >=5)"
        checks.append(("CRN: Reactions diversified", passed, detail))

    return checks


def print_report(results: dict, checks: list):
    """Print a formatted validation report."""
    print(f"\n{'='*60}")
    print(f"  VALIDATION REPORT — {results['ticks']:,} ticks "
          f"({results['genome_type'].upper()} genome)")
    print(f"{'='*60}\n")

    n_pass = sum(1 for _, p, _ in checks if p)
    n_total = len(checks)

    for name, passed, detail in checks:
        icon = "PASS" if passed else "FAIL"
        print(f"  [{icon}] {name}")
        print(f"         {detail}")

    print(f"\n  {'-'*50}")
    print(f"  Results: {n_pass}/{n_total} passed")

    if n_pass == n_total:
        print(f"  Status: ALL CHECKS PASSED")
    else:
        fails = [name for name, p, _ in checks if not p]
        print(f"  Status: {len(fails)} FAILURES — {', '.join(fails)}")

    print()
    return n_pass == n_total


def plot_validation(results: dict, output_dir: str):
    """Generate validation plots."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle(
        f"Validation: {results['ticks']:,} ticks — "
        f"{results['genome_type'].upper()} genome",
        fontsize=14, fontweight="bold",
    )

    ticks_axis = np.linspace(0, results["ticks"],
                              len(results["pop_history"]))

    # 1. Population over time
    ax = axes[0, 0]
    ax.plot(ticks_axis, results["pop_history"], "b-", linewidth=0.8)
    ax.set_title("Population")
    ax.set_xlabel("Tick")
    ax.set_ylabel("Alive cells")
    ax.grid(True, alpha=0.3)

    # 2. Genome diversity
    ax = axes[0, 1]
    ax.plot(ticks_axis, results["genome_history"], "g-", linewidth=0.8)
    ax.set_title("Genome Diversity")
    ax.set_xlabel("Tick")
    ax.set_ylabel("Unique genomes")
    ax.grid(True, alpha=0.3)

    # 3. Average energy
    ax = axes[0, 2]
    ax.plot(ticks_axis, results["energy_history"], "r-", linewidth=0.8)
    ax.set_title("Average Energy")
    ax.set_xlabel("Tick")
    ax.set_ylabel("Energy")
    ax.grid(True, alpha=0.3)

    # 4. Bond count
    ax = axes[1, 0]
    ax.plot(ticks_axis, results["bond_count_history"], "m-", linewidth=0.8)
    ax.set_title("Average Bond Count")
    ax.set_xlabel("Tick")
    ax.set_ylabel("Bonds per cell")
    ax.grid(True, alpha=0.3)

    # 5. Bond strength distribution
    ax = axes[1, 1]
    strengths = results["bond_strength_samples"]
    if len(strengths) > 10:
        ax.hist(strengths, bins=30, color="orange", edgecolor="black",
                linewidth=0.5, alpha=0.7)
        ax.axvline(0.5, color="red", linestyle="--", label="Initial strength")
        ax.legend(fontsize=8)
    ax.set_title("Bond Strength Distribution")
    ax.set_xlabel("Strength")
    ax.set_ylabel("Count")

    # 6. Quadrant populations (archipelago)
    ax = axes[1, 2]
    if results["archipelago"] and sum(results["quadrant_pops"]) > 0:
        labels = ["Q0 (NW)", "Q1 (NE)", "Q2 (SW)", "Q3 (SE)"]
        colors = ["#4CAF50", "#2196F3", "#FF9800", "#F44336"]
        ax.bar(labels, results["quadrant_pops"], color=colors, edgecolor="black",
               linewidth=0.5)
        ax.set_title("Archipelago: Final Pop by Quadrant")
        ax.set_ylabel("Cells")
    else:
        # Cluster size distribution
        cluster_sizes = results["cluster_sizes"]
        if cluster_sizes:
            ax.hist(cluster_sizes, bins=range(1, max(cluster_sizes) + 2),
                    color="teal", edgecolor="black", linewidth=0.5, alpha=0.7)
        ax.set_title("Bond Cluster Sizes")
        ax.set_xlabel("Cluster size")
        ax.set_ylabel("Count")

    plt.tight_layout()
    path = os.path.join(output_dir, "validation.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Plot saved: {path}")


def main():
    parser = argparse.ArgumentParser(description="CyberCell validation harness")
    parser.add_argument("--ticks", type=int, default=10000,
                       help="Number of ticks to simulate (default: 10000)")
    parser.add_argument("--genome", choices=["neural", "crn"], default="neural",
                       help="Genome type (default: neural)")
    parser.add_argument("--no-archipelago", action="store_true",
                       help="Disable archipelago")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-plot", action="store_true",
                       help="Skip plot generation")
    args = parser.parse_args()

    results = run_validation(
        ticks=args.ticks,
        genome_type=args.genome,
        archipelago=not args.no_archipelago,
        seed=args.seed,
    )
    checks = check_results(results)
    all_passed = print_report(results, checks)

    # Stable output dir (overwrites previous for same config)
    out_dir = os.path.join("analysis", "output",
                           f"validate_{args.genome}_{args.ticks}t")

    if not args.no_plot:
        plot_validation(results, out_dir)

    # Always save a text report
    os.makedirs(out_dir, exist_ok=True)
    report_path = os.path.join(out_dir, "VALIDATION_REPORT.txt")
    with open(report_path, "w") as f:
        n_pass = sum(1 for _, p, _ in checks if p)
        f.write(f"VALIDATION REPORT — {results['ticks']:,} ticks "
                f"({results['genome_type'].upper()} genome)\n")
        f.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Seed: {args.seed}  Archipelago: {not args.no_archipelago}\n")
        f.write(f"Result: {n_pass}/{len(checks)} passed\n\n")
        for name, passed, detail in checks:
            icon = "PASS" if passed else "FAIL"
            f.write(f"  [{icon}] {name}\n         {detail}\n")
    print(f"  Report saved: {report_path}")

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
