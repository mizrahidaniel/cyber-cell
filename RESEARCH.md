# Solving CyberCell's six critical failure modes

**CyberCell v4.0's problems — sessile CRN agents, collapsed bonding, degenerate neural controllers, runaway predation, vanishing diversity, and stalled complexity — each have well-studied analogues in the artificial life and evolutionary computation literature, and each has proven solutions.** The interventions below are drawn from systems like ALIEN, Avida, Biomaker CA, Sensorimotor Lenia, and decades of population genetics theory. Implemented together, they address the root causes rather than symptoms: the CRN's hard action threshold blocks evolutionary gradient toward movement, the bond system imposes an unsolvable cooperation dilemma, the neural genome is too large for unstructured mutation, predation energy transfer is 5–10× too efficient, migration is catastrophically infrequent, and fixed-length genomes cannot support the complexity ratchet needed for higher stages.

---

## Problem 1: Breaking the CRN sessile plant trap

The CRN's move chemical (C13) sitting at 0.293 against a hard threshold of 0.5 is a textbook instance of the **bootstrap problem** formalized by Mouret & Doncieux (2009): when all individuals perform equally (all photosynthesize, none move), selection has zero gradient toward movement. Evolution cannot start. The two negative-rate reactions targeting C13 confirm that once photosynthesis provides adequate energy, any mutation improving movement is strictly neutral or costly, so drift pushes move-production rates negative.

**Replace the hard threshold with sigmoid probability mapping.** The entire CTRNN literature (Beer 1995 onward) uses continuous activation — `P(move) = σ(gain × (C13 − center))` where σ is a sigmoid. This converts the binary cliff at 0.5 into a smooth gradient. At C13 = 0.293 with gain = 5 and center = 0.5, the move probability is ~26% rather than zero. Evolution can then incrementally adjust C13 concentration upward. Every successful CRN/GRN controller in the literature — GReaNs, CTRNNs, the 2025 GRN vision-based robot controller — uses continuous output mapping, never hard thresholds.

**Add a minimal viability criterion.** Lehman & Stanley's Minimal Criteria Novelty Search (2010) formalized the "behavioral floor" concept: agents that haven't moved at least N cells in their lifetime are ineligible for reproduction. This creates a hard floor below which sessile behavior is nonviable while imposing no ceiling on how organisms use movement. Alternatively, add a small displacement-proportional fitness bonus (`fitness += 0.01 × distance_traveled`) to break the symmetry between sessile and motile strategies. Sensorimotor Lenia (Hamon et al., 2024, *Science Advances*) found that random search yields "mostly static patterns" — only directed diversity search (IMGEP) reliably produces motile agents.

**Seed initial genomes with at least one positive-rate move-producing reaction** and add constitutive basal production (`∅ → C13` at rate ~0.01/tick). Following the NEAT principle of minimal initialization, start with few reactions and allow mutations to add complexity. Ensure no reaction targeting an action chemical can mutate to a rate below zero — constrain action-producing reaction rates to [0, ∞).

**Make photosynthesis spatially variable.** If light intensity varies across the grid, movement to brighter locations provides direct individual fitness benefit, creating selection pressure for motility even without explicit movement bonuses.

---

## Problem 2: Rescuing the suppressed hidden zone

The hidden chemicals at mean activation −0.04 reflect a fundamental design error: **chemical concentrations cannot be negative in any real or well-designed artificial chemistry.** The entire CRN/artificial chemistry literature enforces non-negative concentrations via mass-action kinetics. Allowing negative values means random reactions can suppress hidden chemicals below zero, where they become computationally inert.

**Clamp all chemical concentrations to [0, ∞).** This single change eliminates the mechanism driving hidden chemicals negative. If signed internal state is needed, use the "dual-rail" representation from CRN computing: two chemicals per channel (positive and negative), each non-negative, with their difference representing signed value. Alternatively, follow the CTRNN approach: let internal state variables be unconstrained but pass them through sigmoid when read by other reactions (`effective_H = σ(H − 0.5)`).

**Add constitutive basal production** for each hidden chemical: `∅ → Hi` at rate β ≈ 0.01–0.02/tick. With decay rate γ, steady-state concentration floors at β/γ. At current γ = 0.05 and β = 0.02, the floor is **0.4** — well above zero and in a computationally useful range.

**Reduce hidden zone decay rate to 0.02/tick** (half-life ~35 ticks). The current 0.05/tick (half-life ~14 ticks) is appropriate for fast-responding action chemicals but too aggressive for memory. The literature on biological GRNs shows a clear design principle: regulatory signals (transcription factors) have short half-lives for rapid response; structural and memory molecules have long half-lives for stability. For CyberCell's three chemical zones, recommended decay rates are **0.01–0.02/tick for sensory chemicals** (persistence), **0.02–0.03/tick for hidden/memory chemicals** (state retention), and **0.05–0.10/tick for action chemicals** (responsive output).

**Add autoregulatory feedback.** Biological GRNs maintain stable attractors through negative feedback loops and toggle switches (mutual inhibition creating bistability). Include a self-activation reaction `Hi → 2Hi` at a small rate capped by a maximum concentration, providing positive feedback that resists suppression while the cap prevents runaway. The Absolute Concentration Robustness (ACR) framework (Enciso et al., 2020) shows that CRN architectures can maintain target concentrations regardless of perturbation through paired production-degradation reactions.

---

## Problem 3: Why bonding collapses and how to make it permanent

CyberCell's bond mechanics impose an impossible evolutionary hurdle. A **0.02/tick decay rate** gives bonds a ~25-tick half-life, requiring constant mutual reinforcement where both cells independently fire bond output >0.5. This is a cooperation dilemma layered on top of a coordination dilemma — the probability that two independent evolved controllers simultaneously produce the right output and sustain it indefinitely is vanishingly small. The literature is unambiguous: **every successful multicellularity simulation uses permanent or near-permanent bonds.**

ALIEN uses physics-based permanent rigid-body connections that break only under excessive force. ProtoEvo uses structural adhesion nodes that persist until cell death. Biomaker CA makes spatial adjacency equivalent to bonding. Snowflake yeast — the most studied experimental model of nascent multicellularity (Ratcliff lab) — uses incomplete cytokinesis where daughter cells simply never separate from parents. **No successful system requires ongoing behavioral reinforcement of bonds.**

The single most impactful change is to **make parent-offspring bonds automatic and permanent at cell division.** This mirrors biological multicellularity's universal mechanism (incomplete cytokinesis) and eliminates the cooperation dilemma entirely, since Hamilton's rule with clonal relatedness r = 1 reduces to B > C — any group benefit suffices. If active bond dissolution is desired, let cells evolve a "sever" output rather than requiring continuous "maintain" signals. Reduce or eliminate bond decay: change from **0.02/tick to ≤0.001/tick** (bonds lasting >1,000 ticks), or make bonds permanent with damage-based destruction only.

**Create task incompatibility to make bonding individually beneficial.** Ispolatov, Ackermann & Doebeli (2012) showed that when two metabolic processes are incompatible (performing both reduces efficiency of each by ≥50%), cells that aggregate and specialize spontaneously outperform generalists. Goldsby et al. (2012, PNAS) confirmed in Avida that task-switching costs drive obligate division of labor — organisms "started expecting each other to be there" and could no longer survive alone. Implement two incompatible resource-processing tasks where bonded cells sharing complementary products achieve **≥2× combined efficiency** versus solo generalists.

**Ensure bonding provides fitness advantage exceeding the drift barrier.** Population genetics (Lynch 2007) establishes that selection is effective only when |Ns| >> 1. For an effective population of ~500, the drift barrier is at s ≈ 0.002. Bonding must provide at least **5–10% fitness advantage** (s = 0.05–0.10) to reliably resist drift and mutational erosion. With the task incompatibility model, bonded-specialized pairs should achieve ≥10% higher lifetime reproductive output than solo cells.

| Parameter | Current | Recommended |
|-----------|---------|-------------|
| Bond decay rate | 0.02/tick | ≤0.001/tick or 0 |
| Bond half-life | ~25 ticks | >1,000 ticks or infinite |
| Reinforcement requirement | Both cells >0.5 | Automatic at division |
| Bonding fitness advantage | <1% (estimated) | >5–10% |
| Task incompatibility | None | 2+ incompatible tasks |

---

## Problem 4: Halting neural mutual information collapse

The MI collapse from 0.20 to 0.005 means evolution is replacing sensory responsiveness with fixed instincts — a well-documented failure mode called **convergence to the degenerate attractor.** The 2,638-parameter direct-encoded genome is the root cause. In parameter space, the set of weight configurations producing fixed outputs (ignoring all inputs) is vastly larger than the set producing input-responsive behaviors. Random mutations overwhelmingly push networks toward this larger basin.

**Implement Safe Mutations through Output Gradients (SM-G)**, the technique from Lehman, Chen, Clune & Stanley (2018). SM-G computes the gradient of network outputs with respect to each weight, then scales mutation magnitude inversely to sensitivity: weights that strongly affect output get tiny mutations, weights with low sensitivity get larger ones. This preserves existing sense-action pathways while exploring new behaviors. SM-G enabled evolution of networks with **over 1 million parameters** — CyberCell's 2,638 are well within range.

**Add mutual information as an explicit fitness objective.** Using NSGA-II multi-objective optimization with two objectives — survival fitness and MI(inputs, actions) — directly prevents the degenerate attractor. Fixed-output strategies score zero on MI and are Pareto-dominated by any strategy that responds to even one input. Mouret & Doncieux (2012) showed that behavioral diversity as an explicit objective prevents premature convergence in evolved robot controllers.

**Consider switching to NEAT-style complexification.** NEAT starts from minimal topology (34 inputs directly connected to 14 outputs = ~524 parameters) and grows incrementally. This guarantees that **every input is connected to outputs from the start** — the degenerate solution of disconnected inputs is structurally impossible. NEAT's speciation protects structural innovations from being immediately eliminated. If architectural refactoring is feasible, this addresses the root cause.

**Alternatively, use CPPN/HyperNEAT indirect encoding.** A Compositional Pattern-Producing Network of ~100–200 parameters generates all 2,638 weights as a function of neuron spatial coordinates. Mutations in CPPN space create correlated, structured weight changes rather than random independent perturbations. The HybrID approach (Clune et al., 2009) starts with indirect encoding for structure discovery, then switches to direct encoding for fine-tuning. This reduces the effective genome from 2,638 to ~150 evolvable parameters while maintaining the full network architecture.

**Increase environmental variation.** Nolfi et al. (2019) demonstrated that environments changing every generation produce more robust, responsive controllers than fixed environments. If CyberCell's environment is too predictable, fixed instincts are genuinely optimal — the Baldwin Effect predicts that evolution will assimilate learned responses into fixed ones when environments are stable. Randomizing food placement, predator behavior, and environmental conditions across generations makes sensory responsiveness necessary for fitness.

---

## Problem 5: Preventing selective sweeps with proper population structure

Going from 1,011 to 11 root ancestors by tick 10,000 reveals that CyberCell's population structure is functionally panmictic despite its archipelago design. The core problem is quantitative: **Nm ≈ 0.002 per tick** (10 migrants / 5,000 ticks ≈ effectively zero continuous gene flow). Wright's FST formula (FST = 1/(4Nm + 1)) predicts near-complete differentiation between quadrants (FST ≈ 0.99), meaning each quadrant evolves independently. Within each quadrant, a beneficial mutation sweeps to fixation before any migration event occurs, and then one quadrant's winner sweeps globally during the next migration pulse.

**Restructure to 12–16 islands in a stepping-stone topology with continuous trickle migration.** The stepping-stone model (Kimura 1953), where migration occurs only between adjacent populations, generates substantially greater genetic differentiation than the fully-connected island model for the same global migration rate. Target **Nm = 1–5 per island per generation** — Wright's "one migrant per generation" rule counteracts drift without homogenizing subpopulations. With 16 islands of ~60 cells each, this means migrating **1–3 cells per island per tick** through nearest-neighbor exchange, not 10 cells globally every 5,000 ticks. Use **random migrant selection**, not best-individual — sending top performers accelerates global convergence.

**Implement spatial resource heterogeneity.** Dolson et al. (2017, ECAL) demonstrated in Avida that spatial resource heterogeneity is the single most effective natural diversity driver — different resources in different grid locations create niches that prevent any single genotype from being universally optimal. Make each island/region offer different resource profiles, light levels, or environmental challenges.

**Add fitness sharing or speciation.** NEAT-style speciation (Stanley & Miikkulainen, 2002) divides the population into species based on genetic compatibility distance, with fitness shared within species. This directly prevents any one lineage from monopolizing reproduction: as a genotype becomes common, its per-capita fitness drops. Implementation: `reproduction_rate_i = base_rate / (1 + α × count_similar_neighbors)`. Even weak fitness sharing (α = 0.1) dramatically slows sweeps.

**Implement negative frequency-dependent selection (NFDS)** — the most powerful natural force maintaining polymorphism. Multiple limited resources that different genotypes specialize on create automatic NFDS: as resource-A specialists become common, resource A depletes, giving resource-B specialists an advantage. Alternatively, add digital parasites that preferentially target common genotypes (Avida's parasite system increases host diversity through coevolutionary arms races). Target: **>100 root ancestors persisting at tick 10,000** with Ne > 500 per deme.

---

## Problem 6: Stabilizing predator-prey dynamics through ecological realism

CyberCell's predation economics violate basic ecological principles. Natural ecosystems transfer only **~10% of energy between trophic levels** (range 5–20%); CyberCell transfers >50% (50% absorption + 2.0 flat bonus). This 5–10× overefficiency is the direct cause of the positive feedback loop: predators gain energy faster than prey can reproduce, accelerating predator reproduction until prey collapse triggers predator starvation.

**Reduce energy absorption to 10–15% and eliminate the flat bonus entirely.** The +2.0 bonus is the primary destabilizer — it makes predation profitable regardless of prey energy content, creating a floor on predation reward that positive feedback amplifies. The Rosenzweig-MacArthur model shows that excess conversion efficiency (parameter e) pushes systems past the Hopf bifurcation into increasingly violent oscillations — the "paradox of enrichment." Target: `net_energy = (victim_energy × 0.12) − attack_cost`, where attack_cost = 5–10% of attacker's own energy.

**Implement a Type II functional response with handling time.** The current linear predation model (Type I) is the most destabilizing form. Holling's disk equation — `kill_reward = a × victim_energy / (1 + a × h × recent_kills)` where h is handling time — creates natural saturation. After each successful kill, impose a **5–10 tick cooldown** during which the predator cannot attack again. Maximum predation rate becomes 1/h ≈ 0.1–0.2 kills per tick regardless of prey density.

**Add Beddington-DeAngelis predator interference.** When multiple predators occupy the same area, each predator's attack efficiency should decrease: `effective_attack = base_rate / (1 + z × nearby_predator_count)` with interference coefficient z ≈ 0.1–0.3. Skalski & Gilliam (2001) found predator-dependent functional responses fit 19 real predator-prey communities better than prey-dependent models. Under mutual interference, the interior equilibrium becomes globally asymptotically stable — coexistence is guaranteed regardless of parameter specifics.

**Add metabolic cost of attack.** Every attack attempt should cost energy regardless of success. Natural predators fail frequently (wolves fail ~85% of hunts, cheetahs ~50–70%) and expend enormous energy pursuing prey. Making unsuccessful attacks costly creates natural selection pressure against excessive predation attempts and prevents the "indolent cannibal" problem documented in Polyworld (organisms mating, fighting, and eating each other in tight loops).

| Mechanism | Current | Recommended |
|-----------|---------|-------------|
| Energy transfer | 50% + 2.0 bonus | 10–15%, no bonus |
| Handling time | None | 5–10 tick cooldown |
| Attack cost | None | 5–10% of attacker energy |
| Predator interference | None | z = 0.1–0.3 |

With these changes, the equilibrium attack fraction should self-regulate to **5–15%** — viable as one of several survival strategies but unable to dominate.

---

## Lessons from ALIEN and Biomaker CA for system architecture

ALIEN (winner of the ALIFE 2024 Virtual Creatures Competition) and Biomaker CA (Google Research) both solve problems CyberCell faces, through fundamentally different design philosophies that bracket the solution space.

ALIEN uses **8 neurons per cell** — not 2,638 parameters — combined with signal propagation across cell networks every 6 timesteps, creating distributed computation from tiny local controllers. Its genome complexity bonus (more complex organisms get attack/absorption advantages) directly prevents parasitic simplification. Its "FindFortunateTimeline" script checkpoints every 100,000 timesteps and **reverts if population drops below 200**, acknowledging that long-term evolution requires extinction prevention through intervention. ALIEN's color-based food chain matrix (7 colors determining who can prey on whom) creates structured trophic relationships that prevent the uniform predation collapse CyberCell experiences.

Biomaker CA enforces **dual nutrient requirements** (earth nutrients from below, air nutrients from above), forcing organisms to solve spatial routing problems that drive morphological complexity. Its mutation strategy is directly relevant: **20% per-parameter mutation probability** per reproduction event (not 100%), with mutation standard deviation **scaled inversely to parameter count** (sd = 0.01 for ~300 params, sd = 0.001 for >10,000 params). Biomaker's authors hand-craft initial parameters that are already viable, then let evolution improve them — they explicitly argue this bootstrapping dramatically accelerates complexification versus random initialization.

Flow Lenia (Plantec et al., 2025, *Artificial Life* journal) demonstrates that **mass conservation** is a powerful architectural constraint: it automatically produces spatially localized entities and enables multi-species coexistence without explicit design. CyberCell should consider whether its energy system is truly conservative — energy created from photosynthesis and destroyed through metabolism should sum to zero net flow, preventing inflationary or deflationary spirals.

---

## The complexity ratchet: reaching stages 3 and 4

CyberCell's stall at Stage 2 (chemotaxis) has a specific cause identified by the Aevol platform research (Liard et al., *Artificial Life* 2020): the complexity ratchet requires **variable-length genomes** that can grow through duplication-divergence. In 75%+ of Aevol simulations, organisms evolved complexity even when simpler organisms were more fit — but only because genome expansion created sign epistasis where mutations that would simplify the organism became deleterious in the new genetic background. **CyberCell's fixed-length genomes (16 reactions in CRN, fixed topology in NN) cannot support this ratchet mechanism.** This may be the single most fundamental architectural limitation.

Constructive Neutral Evolution (Stoltzfus 1999; Muñoz-Gómez et al. 2021) provides the theoretical framework: neutral interactions between components accumulate dependencies that make simplification exponentially unlikely. Each additional dependency is individually reversible, but the random walk through dependency space overwhelmingly moves toward greater interconnection. For CyberCell, this means allowing the CRN to **add new chemicals and reactions through duplication mutations** — starting at 16 reactions but growing to 30, 50, or more as dependencies accumulate.

The PNAS 2024 study on adaptive evolutionary trajectories found that **even under constant selection pressure, 10–35% of simulations reverted from multicellularity to unicellularity.** Permanent transitions require explicit locking mechanisms: task incompatibility that makes solitary life impossible (Goldsby et al. 2023 showed division of labor drives "entrenchment" of multicellularity), bottleneck reproduction through single-cell propagules (suppressing within-organism conflict), and group-level selection competing multicellular clusters against each other.

Avida's research on complexity (Adami et al., PNAS 2000) establishes that **environmental complexity directly drives genomic complexity** — organisms must store information about their environment, and more complex environments require more stored information. For CyberCell to reach Stage 3–4, the environment must demand behaviors that single cells with fixed genomes genuinely cannot perform. The most reliable driver is **multiple limited resources requiring different processing strategies**, combined with **fluctuating environments at intermediate timescales** that maximize evolvability without preventing adaptation.

---

## Choosing between CRN and neural substrates

The literature reveals a clear division of labor between these substrates. CRN/GRN controllers excel at **bounded dynamics, natural temporal memory, robust evolvability in low-dimensional parameter spaces, and developmental/homeostatic processes**. Neural networks excel at **high-dimensional sensory processing, arbitrary function approximation, and fast reactive control.** A 2025 paper in MDPI Sensors demonstrated GRN controllers achieving superior performance to NEAT and deep Q-learning on a robot grasping benchmark while requiring ~100× fewer computational resources — but this advantage comes specifically from the GRN's bounded dynamics and sparse connectivity in a task with strong temporal coordination requirements.

CyberCell's **22× parameter disparity** (2,638 NN vs 120 CRN) is significant for evolvability. Evolutionary search scales poorly with dimensionality, and the CRN's compact representation is substantially easier to optimize through mutation alone. For Stage 3–4 goals, the CRN substrate is likely more appropriate because it naturally supports cell-cell chemical signaling, inherently models metabolic processes, and its bounded dynamics prevent the catastrophic instability that derails complex multi-cell interactions. Dylan Cope's ALIFE 2023 work successfully achieved primitive multicellularity with cell differentiation using a hybrid NN/GRN architecture.

The optimal architecture for CyberCell is likely a **layered hybrid**: the CRN handles slow homeostatic and developmental processes (metabolism, growth decisions, cell-cell signaling) while a smaller neural network (~500–800 parameters, possibly NEAT-evolved) handles fast reactive behavior (sensory processing, motor output). The CRN modulates NN parameters or gating, creating developmental context for behavior — following the neuromodulation pattern documented in the comprehensive AGRN review (Cussat-Blanc et al., *Artificial Life* 2018).

---

## Conclusion: an integrated intervention strategy

The six problems are interconnected. Fixing predation economics (Problem 6) reduces selective pressure that drives sweeps (Problem 5). Automatic parent-offspring bonding (Problem 3) creates the substrate for division of labor that drives the complexity ratchet. Continuous sigmoid outputs (Problem 1) and non-negative chemical clamping (Problem 2) restore the CRN's computational capacity. Safe mutations and MI-as-fitness-objective (Problem 4) preserve the neural genome's sensory responsiveness.

Three architectural changes would have the highest compound impact. First, **variable-length CRN genomes** — allowing chemical and reaction counts to grow through duplication — enables the complexity ratchet that is prerequisite for Stages 3–4. Second, **NEAT-style complexification for the neural genome** — starting minimal and growing — simultaneously solves the degenerate attractor problem, reduces effective genome size, and maintains sensory connectivity by construction. Third, **restructuring to 12–16 stepping-stone islands with continuous trickle migration and spatial resource heterogeneity** — this creates the population structure necessary for soft sweeps, local adaptation, and maintained diversity.

The most actionable immediate changes require no architectural refactoring: replace the 0.5 hard threshold with sigmoid probability mapping, clamp chemicals to [0, ∞) with basal production, make parent-offspring bonds automatic, cut predation energy transfer to 12%, remove the +2.0 kill bonus, and restructure migration to 1–3 cells per island per tick across 12+ islands. These six parameter-level changes address the most acute failure modes and can be implemented in the current v4.0 framework before deeper architectural work begins.