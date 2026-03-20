# Breaking the sessile optimum in artificial life simulations

**The CyberCell simulation converges to stationary photosynthesizers because its environment commits three fundamental design errors: resources don't deplete locally, predation yields are set 4–6× too low, and the population density is too sparse for organisms to interact meaningfully.** The fix requires a layered approach — local resource depletion and waste toxicity create movement pressure, increased per-kill energy absorption makes predation viable, and task incompatibility forces multicellular cooperation. Every successful ALife system that has broken sessile dominance (Avida, ALIEN, Flow-Lenia, Biomaker CA) uses some combination of these mechanisms. This report synthesizes findings from across the ALife literature, ecology research, and practitioner experience to provide specific, parameterized recommendations.

The sessile optimum is not a bug unique to CyberCell — it is the default attractor of nearly every ALife system where autotrophy is available. ALIEN users on Hacker News reported identical outcomes: "dense expanding circles of tiny slow organisms" when energy was distributed equally. The ALIEN developer Christian Heinemann solved it by gradually increasing environmental harshness until movement became mandatory. The core insight from decades of ALife research is that **organisms will always evolve the simplest viable strategy unless the environment makes that strategy insufficient**.

---

## Why every ALife system faces the sessile trap — and how the successful ones escape

The sessile optimum is a thermodynamic inevitability: a stationary organism that absorbs ambient energy has zero transport costs and minimal sensory requirements. Movement costs energy (**0.1/tick** in CyberCell), sensing requires computational overhead, and predation demands both. Unless the environment creates conditions where sitting still is actively harmful, evolution will always converge to the cheapest viable phenotype.

**Avida** escapes sessility through a clever architectural choice — organisms are self-replicating programs that gain metabolic speed bonuses by computing Boolean logic functions on environmental inputs. The critical anti-stasis mechanism is **coevolutionary parasitism**: parasites execute as parallel threads inside hosts, stealing **80% of CPU cycles** upon successful infection. A parasite infects a host only if it performs at least one matching logic function, forcing hosts to continuously evolve new computational capabilities. Zaman et al. (2014, *PLOS Biology*) showed that parasite populations retain a "genetic memory" of past host phenotypes, preventing hosts from reverting to simple strategies. Hosts coevolving with parasites achieved significantly higher functional complexity than hosts evolving alone.

**ALIEN** (Christian Heinemann's CUDA-powered simulator) uses universal **energy dissipation** — all cells continuously radiate energy, and if a cell's energy reaches zero, it dies. This creates an existential requirement for active energy gathering. The simulator's documentation describes a remarkable evolutionary trajectory: dense colonies initially lost movement capability (sessile optimum), but when the developer gradually increased universe size and weapon energy costs, "suddenly there is a necessity for adaptation. The replicators develop moving capabilities in order to consume resources more actively." Further parameter increases produced aggressive movement patterns and boom-bust ecological dynamics.

**Flow-Lenia** (Plantec et al., 2023/2025, *Artificial Life*) solved sessility through **mass conservation** — creatures lose mass continuously and must consume spatially distributed food to survive. The developers observed a striking emergent behavior: "the bottom left creature is static at the start, but, when its mass decays it changes its shape and grows a sort of 'mouth' allowing it to head towards food and consume it." **Biomaker CA** (Randazzo & Mordvintsev, Google Research, 2023) uses dual-resource requirements — organisms need both earth nutrients (from roots) and air nutrients (from leaves), forcing morphological complexity. Cell aging in the "pestilence" configuration ensures no organism survives longer than ~200–300 steps, preventing indefinite sessile dominance.

The pattern across all successful systems is clear: **passive energy income must be insufficient, existence must have ongoing costs, and the environment must change faster than organisms can re-equilibrate**.

---

## The three critical flaws in CyberCell's current design

### Flaw 1: Resources don't deplete locally

CyberCell's **25,000-tick deposit relocation interval** is approximately **25× the likely organism generation time** — far too slow to prevent re-equilibration. Research on Avida's fluctuating environments (Lalejini et al., 2021, *Frontiers in Ecology and Evolution*) found that environmental switching every **~30 generations** (roughly 1× generation time) produced the optimal balance of adaptation and evolvability. Switching every 300 generations allowed re-equilibration to stasis — and CyberCell's interval is even slower than that.

More fundamentally, CyberCell lacks **local resource depletion**. In real ecology, diffusive depletion zones are universal around sessile organisms — plant roots deplete phosphorus **5–10×** within 1–3mm of their surface (Kuzyakov, 2013). Lee, Kempes, and West (2021, *PNAS*) showed that resource competition in sessile organisms is governed by area overlap between neighbors' depletion zones. Without this mechanic, CyberCell's bright zone is an infinite buffet — there is no tragedy of the commons because the commons never depletes.

The fix: implement **Beer-Lambert light attenuation** where each photosynthesizing cell reduces local light for neighbors. The canonical model from plant canopy ecology uses `effective_light = base_light × exp(-k × local_density)`, where k ≈ 0.5 per neighbor within radius 2. With five neighbors, this produces **92% light reduction** — transforming a dense cluster from paradise to poverty. With three neighbors, the reduction is 78%. This single change makes photosynthesis self-limiting and creates natural spacing pressure.

### Flaw 2: Predation energy transfer is 4–6× too low

CyberCell's **12% energy absorption** per predation event conflates two very different ecological concepts. The "10% rule" (Lindeman efficiency) describes **population-level** energy transfer between trophic levels — a statistical outcome of search costs, failed hunts, metabolic overhead, and competition. But **per-kill assimilation efficiency** in real predators is **50–80%** — a wolf absorbs roughly 80% of an elk carcass's energy content. The 10% population-level efficiency emerges naturally from all the costs surrounding each kill, not from low per-kill absorption.

At 12% per-kill absorption, a predator must kill roughly 8 prey to gain the energy equivalent of one prey organism's lifetime photosynthetic income — an impossible proposition when the hunter must also pay movement costs for each pursuit. Zhou et al. (2025, *Ecology*) showed that "energy transfer efficiency along the food chain, rather than the total amount of energy fixed by primary producers, determines the strength of trophic cascades." Even small increases in per-kill efficiency dramatically change food web dynamics.

The fix: **increase per-kill energy absorption to 30–50%**. This makes individual predation events profitable while still allowing the ~10% population-level efficiency to emerge naturally from search and metabolic costs. For reference, Avida's parasites steal 80% of host resources — far more generous than CyberCell's 12%.

### Flaw 3: Population density is too sparse for interaction

CyberCell runs **~3,500 cells on a 500×500 grid**, yielding **~1.4% occupancy**. This is extraordinarily sparse. Avida typically runs at 100% grid occupancy. At 1.4% density, organisms almost never encounter each other — there is no selection pressure for social behavior, predator-prey dynamics, or information processing. The near-zero mutual information between sensors and actions is a direct consequence: organisms don't need to sense anything because there is nothing consequential to sense.

The fix: target **5–20% grid occupancy** (12,500–50,000 organisms). This ensures frequent neighbor encounters, meaningful resource competition, and sufficient population size for evolutionary dynamics (~10,000+ for genetic diversity). SLiM spatial simulation guidelines indicate that when carrying capacity per local patch drops below ~5 individuals, populations cannot self-sustain.

---

## How to create genuine pressure for movement through waste and depletion

The most elegant mechanism for forcing motility comes from **Froese, Virgo, and Ikegami (2014, *Artificial Life*)**, who demonstrated that waste accumulation alone is sufficient to evolve movement — without any other environmental pressure. Using a modified Gray-Scott reaction-diffusion system, they introduced a waste product P that accumulates as a metabolic byproduct and inhibits local metabolism via `exp(-w × P)`. With slow waste decay (kp = 0.0002), waste accumulates faster than it disperses. The result: dissipative structures spontaneously developed motility, literally moving away from their own waste. This is perhaps the most direct demonstration that **niche destruction drives niche exploration**.

For CyberCell, the recommended waste implementation combines three mechanisms working in concert:

- **Local resource depletion**: each photosynthesizing cell consumes local energy at 0.05/tick while natural regeneration provides only 0.02/tick. A single organism barely sustains itself; groups of 3+ rapidly deplete their location.
- **Waste production**: each cell produces waste proportional to energy harvested (0.01 × energy gained per tick). Waste decays slowly (multiplied by 0.998/tick, giving a half-life of ~350 ticks) and diffuses at 5% per tick to adjacent cells.
- **Waste toxicity**: damage = 0.05 × max(0, waste_level − tolerance_threshold) per tick. This creates expanding toxic zones around sessile colonies that grow faster than waste can dissipate.

The combination creates a "use it and lose it" dynamic — the act of photosynthesizing degrades the local environment, rewarding organisms that periodically relocate. Bacterial chemotaxis research supports this: Cremer et al. (2019, *Nature*) showed that *E. coli* uses chemotaxis to expand into unoccupied territories **before** nutrients run out — a form of "navigated range expansion" that gives populations ecological foresight. Ni et al. (2020, *PNAS*) found that bacteria increase investment in motility under poor nutritional conditions proportional to the reproductive fitness advantage of chemotaxis.

The resource relocation interval should drop from 25,000 ticks to **1,000–2,000 ticks** (0.5–2× generation time), but continuous organism-driven depletion matters more than periodic relocation. Depletion that scales with population density creates emergent environmental change that self-calibrates — no manual tuning of relocation intervals needed.

---

## Dark zones and seasonal famine make predation inevitable

Predation will not evolve simply by making it more efficient — the environment must create conditions where heterotrophy is the **only viable strategy** for some organisms. Two mechanisms achieve this reliably.

**Spatial light heterogeneity with dark zones** creates obligate heterotrophy niches. Deep-sea hydrothermal vent ecosystems demonstrate this principle: with zero light at >200m depth, complex food webs with 3–4 trophic levels develop around chemosynthetic primary production. NSF-funded research at Gorda Ridge found that protists consume **28–62% of daily bacterial biomass** at vents. For CyberCell, allocating **15–25% of the grid as zero-light "dark zones"** adjacent to bright zones creates a gradient: bright zones support pure autotrophy, twilight borders support mixotrophy, and dark interiors force obligate predation. Organisms that evolve predatory capability in dark zones may then invade bright zones where prey is more abundant, triggering arms races.

**Seasonal light cycles** create temporal windows where photosynthesis cannot sustain organisms. Implement sinusoidal light variation where intensity drops to **10–20% of maximum** during "winter" phases, lasting long enough that organisms cannot survive on stored energy alone. Himeoka and Mitarai (2020, *Physical Review Research*) showed that feast-famine cycles drive populations to optimize the ratio of growth rate to death rate, while Seid et al. (2024, *Current Biology*) found that 900 days of extreme feast-famine cycling in *E. coli* produced convergent adaptive trajectories with high mutational parallelism. The critical design principle: **famine duration must exceed organisms' energy storage capacity**, making alternative metabolic strategies (predation, scavenging) the only path to survival during lean periods.

Li et al. (2025, *Oikos*) provide an important caveat: under extremely limiting nutrient levels, mixotrophs actually become **more autotrophic**, not less, because prey biomass is too scarce. Predation evolves at **intermediate resource levels** — autotrophic income insufficient for optimal reproduction but prey density high enough to make hunting worthwhile. This Goldilocks zone requires both sufficient prey population (~5–20% grid density) and constrained but not eliminated photosynthesis.

---

## Task incompatibility and predation jointly drive multicellularity

Two experimentally validated mechanisms reliably produce multicellularity: **task incompatibility** and **size-dependent predation resistance**. Neither alone is as effective as both combined.

**Goldsby et al. (2012, *PNAS*; 2014, *PLOS Biology*)** demonstrated that when organisms must perform multiple essential tasks and switching between them imposes costs, cells evolve division of labor. In Avida, organisms earning rewards for computational logic functions evolved to distribute sub-problems among neighboring cells via message-passing when task-switching incurred time penalties. The 2014 extension showed that making metabolic work **mutagenic** (per-site mutation probability ~0.00075 when performing certain functions) drove reproductive division of labor — some cells became soma (performing 88–99% of mutagenic work via a `block_propagation` instruction) while others remained protected germ cells. For CyberCell, making photosynthesis and defense (or any two essential functions) **mutually exclusive** at the single-cell level creates irreducible pressure for specialization.

**Size-dependent predation** is the most experimentally replicated driver of multicellularity. Boraas et al. (1998) showed *Chlorella vulgaris* evolved heritable 8-celled colonies within **<100 generations** when exposed to the flagellate predator *Ochromonas vallescia* — colonies were virtually immune to predation. Herron et al. (2019, *Scientific Reports*) replicated this with *Chlamydomonas reinhardtii* evolving multicellular structures within ~750 generations under *Paramecium* predation. Their 2025 follow-up (*Genome Biology & Evolution*) found that 4 of 12 predation-selected populations evolved multicellularity, with every multicellular isolate arising from one of only two founder genotypes — showing that genetic predisposition interacts with selection pressure.

The snowflake yeast system (Ratcliff et al., 2012–2023) provides the most dramatic demonstration. Daily gravitational selection (only cells that settle fast enough survive) produced multicellular "snowflake" clusters in all 10 replicates within ~300 generations. The remarkable MuLTEE experiment (Bozdag et al., 2023, *Nature*) extended this to 600 rounds of selection: anaerobic yeast evolved to **20,000× larger size** (millimeter scale, macroscopic), with **10,000-fold increased toughness**, through evolved cellular elongation and branch entanglement. This is the first de novo evolution of macroscopic multicellularity from a unicellular ancestor.

For CyberCell, the implementation should combine these pressures. Introduce predator agents (or periodic predation events) with **size-gape limitation**: predation probability = max(0, 1 − cluster_size/threshold), where threshold = 4–8 cells. Simultaneously, make essential tasks trade off against each other so that single cells cannot excel at both. Groups whose collective task profile covers all essential functions out-reproduce specialists and generalists alike.

---

## Collective sensing only matters when the environment demands navigation

CyberCell already has sensory noise at σ=0.15, but near-zero mutual information between sensors and actions reveals that organisms have no reason to process sensory input. Colizzi, Vroomans, and Merks (2020, *eLife*) provide the key insight: using a Cellular Potts Model where cells perform chemotaxis in a shallow, noisy gradient — with **no explicit multicellularity advantage built in** — multicellular aggregates evolved because they navigate noisy gradients more efficiently than single cells. Cells that adhere transfer gradient information in a self-organized manner. But this only works when the environment changes frequently enough that navigation matters: when environments change too rapidly, unicellular dispersal wins instead.

Berdahl et al. (2013, *Science*) demonstrated the same principle in schooling fish: individual fish largely incapable of detecting light gradients, but groups locate preferred regions with increasing accuracy as group size increases. The mechanism is social — individuals perceive gradients through neighbor interactions rather than direct sensing.

For CyberCell to benefit from collective sensing, three conditions must hold simultaneously: resources must be **spatially heterogeneous and temporally changing** (patches appear at random locations each season), organisms must be **motile** (so sensing translates to fitness), and gradients must be **shallow enough** that single cells cannot reliably detect them (σ > signal strength across one cell width). Season length must exceed the time required for a group to navigate to a new patch — too-short seasons favor fast-dispersing unicellular strategies.

---

## Environmental ratchets and open-ended evolution require calibrated disruption

**POET** (Wang, Lehman, Clune, Stanley; GECCO 2019 Best Paper) demonstrates the gold standard for preventing evolutionary stagnation through co-evolving environments. The system maintains a population of environment-agent pairs, continuously generating new environments at the edge of current capability via a **Minimal Criterion** — new challenges must be neither trivially easy nor impossibly hard. The key innovation is **cross-pollination**: solutions trained in one environment are tested in others, enabling "stepping stones" where skills learned in context A unexpectedly solve context B. Enhanced POET (ICML 2020) showed continuously increasing novel environment conquests without plateau, while the base optimizer alone could not solve late-stage environments from scratch.

**OMNI-EPIC** (Faldor et al., ICLR 2025) extends this by using foundation models to generate both environment code and reward functions, filtering for **learnability** (within the agent's zone of proximal development) and **interestingness** (qualitatively novel, not trivial variations). For CyberCell without foundation models, approximate this by tracking which environmental configurations produce measurable behavioral adaptation and preferring structurally novel configurations.

The **complexity ratchet** (Liard et al., 2020, *Artificial Life*) provides a complementary mechanism: once mutations leading to complex solutions are fixed, mutations toward simpler solutions become deleterious due to epistatic interactions. This creates a one-way ratchet — organisms almost never revert to simplicity. Combined with Poelwijk et al.'s (2011, *PNAS*) **trade-off ratchet** — where environmental fluctuations break adaptive stasis by repositioning fitness peaks — these mechanisms create progressive escalation that the sessile optimum cannot resist.

Catastrophic events play a specific role: Lehman and Miikkulainen (2015, *PLOS ONE*) showed that extinction events killing organisms in 90% of niches produced higher evolvability in survivors. **Pulse extinctions** (sudden kills) allow faster recovery than press extinctions (prolonged scarcity). Recommended parameters: catastrophes every **1,000–5,000 ticks**, killing **50–90%** of the population, with stochastic rather than periodic timing to prevent organisms from evolving dormancy strategies.

---

## Concrete parameter recommendations for CyberCell

The following table synthesizes all findings into specific, implementable parameters for the 500×500 grid:

| Parameter | Current value | Recommended value | Priority |
|---|---|---|---|
| Population density | ~1.4% (3,500 cells) | 5–20% (12,500–50,000) | Critical |
| Per-kill predation absorption | 12% | 30–50% | Critical |
| Resource relocation interval | 25,000 ticks | 1,000–2,000 ticks | Critical |
| Local light depletion | None | Beer-Lambert: exp(−0.5 × neighbors in r=2) | Critical |
| Waste production rate | None | 0.01 × energy_gained/tick | High |
| Waste decay rate | None | ×0.998/tick (half-life ~350 ticks) | High |
| Waste toxicity | None | 0.05 × max(0, waste − threshold)/tick | High |
| Dark zone coverage | 0% | 15–25% of grid | High |
| Seasonal light variation | None | Sinusoidal, min 10–20% of max | High |
| Size-dependent predation resist. | None | P(eaten) = max(0, 1 − size/8) | Medium |
| Task incompatibility | None | 2 essential mutually exclusive tasks | Medium |
| Catastrophe frequency | None | Every 1,000–5,000 ticks, kill 50–90% | Medium |
| Habitable grid fraction | ~100% | 60–80% with geographic structure | Medium |
| Resource patch diameter | Unknown | 5–30 cells, Perlin noise distribution | Medium |
| Movement cost ratio | 0.1/tick absolute | 5–15% of per-food-item energy gain | Low |

The implementation priority order matters. **Phase 1** (highest impact): add local resource depletion via Beer-Lambert shading, reduce relocation interval to 1,000–2,000 ticks, and increase population density to 5–20%. These three changes alone should break the static photosynthesis equilibrium. **Phase 2**: add metabolic waste accumulation with slow decay, increase predation absorption to 30–50%, and create dark zones covering 15–25% of the grid. This introduces heterotrophy pressure. **Phase 3**: add size-dependent predation and task incompatibility to drive multicellularity, plus seasonal light cycles and periodic catastrophes for open-ended dynamics.

## Conclusion

The sessile optimum is not a single problem requiring a single fix — it is a symptom of an environment that fails to impose escalating demands on its inhabitants. CyberCell's three critical gaps (no local depletion, unviable predation economics, insufficient density) each independently reinforce sessility. The most important insight from the literature is that **organism-driven environmental degradation** — waste accumulation and resource depletion that scales with population density — is more powerful than externally imposed environmental change, because it self-calibrates and creates emergent spatiotemporal dynamics without parameter tuning.

The Froese, Virgo, and Ikegami (2014) result deserves special emphasis: waste accumulation alone, modeled as metabolism inhibition via exp(−w × P), produced spontaneous motility in a reaction-diffusion system with no other environmental pressure. This is the most direct path from sessile to motile in CyberCell's chemical reaction network framework. Combined with Beer-Lambert light competition (which makes photosynthesis self-limiting at density), realistic predation economics (30–50% per-kill absorption), and spatial light heterogeneity (dark zones forcing heterotrophy), these changes should progressively unlock movement, predation, cooperation, and sensory-driven behavior — not because the simulation mandates them, but because the environment makes the sessile strategy insufficient.