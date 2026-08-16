# Research direction: language-grounded executable world models

## Durable question

Under what inductive biases and representational interfaces can an interactive agent infer,
from sparse language-mediated observations and interventions, a reusable, uncertainty-aware,
executable model of how its environment changes—and transfer that model across mechanics and
language while retaining enough structure to diagnose and verify its reasoning?

## Status after V63r1 (2026-08-16)

The original staged core is now implemented through bounded verification. V50 established exact
history-dependent belief filtering; V53r2 calibrated continuous-parameter SMC² against exact
quadrature and offline PMCMC; V54 validated exact expected-information-gain intervention
selection; V55r1/V59 added delayed-consequence and budgeted observation-contingent planning; V60
showed that the approximate posterior preserves the registered planning decisions; and V61 used an
independent executor plus Storm to verify the bounded execution semantics of all 72 frozen
horizon-3/5/7 policies.

V62 then moved outside the project's internal symbolic DSL. A fresh exact parser, belief filter,
and finite-horizon planner were evaluated on pinned POBAX Tiger and T-Maze models, with 24 official
runtime rollout cells. The immutable V62 run passed 31 of 32 gates but failed a Bellman-residual
measurement because the checker recomposed a reward after an absorbing terminal state. V62r1
preregistered and mutation-tested only that terminal base-case correction. Its one immutable-node
rescore passed all 12 repair gates, with maximum corrected residual `1.78e-15`, zero terminal
residual, and no changes to the other 31 gates or any exact/rollout record. V62 remains recorded as
a failed run; the combined V62/V62r1 evidence supports only measurement-repaired, exact,
finite-state, finite-horizon transfer on those three pinned external models.

The human-authored language track remains deferred under V58 because the required independent
writers, validators, adjudicator, and coordinator are unavailable. Synthetic records must not be
substituted for those roles.

V63 then tested external system identification on a project-authored unknown-dynamics layer anchored
to pinned POBAX Tiger arrays. The hidden family combined a discrete persistent-versus-alternating
identity with a continuous transition parameter, and compared the frozen SMC² implementation with
exact inference, SBC, scale stress, and the official external runtime. The immutable V63 run passed
every gate except mean joint identity-parameter TV (`0.0731` against `0.06`). A post-result audit
localized that failure to the evaluator: unlike the frozen V53r2 reference rule, it scored three
repeat posteriors separately instead of first forming their equal-weight mixture.

V63r1 preregistered only that aggregation repair and reused the original populations, posterior
repeats, SBC, scale, runtime artifact, budgets, and gates by hash. Its single repair evaluation
passed all gates: mean joint TV was `0.0444`, q95 joint TV was `0.0625`, mean identity TV was
`0.000372`, and mean theta Wasserstein distance was `0.00214`. V63 remains a failed run and V63r1 is
a measurement repair, not an independent replication. The evidence qualifies calibrated SMC²
portability for this one externally anchored, project-authored family; POBAX itself does not supply
unknown dynamics.

## Next experimental direction

The next useful technical question is external **active system identification**. V64 should use a
separate pinned external model with at least two informative nonterminal interventions and a
preregistered unknown-dynamics layer. Before constructing an evaluation population, it must prove
that the interventions are structurally distinguishable, that the expected-information-gain
maximizer varies across reachable histories, and that the hidden uncertainty changes a downstream
decision. Exact EIG should then be compared with prospectively frozen fixed and random designs under
matched interaction budgets. Model variants, priors, horizons, seeds, controls, aggregation rules,
and noncompensatory gates must be frozen before any candidate evaluation.

Tiger is ineligible as the substantive active-design benchmark because `listen` is its only
informative nonterminal action. The current outcome lock authorizes only V64 design and
preregistration; it does not yet authorize population construction or evaluation. An exact V64 pass
would qualify the benchmark and acquisition reference only. The next substantive portability gate
would then compare EIG computed from the deployed pooled three-repeat SMC² posterior with exact EIG
at multiple inference budgets, while reporting single-repeat regret and computational cost. Only a
pass there should unlock Bayes-adaptive reward decisions.

Only after that external identification-to-decision bridge passes should the project spend effort
on larger POMDP benchmark suites, approximate long-horizon optimality, or learned representations.
Formal safety remains a separate specification task: bounded value verification is not a safety
property. The V58 language branch should remain paused until real independent participants exist.

The initial project addressed a deliberately narrower instance of the durable question. It grounded
a declared Boolean state ontology, inferred one-step executable outcome programs, and answered
queries under partial state information. The historical evidence below records how that starting
point motivated the later sequential, probabilistic, active, planning, and verification stages.

## Current evidence

The V14–V19 sequence supports three conclusions.

1. Modular decomposition has scientific value. It separated evidence matching, temporal status,
   polarity, schema induction, and execution, and repeatedly converted apparent capacity failures
   into testable support or interface failures.
2. Under V19's registered supported-language interface, the frozen V15 grounder and unchanged V18
   inducer compose exactly across all 40 development episodes in every oracle/frozen condition.
3. Under paired novel ontologies, hard support-grounding errors remove the target behavior from
   exact search. The current weakness lies at the grounding-to-induction certainty boundary; this
   does not yet establish that the frozen representation itself is insufficient.

The immediate confirmatory claim is therefore limited to population transfer across unseen
one-step mechanics under a declared Boolean ontology and supported language. Arbitrary ontology
learning, sequential dynamics, causal discovery, and general world-model learning are excluded.

## Research commitments

The durable commitments are explicit state, executable dynamics, mechanic-level evaluation,
intervention-based identification, uncertainty that propagates across module boundaries, and
verification wherever the model class permits it. Exact finite enumeration is an implementation
choice, not a permanent commitment.

Neural components may propose representations and hypotheses. Structured inference should remain
authoritative about the semantics of discrete candidates. When exact verification becomes
impossible, it should be replaced by explicit probabilistic or statistical guarantees—not an
uncalibrated confidence score.

## Staged program

1. **Population replication.** Test the unchanged hard modular system over a sealed population of
   mechanics drawn from several declared construction families. Mechanics, not queries, are the
   inferential unit.
2. **Probabilistic interfaces.** Preserve calibrated alternative groundings and propagate them into
   executable induction. Test whether this repairs ontology transfer without indiscriminately
   widening answer sets.
3. **Relational state.** Replace a fixed determinant vector with typed entities, attributes, and
   relations, and test extrapolation to new bindings, relation graphs, and entity counts.
4. **Sequential belief dynamics.** Introduce partial observations, stochastic or delayed effects,
   and persistent next-state change.
5. **Active identification.** Let the agent choose interventions that distinguish remaining models
   and compare sample efficiency with fixed or random traces.
6. **Open concepts and natural interaction.** Introduce concepts through definitions, examples,
   relations, and interventions; later replace generated surfaces with independently authored
   interaction.
7. **Matched architectural challenge.** Compare the evolved probabilistic structured system with a
   joint intervention-aware relational neural world model under the same information and interaction
   budgets, while accounting explicitly for privileged structural supervision.

The stages are falsification gates, not an obligation to preserve the current architecture.

## Continuation, pivot, and stop criteria

Continue the structured modular program if mechanic-level transfer replicates across construction
families, probabilistic grounding repairs ontology shift without destroying identifiability,
relational structure improves extrapolation, and active selection reduces the interactions required
to identify mechanics.

Demote the symbolic model to a verifier or planner if a joint relational model consistently learns
better state and dynamics representations while extracted structure remains useful for checking
plans, constraints, or causal hypotheses.

Pivot toward joint representation learning if, at matched budgets, it dominates on new ontologies,
entity counts, noisy sequential dynamics, and natural language, and its latent state remains stable
under independent interventions.

Stop treating executable induction as a central hypothesis if its advantages disappear when oracle
concepts, clean supports, fixed entity sets, and bounded DSLs are removed, or if each domain needs a
hand-authored grammar that already encodes its essential solution. Stop the broad external claim if
performance inside shared generators repeatedly fails to predict independently generated or natural
interaction.

## Fine-tuning eligibility

Weight adaptation is eligible only after multiple ontology families show a systematic,
high-confidence representation failure despite valid semantic support, calibrated alternative
groundings, definitions or retrieval, and localized readout tests. Any training must remain confined
to development ontologies and improve downstream schema retention on separately sealed ontologies
without degrading supported operators or surfaces.

## Load-bearing sources

| Source | Status | Direct relevance | Important boundary |
|---|---|---|---|
| [Harnad, *The Symbol Grounding Problem*](https://doi.org/10.1016/0167-2789(90)90087-6) | Peer-reviewed (1990) | Distinguishes grounding supplied symbols from explaining where a symbol inventory comes from. | Does not provide a dynamics-learning algorithm. |
| [Ljung, *System Identification: Theory for the User*](https://www.control.lth.se/fileadmin/control/Education/DoctorateProgram/SystemIdentification/2018/Ljung--System_Identification_Theory_for_the_User.pdf) | Textbook | Centers model class, noise, experimental design, and identifiability. | Does not address language grounding. |
| [Yang, Wu, and Jiang, ARMS](https://doi.org/10.1016/j.artint.2007.01.005) | Peer-reviewed (2007) | Learns symbolic action models from incomplete plan examples. | Assumes symbolic predicates and actions. |
| [Lamanna and Serafini, NOLAM](https://ojs.aaai.org/index.php/ICAPS/article/view/31493) | ICAPS 2024 | Infers posterior action-model components from noisy traces. | STRIPS-style state differs from V19 language grounding. |
| [Lake, Salakhutdinov, and Tenenbaum, Bayesian Program Learning](https://doi.org/10.1126/science.aab3050) | Peer-reviewed (2015) | Shows the few-shot value of strong compositional program priors. | Success depends on a well-specified domain language. |
| [Ellis et al., DreamCoder](https://arxiv.org/abs/2006.08381) | Published version linked from preprint | Combines neural search guidance with reusable learned program abstractions. | Demonstrations are small synthesis domains, not interactive world models. |
| [Locatello et al., unsupervised disentanglement](https://proceedings.mlr.press/v97/locatello19a.html) | ICML 2019 | Shows why latent semantic variables require assumptions or supervision. | Impossibility is not a claim against intervention-aided learning. |
| [Lippe et al., CITRIS](https://proceedings.mlr.press/v162/lippe22a.html) | ICML 2022 | Uses temporal sequences and known intervention targets for identifiable causal factors. | Requires assumptions absent from the current language-only setting. |
| [Lippe et al., BISCUIT](https://proceedings.mlr.press/v216/lippe23a.html) | UAI 2023 | Learns causal variables and unknown binary interaction variables jointly. | Evidence comes from robotic-inspired visual environments. |
| [Kansky et al., Schema Networks](https://proceedings.mlr.press/v70/kansky17a.html) | ICML 2017 | Demonstrates transferable object-centered causal schemas. | Does not solve open-vocabulary language grounding. |
| [Battaglia et al., Interaction Networks](https://arxiv.org/abs/1612.00222) | NeurIPS 2016 workshop / preprint | Motivates object/relation factorization across changing configurations. | Prediction structure is not automatically semantic or verifiable. |
| [Zhao et al., HOWM](https://proceedings.mlr.press/v162/zhao22b.html) | ICML 2022 | Formalizes compositional generalization for object-oriented world models. | Tests a controlled object-library population. |
| [Koh et al., Concept Bottleneck Models](https://proceedings.mlr.press/v119/koh20a.html) | ICML 2020 | Supports intervenable structured concept interfaces. | Supplied concepts can leak or omit predictive information. |
| [Hernandez Cano et al., SWMPO](https://proceedings.mlr.press/v267/hernandez-cano25a.html) | ICML 2025 | Learns structured finite-state abstractions for sequential decisions. | Does not establish open language grounding or relational causal state. |
| [Mosbach et al., SOLD](https://icml.cc/virtual/2025/poster/44962) | ICML 2025 | Evidence that object-centric latent dynamics can outperform holistic baselines in relational manipulation. | A visual-control result, directly analogous only to a later stage. |
| [Markham et al., intervention-based composable representations](https://openreview.net/forum?id=4P08CBsSw7) | Withdrawn ICLR 2026 submission | A useful challenger hypothesis for expressive joint models with intervention structure. | Provisional evidence; it must not be cited as an established conference result. |

## Immediate decision

Preregister the population final before constructing it. Develop the uncertainty-aware challenger
only on exposed V15/V19 artifacts. Freeze both systems, the generator, delayed seed rule, metrics,
decision hierarchy, and evaluation implementation before any final record is materialized or any
final model feature is extracted.
