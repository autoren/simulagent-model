# Research direction: language-grounded executable world models

## Durable question

Under what inductive biases and representational interfaces can an interactive agent infer,
from sparse language-mediated observations and interventions, a reusable, uncertainty-aware,
executable model of how its environment changes—and transfer that model across mechanics and
language while retaining enough structure to diagnose and verify its reasoning?

## Status after V70 (2026-08-17)

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

V64 then moved active identification to pinned POBAX `4x3_nonterminating.POMDP` arrays with a
project-authored actuator-failure family. Its two hidden identities map failed commands clockwise or
counterclockwise, while a continuous transition parameter controls commanded-action reliability.
The exact reference showed that all four interventions can be strict EIG maximizers across reachable
histories and that the remaining uncertainty changes the downstream reward-optimal action in 4 of
11 states. Independent scalar checks agreed with the candidate implementation to numerical
precision, all 13 implementation mutants and all six evaluation controls were detected, and the
sole immutable evaluation passed every noncompensatory gate.

At interaction budget 8, exact adaptive EIG achieved mean latent-information gain `0.2561`, versus
`0.1700` for the fixed cycle and `0.1820` for uniform random selection. Paired lower 95% bounds for
the adaptive advantages were `0.0608` and `0.0502`, respectively. Candidate one-step EIG agreed
with the independent reference to `5.42e-16`; the selected action was always an exact optimizer;
and SBC, access, one-shot, source-binding, and truth-firewall checks all passed. This qualifies the
external benchmark and exact acquisition reference only. It does not yet establish that the
deployed approximate posterior preserves acquisition decisions.

## External inference-to-confirmation sequence

V65r1 attempted the paired SMC²-to-exact acquisition-portability test after prospectively sealing 48
public histories and freezing a Rao–Blackwellized acquisition repair. Its sole immutable evaluation
did not produce accuracy results. One identity-conditioned outer filter reached zero likelihood for
every particle and the implementation aborted before writing any record-budget cells. V65r1 is
therefore frozen as a failed run and does not authorize Bayes-adaptive reward decisions.

An exact support-only post-failure audit found one guaranteed fatal sealed history. It has positive
likelihood under the clockwise-failure identity and exactly zero likelihood under the
counterclockwise-failure identity. The joint posterior is well-defined and should assign the latter
identity zero mass; the frozen code incorrectly treats extinction of either identity branch as
extinction of the full model. No V65r1 rerun or post-failure candidate EIG scoring occurred.

V65r2 preregistered that narrow repair, but was rejected during implementation development. One
unit test called the candidate four-action EIG scorer on the sealed fatal record, contrary to its
explicit implementation-stage firewall. No values were printed or used, no implementation lock was
written, and V65r2 had zero evaluation attempts, but the access claim cannot be retained.

V65r3 retained the same narrow repair with a synthetic-only implementation EIG firewall and a
durable atomic attempt protocol. Its one immutable evaluation completed all 144 record-budget rows
and 432 repeat cells and passed every original gate. At the primary 509-particle budget, mean/q95
four-action EIG error was `0.000188/0.000688` nats, strict/ε-optimal membership was `0.979/1.000`,
and mean selection regret was `1.88e-9` nats. Seven controls were detected or dominated. All nine
expected exact-zero identity branches were handled with zero mass and no atom, with no
positive-support particle collapse. An independent raw-cell reaggregation reproduced the result.

V66 then tested bounded-horizon Bayes-adaptive reward decision quality on the unchanged 48 public
histories. Fresh 509-particle, 127-inner-particle, three-repeat SMC² posteriors were pooled before
horizon-three planning, and every resulting contingent policy was evaluated under the exact joint
posterior predictive. The sole immutable run passed every gate: mean/q95/max exact value regret was
`4.63e-18/2.78e-17/2.78e-17`, strict and epsilon-optimal root membership were both `1.000`, mean
root-Q error was `3.27e-18`, and mean/q95 self-value calibration error was
`0.000199/0.000570`. Five controls were detected or dominated, and independent raw-cell
reaggregation reproduced the result.

The control boundary matters. MAP certainty equivalence and the first repeat alone selected the same
root actions and attained the same exact values on these records. The valid persistent
posterior-sampling mixture was also not dominated under the registered rule, although its mean
regret was `0.00245`. Information-only and myopic policies were materially worse. V66 therefore
shows preservation by the deployed pooled posterior, not that pooling or Bayes adaptation is
strictly necessary for every short-horizon reward decision in this family.

V67 independently verified the frozen V66 execution layer. A new parser and exact posterior
executor reconstructed the pinned arrays, quadrature, persistent actuator family, public-history
conditioning, reachable observation branches, and discounted source rewards without calling the
V62 parser, V64 filter, V66 planner, or V66 evaluator. Storm 1.13.0 then completed 192 property
checks over 96 sealed DTMCs. Termination error was zero; maximum Storm-to-independent return error
was `5.55e-16`; maximum independent-to-V66 value error was `5.00e-16`; and the exact-minus-SMC²
paired result was reproduced within `2.78e-17`. Every binding, totality, normalization, deadlock,
finite-value, mutation, one-shot, and access gate passed.

The external identification-to-decision-to-verification bridge is therefore closed for this one
pinned family and horizon. V67 verifies archived policy execution, not the V66 optimizer itself,
and remains posterior-expected bounded model checking rather than a safety property or
infinite-horizon proof.

V68 then screened the unchanged V64 command-channel uncertainty family for exact decision
sensitivity on 59 records from four previously exposed development models. Its first attempt stopped
before persisting any result when a posterior-sampling point policy encountered an observation with
zero probability under its selected model but positive probability under the full mixture. V68r1
prospectively totalized that control and then exposed the same mathematical partiality in the MAP
point policy, again before any record result or confirmatory-model access. V68r2 locked one common,
non-smoothing totalization rule for both point controls and completed the full development census.

V68r2 was a valid negative result. Exact Bayes-adaptive and MAP root actions disagreed on zero
records, neither MAP nor posterior sampling had a material-regret record, and maximum normalized
MAP regret was `0.00151`, below the frozen `0.01` gate. The family stopped without scoring any
confirmatory model. The negative outcome motivated a materially different dominant latent
action-remapping family, which V69 preregistered and evaluated only on the same four exposed models.
V69 passed all development gates: it produced eight BA/MAP root-action disagreements, eight
material MAP-regret records, sixteen material posterior-sampling records, and maximum normalized
MAP regret `0.0276`. It authorized only prospective confirmatory locking; no holdout was scored.

V70 then applied the unchanged V69 family and frozen reporting hierarchy to a sealed census of 244
records from nine previously untouched, externally sourced POMDP base models. The single evaluation
retained every record and passed all 22 gates. Four models met the paired action-disagreement and
material-MAP-regret criterion, where three were required; the qualifying set contained two of three
structurally related and two of six novel models. Six models showed material posterior-sampling
regret, and maximum normalized MAP regret was `0.09545`. The independent outcome audit reproduced
every aggregate and gate and verified the one-shot, census, source, and access chain.

The evidence is deliberately tiered. `4x3.POMDP` and the novel `network.POMDP` qualified without the
MAP control ever entering its totalization rule, establishing that the central effect is not solely
a fallback artifact. The qualifying `fully_observable_tmaze2.POMDP` and `hallway.POMDP` effects
overlap totalization and support the complete locked point-control comparison, not an isolated
theorem about MAP collapse. The two cheese models are Tier B only: they showed material point-control
regret but no first-action disagreement, and every material effect overlapped totalization.

The V68r2–V70 sequence therefore confirms a finite-horizon decision benefit for retaining the exact
posterior over one project-authored latent action-remapping family across multiple external base
environments. It does not establish an externally supplied uncertainty family, approximate-inference
portability, long-horizon scaling, or unrestricted real-world control. The canonical synthesis is
recorded in `docs/v68r2-v70-development-confirmation-synthesis.md`.

The next direction is synthesis and fresh boundary testing. The V69 family and all V70 models are
closed for development: they must not be rerun, rescored, tuned, or converted into a new development
set. Any broader study must preregister a materially different uncertainty family and fresh
development and confirmatory models before outcomes are computed. Fallback-free and complete-
procedure effects should be separate registered estimands. Approximate inference, longer horizons,
human interaction, and model or adapter experiments remain separate future programs with their own
feasibility and validation gates. The V58 language branch remains paused until real independent
participants exist.

V71 performed that fresh boundary test with a binary latent observation-label codebook rather than
action or transition remapping. The construction mixed canonical and reversed source observation
labels at a prospectively fixed reliability of `0.85`; its two point models had identical support by
construction, so fallback was forbidden. A new official `pomdp-solve` source census produced three
fresh development models and five protected confirmation models after excluding prior-domain reuse
and malformed or non-normalized files without repair. Source, resource, census, and evaluator locks
were completed before the one authorized development outcome.

The V71 development result was cleanly negative. All 21 root/depth-1 records were complete,
fallback-free, normalized, finite, and source-valid, but exact Bayes-adaptive, MAP, open-loop, and
myopic control had identical values throughout. All three root policies agreed, no model had material
MAP or posterior-sampling regret, and maximum normalized MAP regret was zero. The independent audit
reproduced the four failed scientific gates and verified zero protected-confirmation access.

V71 therefore stops before confirmation; its reliability, horizon, models, controls, and gates must
not be tuned retrospectively, and its five protected models must remain unopened. A successor sensor-
semantics study is justified only after fresh environments are sourced whose declared structure
contains action-dependent sensing, delayed state-dependent reward, and a genuine sensing-versus-
control tradeoff. Those properties must be audited from source metadata before outcomes. Until then,
the correct action is to preserve V71 as a negative boundary result rather than search its protected
set or relax the protocol.

V72 first built a preregistered shared-support mechanism oracle with an explicit calibration action,
target inspection, and condition-dependent repair. The one authorized oracle run passed all 13 gates:
exact Bayes-adaptive planning chose calibration and then inspection, MAP and persistent posterior
sampling chose inspection immediately, and each point control had normalized regret `0.0956`. A
dominant-action negative control had exactly zero regret. Both latent observation models had common
support and zero fallback. This is an implementation/mechanism check only, not scientific evidence.

V72 then performed metadata-only discovery across five pinned repositories and six candidate source
families. Public landing pages for three leads had been previewed during initial planning, so every
inspected repository was conservatively treated as development-exposed. The continuous or simulator-
oriented SBO_AIPPMS models were not exact-exportable, SARSOP's 12,545-state RockSample was resource-
deferred, BetaZero's root license was unresolved, and POMDPModels MiniHallway lacked a distinct sensing
action. The pinned MIT-licensed `RockSample.jl` source supplied the only small configurable exact
candidate. Its frozen 2x2 export passed ten structural tests and the resource audit with 17 states,
7 actions, 3 observations, 38,080 dense-kernel bytes, and a horizon-four Bellman upper bound of 820.

The sole V72 external development outcome was nevertheless cleanly negative. Exact, MAP, persistent
posterior sampling, and best open loop all chose `west`; the common policy moved to the known-good
reference rock, sampled its guaranteed reward, and exited. Its value `13.786875` was independently
reproduced as `0.95*10 + 0.95^3*5`. MAP and posterior-sampling regret and the exact-over-open-loop
advantage were all zero. Seven of nine scientific gates failed, while common support, zero fallback,
and a strict root margin held. V72 is closed before protected-source discovery or confirmation.

The successor constraint is now sharper than V71's. It is not enough to provide explicit sensing and
delayed state-dependent reward: the calibration reference must itself be non-harvestable (for example,
a known-bad reference or observation-only beacon), and a pre-outcome structural dominance audit must
reject any immediately available known-reward route that bypasses sensing. The same audit must certify
that the frozen sensor discriminability can cross the final good/bad control threshold. V72's source,
model, parameters, horizon, controls, and gates must not be tuned or reused for successor outcomes.

V73 implemented that successor gate before running any optimal planner. A fresh Apache-2.0 IMPRL
maintenance source supplied three-state deterioration, inspection, replacement, cost, initial-belief,
and discount parameters. The project-authored adapter added a non-harvestable calibration beacon and
binary label-codebook uncertainty. The beacon used the source do-nothing transition, cost `-8`, had no
positive reward, and could not be controlled or harvested. All ten structural exporter tests passed;
the point models had identical support; calibration mutual information was `0.3681` nats; inspection
healthy/degraded TV was `0.85`; and paired label histories crossed the registered replacement
threshold.

The preregistered dominance gate nevertheless failed. The fixed calibrate-then-inspect adaptive policy
had value `-118.9418`, versus `-119.4345` for the best of 1,024 open-loop sequences, a raw improvement
of `0.4927` but only `0.000403` of the frozen return scale, below the `0.005` minimum. An independent
implementation reproduced the policy value, open-loop optimum, sequence, scale, and failed gate.
V73 therefore stopped before exact Bayes-adaptive, MAP, posterior-sampling, myopic, EIG, protected-
source, human, or model outcomes.

This sharpens the next constraint again. Non-harvestability, mutual information, sensor separation,
and threshold crossing are still insufficient: a fresh source must pass a prospective *economic value
of information* lower bound after all sensing, delay, deterioration, and control costs. That lower
bound must clear the material-effect threshold by margin before an adapter or optimal evaluator is
implemented. V73's repository, component, parameters, adapter, horizon, fixed policy, and gates are
closed and cannot be tuned or reused for successor outcomes.

V74 applied that economic gate before writing an adapter. A fresh MIT-licensed pomdp-py Tiger source
supplied two hidden states, listen/open dynamics, parameterized observations, `+10/-100` opening
rewards, a `-1` target-listen cost, and a `0.95` discount. The source-exposed observation noise was
prospectively fixed at `0.01`; a project-authored non-harvestable reference beacon cost `-0.5`, and a
persistent binary codebook either preserved or reversed the two observation labels. This is a
source-grounded development configuration, not an unchanged external environment.

The closed-form pre-implementation screen passed: the fixed calibrate-listen-open policy had value
`5.609355`, versus `-1.42625` for the best open loop, a raw advantage of `7.035605` and normalized
advantage `0.0224225`, with `0.0074225` margin over the registered threshold. The implementation
reproduced the result after all ten structural tests passed; both point models had common support,
the beacon remained non-harvestable, paired decision accuracy was `0.9802`, and the best of all 64
open-loop sequences was three beacon actions.

The single locked development evaluation then passed every gate. Exact Bayes-adaptive planning
uniquely calibrated first, listened to the target after either beacon label, and used both final door
actions with value `5.609355`. MAP and persistent posterior sampling listened immediately and each
had exact mixture value `-44.20125`, normalized regret `0.158746`, common support, and zero fallback.
Best open loop and myopic control each retained value `-1.42625`. An independent reference recursion
reconstructed all root Q-values, control values, actions, scale, and gates.

V74 therefore closes the V71-V73 design loop in one configured development model: sensor-codebook
uncertainty becomes control-relevant when calibration is non-harvestable, accurate enough, and
economically worthwhile relative to delayed state-dependent loss. The result does not validate the
chosen `0.99` accuracy or `-0.5` beacon cost externally, and its `0.0250` raw root margin depends on
that prospectively fixed cost asymmetry. The next stage is a fresh confirmation-design program that
must seek source-native sensing fidelity, calibration cost, and delayed control loss, pass the same
economic screen before outcomes, and remain untouched until a new source and evaluator are locked.
V74's source, noise, beacon, horizon, adapter, controls, and gates are closed to tuning or rerun.

V75 then tested the mechanism in the valid four-state NOVA paint/inspect model. Source-level
economics were frozen before implementation: a fixed reference-inspect-contingent-control policy
had value `0.166398`, versus `0` open loop, for normalized advantage `0.0224264`. The source supplied
`0.75/0.25` inspection accuracy, paint dynamics, unit ship/reject rewards, reset dynamics, and
discount `0.95`; the project added only a zero-reward identity reference inspection and persistent
canonical/reversed label codebook. All ten source-parity and structural tests passed with common
point-model support.

The sole exact replication attempt passed every gate. Reference-first and target-first sensing tied
at the root; deterministic tie-breaking selected reference calibration. Matching labels led to
paint then ship, while differing labels led to reject. Exact value was `0.166398`; MAP and persistent
posterior sampling each inspected the target immediately and had true-mixture value `-0.00428688`,
normalized regret `0.0230042`, and zero fallback. Myopic and open-loop values were both zero. An
independent recursion reproduced all values, root actions, contingent controls, and gates.

This is outcome-untouched external-domain replication, not discovery-clean confirmation. V68 had
previously inspected and excluded a malformed POBAX variant of the same classic paint problem before
any policy result. V75 used a separately pinned valid MIT source and accessed no prior paint policy
outcome, but the prior structural exposure remains a real limitation. V75 is closed to tuning or
rerun. A broader claim requires a newly preregistered source census over domain families never
previously inspected by the project, with common support and source-level economic screening before
adapter implementation.

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

Freeze the complete V68r2–V70 development-to-confirmation sequence and authorize reporting and
synthesis only for the completed V69 family. Preserve V68 and V68r1 as pre-result policy-domain
failures, V68r2 as the negative command-channel development result, V69 as a positive development
screen rather than replication evidence, and V70 as the sole confirmatory evaluation. Keep
fallback-free, complete-procedure, and Tier B evidence visibly separate.

Do not modify or rerun V69 or V70, revise their gates, rescore development or confirmatory models,
or use the V70 suite for retrospective family design. A broader claim requires a new preregistered
uncertainty family and fresh protected models. Preserve the unsuccessful V65r1/V65r2 outcomes, the
pooled-estimator qualification, the negative V66 MAP/first-repeat separation result, the disclosed
nonblinded interface inspection, and the distinction between bounded value verification, planner
optimality, and safety.

Also freeze V71 as a failed, fallback-free development boundary study. Do not open its protected
confirmation set, change its sensor reliability or horizon, replace its models, or reinterpret its
zero-regret result as confirmation evidence. The next sensor-semantics attempt requires a new source
census and preregistration that establishes an active-sensing tradeoff structurally before any policy
outcome is computed.

Freeze V72-V75 as the completed active-sensing design sequence: V72's harvestable-reference bypass,
V73's economically immaterial sensing result, V74's positive source-grounded development mechanism,
and V75's positive outcome-untouched external-domain replication. Do not modify or rerun V75, change
its source, beacon, horizon, controls, or thresholds, or relabel it as discovery-clean confirmation.
Authorize reporting and synthesis now. Any further empirical stage must begin with a fresh
preregistration and a domain family not previously inspected anywhere in this project.

## Status after V76 (2026-08-17)

V71-V75 now form a frozen falsification sequence with a bounded mechanism claim. Uncertainty and
common support alone failed in V71; active sensing was bypassed by a harvestable known-reward route in
V72; non-harvestability, information, and threshold crossing remained economically immaterial in V73;
and the complete prospectively screened mechanism passed in V74 development and V75 outcome-untouched
external-domain replication. V75 remains explicitly short of discovery-clean confirmation because the
paint family was structurally exposed in V68.

The next authorized action is metadata-only source discovery under the V76 census preregistration.
Every repository and domain family exposed in V62-V75 is excluded. Candidate roles must be assigned by
the frozen SHA-256 ordering across at least two repository-disjoint eligible families. No candidate
implementation may be opened, cloned, executed, or scored before the complete metadata inventory and
role partition are durably locked. If fewer than two families qualify, defer without relaxing the
exposure, structural, resource, economic, or role-assignment gates.

## V76 metadata-census outcome (2026-08-17)

The prospectively authorized metadata-only census completed without opening or running a candidate
implementation and without computing any candidate decision statistic. Fourteen repository candidates
were recorded after 16 queries; zero passed all eight provisional metadata gates. The closest structural
lead, `pymdp` cue chaining, exceeded the registered 64-state envelope in the official dimensions exposed
by a search result and was conservatively marked development-exposed. Other candidates lacked a source-
native reference/comparison pathway, an exact finite model, delayed physical control, reuse terms, or an
unexposed domain family.

No development or protected-confirmation role was assigned because the frozen minimum was two eligible
repository-disjoint families. Freeze the source-feasibility deferral before candidate implementation.
Do not weaken or rerun the census, inspect a rejected candidate, return to Tiger/RockSample/paint/
maintenance or V71 protected models, or add a project-authored beacon/noise layer. The correct current
decision is to report and defer the active-sensing empirical branch until a new preregistration is
justified by a materially different target or genuinely new public source-native benchmarks.
