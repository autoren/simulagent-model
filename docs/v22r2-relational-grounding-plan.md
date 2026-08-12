# V22r2 protocol: open relational-language grounding

## Scientific question and claim boundary

V22r2 asks whether the already validated V22 typed-relational symbolic system can be populated from
supported natural-language descriptions by one frozen language representation and two fixed linear
readouts. It is an open development experiment, not a final evaluation.

The claim, if the registered gates pass, is limited to exact one-step Boolean state grounding under
the declared entity and predicate ontology. Entity discovery, ontology induction, persistent state
mutation, stochastic effects, active interventions, long-horizon planning, weight adaptation, and
general relational world-model learning remain out of scope.

## Immutable foundation

The V22 configuration, generator, semantic kernel, manifest, structural audit, and oracle result are
inputs. V22r2 does not edit them. The V22 oracle result must still authorize relational grounding,
and all source hashes are recorded before model access.

The 24 V22 target programs and identifying support traces are retained. The public language layer is
rebuilt because V22 exposed query axes and encoded query roles in sequential identifiers and list
order. V22r2 never supplies those fields to a model or learned head.

## Split and surface protocol

Within each of the four construction families, V22 development-fit episode ordinals 0 and 1 form
`grounding_fit`; ordinal 2 forms `grounding_calibration`. The 12 V22 development-evaluation episodes
form `grounding_evaluation`. Calibration is report-only and cannot select a feature, threshold,
regularization value, template, or branching policy.

Fit and calibration use surface banks `fit_a` and `fit_b`. Evaluation uses disjoint banks `eval_c`
and `eval_d`. All banks express the same registered semantic cells: unary or directed relation,
true/false/unknown status, affirmative/negated-opposite/contrastive/explicit-unknown operator, and
direct or inverse relation realization. Evaluation tests paraphrase transfer, not unseen semantics.

Each public state item has:

- an independently permuted typed entity inventory and action binding;
- opaque entity, item, evidence, and atom-candidate identifiers;
- a shuffled set of positive atom-candidate statements;
- a shuffled natural-language evidence list; and
- no query-axis, target-program, truth table, semantic signature, answer, or sequential role marker.

The ontology and candidate atom inventory are deliberately provided. V22r2 tests proposition
alignment and truth status, not open-vocabulary predicate or entity discovery.

## Controlled counterfactuals

Every orientation pair shares entities, binding, entity renaming, and all background facts. Only
`linked(actor,target)` and `linked(target,actor)` exchange truth values. Every topology pair likewise
shares its background world; only the registered `linked` edges distinguish a directed chain from a
common-parent fork. Audits operate on oracle graphs, not text similarity.

The remaining entity-count, partial-observation, distractor, and permutation pairs retain their V22
oracle semantics. Public query and support order is independently permuted, so axes cannot be read
from position.

## Frozen representation and fixed hard interface

The only representation is `mlx-community/Qwen3.5-4B-4bit` at revision
`0e7ffd5c629ef7719d4cbc04069232580bfa9d9c`, layer 8. Each scene is encoded once as a joint,
label-free inventory of positive atom candidates followed by its evidence clauses. Candidate and
evidence representations are their respective target-token means accumulated in float32. This is
exactly 384 forward passes; no target mapping or truth label appears in the prompt. The model is
frozen.

Two preregistered heads are fitted once on `grounding_fit`:

1. Atom matching uses the absolute difference and elementwise product of evidence and candidate
   embeddings. Each positive receives three deterministic within-scene negatives. A fixed balanced
   logistic regression with C=1 and a maximum-weight one-to-one assignment yields the graph edges.
2. Truth status maps the evidence embedding to `false`, `true`, or `unknown` with a fixed balanced
   logistic regression with C=1.

No calibration-based threshold or hyperparameter selection is allowed. Uncertainty branching is
deferred until hard-grounding errors have been decomposed.

## Metrics and integration

Grounding reports atom-assignment accuracy, ordered-relation accuracy, truth-status accuracy, exact
scene graph, exact support graph, exact query graph, semantic-cell accuracy, and surface-bank
accuracy. Metamorphic consistency is reported separately for orientation, topology, permutation,
distractor, entity-count, and partial-information groups.

The frozen graph is then integrated with the unchanged V22 enumerator and executor under four
conditions:

1. oracle support / oracle query;
2. frozen support / oracle query;
3. oracle support / frozen query; and
4. frozen support / frozen query.

For each support prefix, the audit records target retention, empty version spaces, and version-space
size. End-to-end metrics are transition-set exact match and complete-episode exact match by family,
axis, entity count, truth status, and surface bank. An empty version space returns the complete
visible outcome vocabulary rather than an invented program.

The executor enumerates at most four predicted unknown atoms in a scene. A support prediction above
that bound empties the version space; a query prediction above it returns the full outcome
vocabulary. This fixed conservative policy prevents a grounding error from creating unbounded
execution and is not selected on calibration or evaluation.

## Firewall and decision rule

Before model loading, a structural audit must verify source hashes, oracle reproduction, opaque IDs,
randomized order, absence of public axis/oracle fields, surface coverage, disjoint surface banks,
zero exact fit/evaluation prompt overlap, controlled counterfactual isolation, graph split checks,
and zero V22r2 model artifacts. The corpus, plan, prompt construction, model, heads, evaluator, and
one-shot limits are then hash-locked.

Passing the development gates authorizes a separately registered relational final design. A hard
grounder that fails mostly through early support errors authorizes development of a probabilistic
support interface. Failures localized to held-out wording call for language coverage changes.
Failures with correct graphs call for integration debugging. Broad graph failures do not authorize
LoRA, grammar expansion, a joint neural challenger, or a final suite automatically.
