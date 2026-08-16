# V31 protocol: matched-head signed-fact adaptation

## Scientific question

V31 asks whether V30 failed because its zero-shot decoding interfaces were inadequate or because
the frozen backbone does not expose a sufficiently systematic relational representation:

> Can a supervised structured readout recover surface-invariant signed facts from a frozen
> full-depth representation, or does the same readout require limited LoRA adaptation of the
> backbone?

The study is language-only. It receives no transition outcome, target program, program identity,
V28 score, support consistency signal, or executor feedback during representation extraction,
training, calibration, or sealed language evaluation. Passing does not establish ontology
induction, open-vocabulary grounding, general world-model learning, or sequential reasoning.

## Fresh family-disjoint corpus

V30 is exposed and is not reused for V31 training, calibration, or evaluation. V31 constructs 55
new syntactic surface families: five fit, one report-only calibration, and five sealed evaluation
families for each of the five supported semantic operators. Every construction template and its
normalized construction hash is disjoint from V30 and across V31 splits.

Each family contains four typed base scenes and the same controlled direct, inverse, distractor,
and reversed-argument variants. Facts span the declared unary and directed-relation predicates,
`true`, `false`, and `unknown`, three-to-five entities, and short-to-long clauses. Registered pairs
test distractor invariance, direct/inverse equivalence, argument reversal,
affirmative/negated-opposite equivalence, affirmative/double-negation equivalence, and
false/unknown separation. The surface family—not an individual clause—is the statistical unit.

Before model access, a structural audit must reproduce every target and surface, verify type
validity and every pair, prove semantic support, confirm zero exact or construction overlap, and
hash the corpus, methods, training code, gates, and this plan.

## Matched learned systems

Both learned systems use the same Qwen3.5-4B revision, prompt, full-depth clause representation,
entity-span representations, structured pointer head, multitask loss, fit population, record
order, optimizer, learning rate, epoch count, and three seeds.

The shared head predicts:

1. one of the five declared predicates;
2. argument 1 by pointing to a type-compatible entity;
3. argument 2 by pointing to a type-compatible entity or `N/A` for unary predicates; and
4. `true`, `false`, or `unknown`.

Its fixed nonlinear interaction combines the clause vector, an entity vector, absolute
difference, and elementwise product. Type masks are deterministic consequences of the declared
ontology. Inverse wording must still produce canonical `predicate(source,target)` order.

The **frozen-readout** system freezes every backbone parameter and trains only this head. The
**LoRA-readout** system trains the identical head while adapting only rank-8 LoRA modules in the
registered projections of the final eight layers. Full-model tuning, alternate ranks, layer
sweeps, prompt selection, checkpoint selection, and post-evaluation thresholding are forbidden.

The locked V30 candidate-independent field decoder is evaluated as a zero-shot reference but does
not select either learned system. Calibration is report-only. No seed is selected or discarded.

## Training and trained-system lock

Each seed processes one fixed epoch of the complete fit population with batch size one and four-way
gradient accumulation under Adam at the registered learning rate. Inverse-frequency fit-class
weights are fixed before training. All four field losses have equal weight. The complete head and,
where applicable, LoRA parameters are saved after the fixed epoch.

Before opening evaluation, a second lock must hash all six trained systems, their immutable
training ledgers, report-only calibration metrics, parameter counts, source protocol, and proof that neither evaluation
features nor predictions were read. No training or method edit is permitted after that lock.

## Sealed evaluation and gates

All three seeds of both systems are evaluated once on all 25 unseen evaluation surface families.
Report predicate, arguments, relation order, truth, exact signed fact, exact scene, truth class,
operator, predicate, entity count, sentence length, scene variant, every surface family, and every
controlled pair. Report mean and minimum across seeds. A system passes only when all three seeds
meet every per-seed gate and its aggregate stability gates pass.

For attribution, compute the paired LoRA-minus-frozen difference per evaluation surface family
and a fixed family bootstrap interval. LoRA has a material advantage only if its registered exact
fact and exact-scene deltas pass and the family-bootstrap lower bound is positive.

Decision order:

- If frozen passes and LoRA does not materially improve it, select the frozen readout.
- If frozen fails and LoRA passes, select LoRA and infer that limited representation adaptation is
  necessary for this interface under the registered control.
- If both pass, prefer frozen unless LoRA clears every material-advantage rule.
- If both fail, stop. Do not sweep rank, add graph reranking, relax gates, or open V28.

Only the selected passing language system may be frozen and replayed once through unchanged V28.
The replay averages field logits across all three registered seeds, so it does not select a seed.
The ontology, graph semantics, DSL, executor, support outcomes, program inventory, marginal-program
MAP rule, integration conditions, and integration gates remain immutable.
