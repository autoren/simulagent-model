# V32 protocol: supervision-matched factorized polarity and scope

## Scientific question

V32 asks whether V31's localized truth errors are repaired by explicit semantic decomposition:

> Does separately learning the embedded literal's lexical sign and the outer evidence operation,
> followed by deterministic truth compilation, transfer to unseen sign-by-scope combinations better
> than direct truth prediction from the same frozen representation and supervision?

The scope remains a declared ontology and supported language. V32 does not learn an ontology,
programs, transition dynamics, or arbitrary natural-language semantics.

## Forensic premise

The read-only V31 audit established exact frozen/LoRA functional equality before updates, an exact
saved-parameter allowlist, finite artifacts, and final LoRA fit exact-fact accuracies of 0.123,
0.125, and 0.062. Thus the registered adapter objective collapsed on fit rather than merely
failing held-out transfer. V32 contains no adapter condition and does not sweep that result.

## Formal semantics

Every clause has two generator-derived intermediate variables hidden from agent input:

- lexical sign: `positive` or `negative`;
- outer operation: `assert`, `deny`, `double_deny`, `contrast_select`, or `unresolved`.

The registered compiler exhaustively maps their ten combinations to `true`, `false`, or
`unknown`. `assert`, `double_deny`, and `contrast_select` preserve lexical sign; `deny` inverts it;
and `unresolved` maps either sign to unknown. Compiler correctness must be exactly 1.000 before
model access.

## Fresh evaluation strata

All V32 constructions are disjoint from V30 and V31 and across V32 splits.

The paraphrase stratum uses unseen surface families with sign-operation cells supported during
fit. The composition stratum uses fresh families for negative literals under `deny`,
`double_deny`, and `contrast_select`, combinations absent from fit and calibration. Negative
`assert` is included as a seen-cell control. Both component values and all outer operations occur
in fit; only registered combinations are held out.

Direct/inverse, argument reversal, distractor, lexical-sign, unresolved-invariance, and
scope-operation pairs are audited. Surface family is the statistical unit.

## Supervision-matched comparison

All systems use the same frozen model revision, full-depth clause/entity representations, head
architecture, capacity, fit population, seed order, optimizer, and fixed epoch.

The monolithic system trains predicate, typed arguments, and direct truth. Its otherwise present
sign and operation heads receive zero task loss. The joint-auxiliary system trains those same
fields plus lexical sign and outer operation.

The joint-auxiliary artifact is evaluated twice:

- auxiliary-direct uses its direct truth head;
- factorized-compiled ignores that direct truth prediction and uses the fixed compiler over its
  sign and operation predictions.

Because the latter two outputs share identical parameters and supervision, their difference
isolates deterministic composition rather than extra labels, capacity, initialization, or
training.

## Two independent decisions

Scientific factorization success is evaluated specifically on composition holdouts and requires
strong intermediate accuracy, strong compiled truth, a material exact-fact improvement over
auxiliary-direct, and a positive surface-family bootstrap lower bound. Intermediate-supervision
benefit is separately measured by auxiliary-direct versus monolithic.

Operational language-interface success retains strict end-to-end gates over the full evaluation
population. A scientifically meaningful relative improvement does not authorize V28 unless one
complete system independently passes every absolute multiseed gate.

The simplest absolute-passing system is selected unless a more structured system clears every
material-advantage rule. No seed is selected or discarded.

## Firewall

One fresh corpus, one fit/calibration feature extraction, three monolithic and three joint-
auxiliary training runs, one trained-system lock, and one sealed evaluation are allowed. There is
no checkpoint selection, hyperparameter selection, adapter training, evaluation-guided repair, or
V31 reuse for V32 selection.

V28 remains unchanged. Only a selected absolute-passing language system may be frozen and replayed
once, using the mean logits of all three registered seeds. If no system passes, integration stops.
