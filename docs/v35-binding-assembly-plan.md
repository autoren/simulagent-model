# V35 preregistration: regularized atom binding and modular assembly

## Rationale

V34 repaired the isolated outer-operation interface: the fit-selected operation-focused hidden
readout reached 0.997 calibration accuracy and passed every registered gate. A post-V34 bounded
diagnostic found that a regularized role classifier over existing entity spans reaches 0.981 atom
accuracy and 0.964 relation order when the predicate is oracle-provided. Under the predicted legacy
predicate, however, the assembled interface falls to 0.931 atom accuracy and 0.918 exact fact.

V35 therefore tests the remaining local hypothesis: an atom-focused frozen representation can
repair predicate identity and stabilize typed role decoding, after which independently repaired
components can be assembled with the fixed symbolic truth compiler.

## Firewall

V35 uses only V32 `factor_fit` and `factor_calibration`, plus the audited and frozen V34 operation
predictions on exactly those records. It must not read either V32 evaluation split or any V32
evaluation feature/prediction, construct a new suite, train the backbone or an adapter, or run V28.
It is a development result, not a new generalization claim.

## Frozen extraction

One atom-focused forward pass is made for each of 1,820 development clauses. The prompt contains
the declared typed ontology, entity inventory, evidence, canonical relation order, and fixed
predicate choices. It contains no target predicate, arguments, sign, operation, candidate fact, or
truth status. The extraction records:

- the final hidden state;
- float32 A--E predicate label logits; and
- contextual means for each entity mention inside the evidence text only.

## Fit-only selection

All learned readouts use a balanced L2 ridge classifier and leave-one-fit-surface-name-out cross-
validation. Alpha selection maximizes mean fold primary accuracy, then worst-fold accuracy, then
the larger alpha. Calibration chooses nothing.

Predicate comparison: legacy hidden state, atom-focused hidden state, and atom-prompt native
label logits. Only the two new prompt methods are eligible for the modular system.

Binding comparison: legacy entity spans and atom-focused evidence-only entity spans. Each entity
is classified as neither, argument 1, or argument 2 after a fixed, label-independent 256-dimensional
Gaussian projection. Binding selection uses relation-order accuracy with the oracle predicate on
fit folds. Final decoding uses the selected predicted predicate and deterministic ontology type
masks.

Lexical sign uses the regularized legacy hidden readout. Outer operation is the fixed V34 selected
component. Their predictions are compiled by the unchanged V32 truth table.

The registered budget is 150 ridge fits: 75 predicate, 50 binding, and 25 lexical-sign fits.

## Gates and decision

The modular assembly must reach calibration predicate 0.98, exact atom 0.95, relation order 0.95,
lexical sign 0.95, outer operation 0.95, compiled truth 0.95, and exact fact 0.90. It must also
improve exact fact by at least 0.03 over the legacy-predicate/legacy-binding assembly.

Passing authorizes preregistration—but not construction—of a separate development confirmation
suite with new surface families. Failure remains local to the atom interface and does not authorize
V32 evaluation reuse, V28, LoRA, or a final-suite claim.
