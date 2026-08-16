# V38 preregistration: ontology-anchored focus parser

## Why this pivot

V37 repaired outer-operation transfer to 0.978 but left lexical sign at 0.786. Candidate-conditioned
neural scoring did not beat the broader direct readout, and compiled truth improved only 0.058 over
the untouched V36 interface. The registered decision therefore stops linear prompt/readout
iteration.

A post-outcome diagnostic established a narrower constructive fact: if the ontology's positive and
negative lexical forms are available, exact grounding recovers lexical sign perfectly on V32 fit,
exposed V36, and exposed V37. That diagnostic is not a deployable result because those forms were
private generator data, and clauses containing both a literal and its opposite were resolved by
their order.

V38 tests the missing capability directly: select which of several ontology-grounded literal spans
is the discourse focus, then inherit the selected span's sign.

## Interface contract

The declared ontology now includes the canonical positive and negative lexicalizations of every
predicate and relation orientation. A deterministic typed matcher enumerates every grounded literal
span in the evidence. It does not decide which span is asserted, denied, contrasted, or merely a
decoy.

The only learned or parsed decision is focus selection. Candidate focus prompts bind one grounded
span and ask whether the surrounding discourse treats it as the embedded proposition. The selected
span supplies predicate, arguments, orientation, and lexical sign. The V37 operation readout and V32
truth compiler remain fixed.

## Anti-shortcut development population

Fit and fresh validation use disjoint surface families. Both contain positive and negative literals,
direct and inverse relation forms, reversed arguments, and three to five entities. Critical controls
place the true focus both before and after:

- its exact opposite;
- another valid grounded atom; and
- a non-state distractor.

Thus neither first-match, last-match, negation-token, nor single-literal-presence heuristics can pass.
Exact evidence and normalized templates must not overlap between fit and validation.

## Selection and gates

Method and regularization selection use only grouped fit cross-validation. Fresh validation selects
nothing. Compare a shared candidate-span hidden-state ridge, native candidate margins, and a frozen
deterministic discourse parser.

Passing requires 0.95 focus and lexical-sign accuracy; 0.95 separately for focus-first,
focus-second, exact-opposite, and different-atom controls; 0.90 worst-family accuracy; and 0.95
compiled truth when paired with the frozen V37 operation interface.

- If all gates pass, preregister—but do not construct—a fresh semantic confirmation.
- If only the deterministic parser passes, retain the constrained parser as the semantic frontend.
- If all focus interfaces fail, preregister a controlled stronger-frozen-grounder comparison.

No V38 outcome may reopen V32 evaluation, V28, atom binding, adapter training, or the end-to-end
relational suite.
