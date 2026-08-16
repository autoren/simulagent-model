# V39 preregistration: complete declared-language compiler

## Interpretation of V38

V38 resolved the lexical-sign problem within the declared ontology. Its deterministic focus parser
selected the correct grounded literal on every fresh validation clause, including focus-first,
focus-second, exact-opposite, different-atom, direct, and inverse controls. Lexical-sign accuracy was
therefore 1.000.

The untouched neural operation readout fell to 0.633 on the new two-literal grammar, reducing
compiled truth to 0.767. This is not evidence against the focus parser. It shows that leaving outer
operation as an unconstrained neural class boundary recreates the same surface-transfer problem at
the next semantic layer.

## V39 claim

V39 will test a narrower, explicit contract:

> Given a declared predicate lexicon and declared operator grammar, the frontend exactly compiles
> supported evidence into a grounded signed fact and outer operation, and safely abstains on
> unsupported or ambiguous input.

This is a controlled-language compiler claim. It is not open-ended natural-language understanding,
ontology induction, or paraphrase generalization.

## Architecture

The predicate ontology supplies typed positive and negative lexical forms. A separate operator
ontology supplies cue primitives and structural productions for assertion, denial, double denial,
contrast selection, and unresolved evidence. The compiler:

1. enumerates typed grounded literal spans;
2. parses the declared discourse production;
3. identifies the focused span and outer operation jointly;
4. emits predicate, ordered arguments, lexical sign, and operation; and
5. applies the unchanged V32 truth table.

It must never resolve ambiguity with first-match or last-match heuristics. Unsupported, malformed,
and genuinely ambiguous clauses receive an explicit abstention or a set of supported parses.

## Evaluation

The supported evaluation holds out combinations of declared primitives—not undeclared words. It
crosses literal position, decoy type, sign, operation, direct/inverse orientation, punctuation,
entity count, and argument reversal. Separate safety challenges cover malformed grammar, equally
marked focus literals, unknown predicates, and unknown operation cues. Novel natural-language
paraphrases are reported only as a non-gating scope diagnostic.

All supported-language and safety gates are exact: 1.000 coverage, parse, compiled truth,
composition-cell accuracy, malformed/unknown abstention, and ambiguity safety.

Passing authorizes preregistration—but not construction—of a fresh supported-language semantic
confirmation. It does not authorize V32 evaluation, V28, adapter training, backbone changes, or an
end-to-end relational suite.
