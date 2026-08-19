# V135 Controlled Open-world Minimal-pair Benchmark Plan

## Question

Can a completely deterministic synthetic-development benchmark separate exact known capability use,
valid-but-undeclared capability use, unsupported requests, and genuinely insufficient language without
letting a hidden source identifier determine the label?

## Prospective design

V135 is a separate branch from the frozen V134 SGD asset. It must read zero V134 language and run no model.
Five formal ambiguity families pair a declared known capability with either a valid undeclared capability or
an unsupported capability. Each of eight variants produces five fixtures:

1. a sufficient clear-left request;
2. a sufficient clear-right request;
3. an ambiguous request whose correct answer is `A00`;
4. the same ambiguous request followed by a left-resolving clarification answer;
5. the same ambiguous request followed by a right-resolving clarification answer.

The development and test splits use disjoint slot variants. Every sufficient surface contains its registered
decisive cue and excludes the opposing cue. Every ambiguous surface excludes both cues. A clarification
answer adds exactly one registered side-specific cue. The ambiguous fixture has no hidden intent label: its
gold answer is insufficient evidence by construction. Separate hidden and public artifacts prevent truth,
phase, possible-choice, and cue metadata from appearing in future model prompts.

The presented candidate is always the family's declared known choice. Clear-right and clarified-right cases
therefore test the dangerous boundary where a plausible known candidate is wrong.

## Gates and boundary

The frozen asset must contain nine safe choices, five families, forty minimal-pair groups, and two balanced
100-fixture splits. Cue validation, truth derivation, and clarification resolution must be exact. All nine
choices must appear in each split, and public gold leakage must be zero.

Passing authorizes only a model-free sequential value audit of the frozen clarification structure. It does
not authorize a local or API model run, V134 language access, human-equivalence claims, induction, training,
authority, actions, or execution. Synthetic language remains development evidence only.
