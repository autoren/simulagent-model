# V131 Complete-clarification Realization Population Plan

## Question

V130 found that a complete eleven-choice answer is abstractly sufficient if its effective one-answer
reliability reaches 97.25% in the hardest frozen prior/error condition. V131 asks only whether untouched SGD
records can instantiate the same 66 truth-by-presented-candidate cells for a prospective realization audit.

## Text-free design

Use the unconsumed remainder of the pinned SGD test partition. Exclude every identifier selected by V125,
V127, or V128. Freeze six exact declared service-intent choices, three valid-undeclared domain composites,
one unsupported composite, and one missing-observation control. Cross every truth choice with every declared
known candidate and select four fixtures per cell, for 264 fixtures total.

The 240 source-backed fixtures are selected from identifiers and structural annotations only. Twenty-four
missing-observation controls instantiate `A00`. No utterance, dialogue context, slot value, schema
description, model, or response is read during selection.

## Gates and boundary

All 66 cells must contain exactly four fixtures. Every truth must occur 24 times and every presented known
candidate 44 times. The population must cover all six known pairs, all three novel domains, and all four
remaining unsupported domains, with no excluded identifier overlap.

Passing authorizes only a separately locked local-model realization protocol. It does not authorize language
extraction or generation before that lock and does not authorize induction, APIs, training, authority, or
execution.
