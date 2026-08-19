# V135 Controlled Open-world Minimal-pair Benchmark Results

## Outcome

V135 passed every preregistered structural, observability, leakage, and access gate:

> `freeze_controlled_minimal_pair_asset_authorize_model_free_sequential_value_audit_only`

The frozen synthetic-development asset contains nine safe answer choices, five formal ambiguity families,
forty counterfactual groups, and 200 fixtures. Development and test each contain 100 fixtures from disjoint
slot variants. Every group contains a clear known-side request, a clear alternative-side request, an
ambiguous request, and two clarification-resolved versions of that same ambiguous request.

Registered decisive-cue validation and clarification resolution were both 100%. Every ambiguous fixture
omits both side-specific cues and has `A00` as its answer by construction rather than inheriting an unseen
source intent. Every clarified fixture adds exactly one side-specific answer cue. All nine choices appear in
both splits.

## Boundary repaired

The label no longer depends on a hidden service or schema version. Clear fixtures make the decisive semantic
distinction observable; ambiguous fixtures deliberately withhold it; clarification supplies it. The
presented candidate is always the family's declared known choice, so the right-side cases retain the
dangerous condition in which a plausible known candidate is wrong.

Separate public and hidden artifacts passed the zero-leakage gate. Public fixtures contain only an opaque
identifier, split, fallible presented candidate, and conversation. They contain no truth, family, phase,
possible-choice, or decisive-cue metadata.

## Access and claim boundary

- V134 or other external language read: 0
- Model loads or generations: 0
- API calls or training runs: 0
- Actions, side effects, or executions: 0

This is positive benchmark-construction evidence only. Its language is deterministic and synthetic, so it
does not establish human identifiability, external-language transfer, LLM performance, or unrestricted
open-world understanding. Passing authorizes only a frozen model-free sequential value audit of whether the
registered clarification actions are decision-relevant under explicit costs. It does not authorize a model
run, V134 access, induction, training, authority, action, or execution.
