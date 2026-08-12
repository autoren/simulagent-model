# V9r2 grounding protocol amendment

The first V9 corpus passed every structural, metadata, and position audit, but
the synthetic hexadecimal scene identifier reached 0.619 balanced accuracy in
one held-out-mechanic cell against a locked ceiling of 0.55. Model access was
correctly stopped.

V9r2 makes exactly one data change: remove the first observation line containing
the synthetic scene identifier and shift every evidence offset accordingly.
The already locked, context-specific evidence ordering keeps all 2,160 visible
prompts unique, so the identifier is unnecessary for split integrity.

V9r2 must preserve:

- every natural-language evidence unit;
- every determinant, allowed-value, temporal, span, and symbolic target;
- all context, template, mechanic, operator, surface, and intervention splits;
- zero cross-split or conflicting duplicate prompts; and
- the original frozen-model architecture, 13 evaluation folds, metrics, and
  advancement gates from the V9 preregistration.

Before model access, V9r2 must have zero synthetic context identifiers and pass
the unchanged 0.60 ceilings for metadata-only and position-only evidence-match
balanced accuracy in every fold. Legitimate character n-gram linguistic
baselines remain report-only.

This amendment permits one deterministic corpus transformation and one new
pre-model audit. It permits no model extraction, LoRA, final mechanic, Tone
Drift, V3 test, prior holdout, or V7 result access. Passing authorizes a separate
frozen-grounding lock; it does not itself authorize model access.
