# V140 Thinking Qualification-Gap Audit Plan

## Purpose

V140 uses only frozen V139 answer IDs, validation categories, counts, and hidden synthetic labels to decide
whether the near-miss can be explained by one mechanism. It does not read generated text or traces, rerun
the model, or rescore V139 under a new rule.

## Tests

The audit will recompute:

- failed gate families and their integer pass thresholds;
- invalid-output reasons, phases, and token-ceiling coincidence;
- apparent and valid-only ambiguous accuracy;
- paired direct/thinking correctness transitions;
- completion-only and semantic-only counterfactual sufficiency.

Completion-only means making the three existing fallback decisions structurally valid without changing
their answer IDs. Semantic-only means correcting the valid ambiguous overcommitments while leaving the three
invalid records invalid. Neither may qualify unless both failure families are independently addressed.

## Boundary

Passing authorizes only a model-free feasibility study for a bounded finalizer plus explicit
evidence-sufficiency gate. It does not authorize another prompt, token increase, language population, model,
API, training, V134 access, induction, authority, action, or execution.
