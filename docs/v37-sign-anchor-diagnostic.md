# V37 post-outcome lexical-sign anchoring diagnostic

Status: descriptive only. This did not amend V37, fit a model, access V32 calibration/evaluation, or run V28.

## Finding

Exact matching against the generator's positive and negative ontology lexicalizations recovers a literal in 100.0% of V37 clauses and obtains 1.000 sign accuracy. It also reaches 1.000 on exposed V36 and 1.000 on V32 fit.

This shows that sign is mechanically recoverable once the grounded literal's positive/negative lexical forms are available. It does not show that the current agent can do so: those forms are stored in the generator config, not exposed in `agent_input`, and the diagnostic resolves two-literal cases by choosing the earliest match.

## Consequence

The justified pivot is an ontology-anchored constrained parser, not another linear hidden-state prompt. A proper next test must expose lexical definitions through the declared ontology, parse grounded literal spans, and include counterexamples where the mentioned opposite precedes the focused literal so that first-match heuristics cannot pass.

A stronger frozen grounder remains a later comparator, but V37 does not justify changing the backbone before testing whether the declared symbolic interface can supply the missing lexical anchor.
