# V127 Fresh SGD Typed-constraint Selectivity Feasibility Results

## Outcome

V127 failed the preregistered oracle-feasibility gates:

> `oracle_typed_constraint_selectivity_infeasible_do_not_realize_parser`

The one-time run automatically read source-authored slot names for all 576 fresh records without accessing
utterance fields or slot values. Typed evidence was nonempty for 405 records (70.31%). The frozen unique
compatibility rule skipped clarification on 86 records (14.93%): 83 exact-known-class records and three
novel-valid records.

The skip set was not safe or low-value. Only 87.21% of skipped shadow actions were exact, below the locked
95% requirement. Across the nine prior/correlation conditions, skipped average clarification value was
1.0169--1.0861, over three times the 0.30 cost. Selective regret was consequently worse than query-all in
every condition, even though it stayed below the 1.1667 ask-always ceiling.

Overall known exact probability was only 66.58%--68.71% under the selective rule. The typed-schema
candidate was often wrong on queried known cases, and the V119 identity-versus-support channel cannot name
a different exact known intent after rejecting its candidate. Unsupported correctness remained
88.05%--93.13%, and false-known probability remained below 1.78%, but those safety results cannot
compensate for failed known grounding and unsafe skipping.

This is a stronger boundary than a failed parser implementation: the mechanism failed while using perfect
source-authored typed annotations. A real parser or LLM could only add extraction error. Freeze V127 as a
negative oracle-feasibility result. Do not build or evaluate a parser for this rule, add thresholds, inspect
individual failures, or combine it post hoc with V126 similarity. A successor needs information richer
than a set of slot names—such as independently typed relations, constraints, or counterfactual effects—and
must first receive its own model-free identifiability audit. Language/model runs, protected access,
capability induction, richer planning, APIs, training, authority, and execution remain closed.
