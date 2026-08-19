# V154r1 outcome-verifier repair plan

## Purpose

V154 completed and its scientific result was written before outcome freezing. The locked V154 verifier then passed every access, reconstruction, safety, and decision check except the two exact-summary comparisons. The displayed recomputed and persisted summaries were numerically identical.

The failure is a representation-only mismatch. `evaluate_condition` returns `rank_counts` with integer dictionary keys. JSON persistence necessarily converts object keys to strings, so reading the V154 aggregate result back produces string keys. Direct Python equality therefore fails even though the JSON data are identical.

V154r1 repairs only that comparison. It preserves the original verifier, its failed audit, all V154 results, and the negative development decision.

## Locked repair

Before writing a repaired outcome, V154r1 must prove for both direct and bounded-low conditions that:

1. the original V154 analysis lock and all dependencies are unchanged;
2. the original failed audit has exactly two false checks, both summary comparisons;
3. recomputation is unequal before canonicalization;
4. the sole unequal field is `metrics.rank_counts`;
5. recomputed keys are integers and persisted keys are their exact string equivalents;
6. removing `rank_counts` makes the summaries exactly equal;
7. a recursive JSON round trip makes each complete recomputed summary exactly equal to its persisted summary;
8. the selected condition remains null and the frozen negative decision remains unchanged;
9. no model/tokenizer load, generation, raw-language inspection, evaluation access, API, training, service, side effect, or execution occurs.

The repaired verifier must rerun the original substantive checks and replace only the two comparisons with canonical JSON comparisons. It may then freeze a technical V154r1 outcome lock. It may not overwrite the V154 failed audit or create a nominal V154 outcome lock.

## Claim boundary

This repair cannot improve V154's scores or qualification status. Direct still misses the top-1 and MRR gates; bounded low reasoning still misses structural, ranking, mean-rank, and cost gates. The decision remains:

`local_question_order_conditions_fail_development_gates_close_without_evaluation_or_tuning`

V154r1 authorizes no evaluation, retry, rerun, reprompt, reasoning change, threshold fitting, calibration, model change, API, training, authority, action, or execution.
