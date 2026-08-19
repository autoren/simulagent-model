# V185 deterministic SGD candidate-set controls plan

## Question

Can two simple, reproducible views of externally authored SGD language safely identify a subset worth asking a cheap candidate-specific clarification, while routing every other request to a generic trusted clarification?

This is not a direct classifier. Retrieval outputs are shadow candidate sets. A trusted typed answer remains the only source of terminal semantic state.

## Frozen development split

The 132 V184 development records are split by fixed hash within hidden truth kind into 66 calibration and 66 evaluation records. Each half contains 36 known, 18 provisional, 6 unsupported, and 6 missing controls. Split membership cannot use language, predictions, scores, or policy outcomes. Protected language remains sealed.

## Deterministic views

The first view computes character 3–5-gram cosine similarity between user-only conversation language and each of the six declared schema documents. The second computes token-count cosine overlap. A third exact intent-alias lookup is retained as a comparator.

Each retrieval view returns a shadow set, not an answer. Below-threshold or near-tied evidence produces multiple candidates and therefore `INSUFFICIENT`. The complete authoritative hypothesis universe is never pruned.

## Prospective calibration

A joint finite threshold grid is selected on calibration only. Candidate-specific routing is allowed only when the character and token views return the same singleton. Otherwise the controller asks the generic clarification.

A grid point qualifies only with at least 95% candidate-specific precision, at most 5% false-specific routing on non-known requests, and at least 20% known-specific coverage. Among qualifying points, selection minimizes trusted clarification cost, then maximizes precision and known coverage, then uses the lexicographically smallest parameter tuple. If none qualify, the locked fallback is always generic.

Evaluation cannot change thresholds.

## Decision costs and safety

A correct candidate-specific clarification costs 0.25. A generic clarification costs 0.40. A wrong or non-known specific question yields no semantic witness and then falls back to generic, costing 0.65. Missing input costs zero and remains insufficient.

All observed paths terminate exactly after the trusted answer. Retrieval never accepts a candidate, declares novelty, rejects a capability, registers an ontology item, mutates trusted state, acts, or executes.

The prediction-defined residual is precisely the evaluation records without same-singleton consensus. Its membership cannot use truth. Hidden labels may only audit whether the residual remains scientifically diverse.

## Gates and successor

The router must preserve at least 95% specific precision, at most 5% non-known false-specific routing, at least 20% known-specific coverage, final exactness of 100%, and mean clarification cost at least 0.02 below always-generic routing. The residual must contain 24–100 records and cover known, provisional, and unsupported truths.

Passing authorizes only design of one local shadow proposer on the frozen residual. It does not authorize a model run, protected access, API, training, registration, authority, action, or execution. Failure freezes the deterministic boundary and closes this route without threshold edits.
