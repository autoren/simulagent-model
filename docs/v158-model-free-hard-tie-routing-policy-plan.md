# V158 model-free hard-tie routing policy plan

## Purpose

V158 tests whether a frozen deterministic evidence gate can choose between a registered specific question and safe generic route `Q70` across V157's four evidence strata.

The policy uses V156's fixed lexical scoring. It asks the top specific question only when the top score is at least 6, the top-two margin is at least 4, and the top score is unique. Otherwise it asks `Q70`. If a selected specific question is wrong, its answer is treated as irrelevant `NO_SELECTION`, the state remains `A00`, and the policy falls back to `Q70`.

Specific questions cost 0.3 and generic routing costs 0.2. A family route is followed by its specific trusted question; `UNCLEAR` ends safely at `A00`. Only a specific trusted answer can produce a semantic witness.

## Prospective gates

The policy must select a specific question on every lexical control and `Q70` on every uncatalogued paraphrase, relational tie, and insufficient request. Its initial-action accuracy must be 100%, with zero incorrect specific selections on fallback strata.

Across 120 development episodes it must achieve exact final states, complete intermediate fail-closure, complete hypothesis retention, mean cost at most 0.4, improvement over no query at least 0.4, and improvement over always-generic routing at least 0.04.

The algorithm, thresholds, comparators, costs, and gates are frozen before development truth is scored. Evaluation, models, hybrid policies, fitting, calibration, APIs, training, capability induction, authority, action, and execution are prohibited.

Passing authorizes only design of a separate local non-authoritative tie-breaker protocol. It does not authorize the model run itself.
