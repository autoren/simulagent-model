# V158 model-free hard-tie routing policy results

## Verdict

V158 is a clean negative development qualification with complete architectural safety. The frozen margin-gated router failed five of 96 initial-action targets and missed every cost/selectivity gate affected by those errors. It closes without threshold changes, term edits, rerun, evaluation access, or model fallback.

The router was exact on all 24 lexically decisive controls and all 24 insufficient requests. It sent 23 of 24 uncatalogued paraphrases and 20 of 24 relational ties to generic route `Q70`. Overall initial-action accuracy was 94.792% rather than the required 100%. Its incorrect-specific rate across fallback strata was 6.944% rather than zero.

All five errors were high-margin false-specific decisions:

- one uncatalogued compost/food-rescue paraphrase was sent to school-accommodation question `Q73`, with top score 7.0 and margin 6.25;
- all four development variants of the irrigation/air-monitor relational tie were sent to heritage-record question `Q75`, with top score 6.75 and margin 6.25.

The second pattern is especially informative. The relational request intentionally mentioned both an environmental device and a heritage-image workflow. Lexical scoring strongly recognized the decoy side and therefore appeared confident rather than tied. A score-margin gate cannot detect semantic cross-family ambiguity when only one relation is lexically well covered.

Across 120 episodes, the router's mean cost was 0.4225, versus the preregistered maximum 0.4. Improvement over no query was 0.3775 rather than 0.4, and improvement over always-generic routing was 0.0175 rather than 0.04. The information oracle cost 0.36; always-generic cost 0.44; source and seeded-random specific ordering cost 1.24 and 1.265.

Freeze the decision:

`margin_gated_generic_router_fails_development_gates_close_without_tuning_or_model`

## Safety result

The five initial errors did not become semantic errors. A wrong specific answer yielded no trusted witness, remained `A00`, and automatically fell back through `Q70`. All 120 margin-router episodes and every interactive comparator ended in the exact final state. All 787 irrelevant intermediate events across the router, source, and random policies failed closed. Authoritative hypothesis retention was 100%, candidate-state fields were absent, and execution was zero.

This establishes the intended failure mode:

> A non-authoritative routing error can increase evidence cost but cannot accept the wrong capability.

## Claim boundary and next step

V158 is project-authored synthetic model-free development evidence. It does not justify V157 evaluation access or a local-model tie-breaker. The parent policy explicitly required every routing gate to pass before such a protocol could be designed.

Do not alter the 6.0 score threshold, 4.0 margin, retrieval terms, weights, costs, strata, or gates. Do not rerun V158 or test a model/API on this population.

The next permissible work is model-free failure decomposition only. It should quantify whether an explicit relational-conflict detector can be specified independently of the observed five records—for example, by requiring evidence for both asserted alternatives rather than relying on score uncertainty. Any successor must use fresh language and a prospective rule. Evaluation, model/hybrid access, calibration fitting, training, induction, authority, action, and execution remain closed.
