# V194 deterministic language-to-menu ranker results

## Verdict

V194 passed every population, ranking, missing-control, minimum-signal, trusted-controller, baseline, oracle, and
access gate. More importantly, two simple character rankers already have material economic value. The finite-menu
language task is therefore not an LLM-only problem.

## Main result

The fixed champion was `CHAR_LAST`, character 3/4/5-gram cosine on the final user utterance:

- primary weighted top-1 recall: `0.6027777778`;
- primary weighted top-3 recall: `0.8541666667`;
- balanced top-1 recall: `0.6547619048`;
- balanced top-3 recall: `0.8809523810`;
- top-1 controller cost: `0.2588888889`; and
- top-3 controller cost: `0.2583333333`.

Both policies materially beat the frozen `0.38` hierarchy and the `0.36` gate. Top-3 was slightly cheaper despite its
higher initial question cost because it avoided more `OTHER -> generic` fallbacks.

`CHAR_ALL` also had material value, at primary costs `0.3072222222` for top-1 and `0.2927777778` for top-3. The token
and reciprocal-rank-fusion controls did not: their primary top-3 costs were `0.4422222222` and `0.4116666667`.

## Class-conditioned behavior

For `CHAR_LAST`, balanced top-3 recall was:

- known: `0.7777777778`;
- provisional: `0.9722222222`; and
- unsupported: `0.9166666667`.

Thus the gain is not produced solely by selecting familiar declared contracts. On this finite menu, compact character
overlap is especially effective for the provisional and unsupported labels. That does not make those contracts
authoritative or establish unrestricted novelty recognition.

## Safety and controls

All four rankers emitted valid three-option rankings for all 84 observed records. All 14 missing controls emitted
`INSUFFICIENT`. The trusted answer, never the retrieval score, determined the exact terminal state; target retention
and final exactness were both `1.0`.

The fixed controls reproduced:

- V190 hierarchy: `0.3800000000`;
- always generic: `0.40`;
- oracle top-1: `0.10`; and
- oracle top-3: `0.20`.

Manual inspection, protected-language access, model/API use, training, ontology registration, trusted mutation,
service calls, side effects, action, and execution were zero.

## Decision and next comparison

Freeze:

`freeze_V194_controls_and_authorize_one_separately_preregistered_bounded_local_model_shadow_comparator`

The local-model study must now answer a stricter question: does one frozen bounded local model improve on
`CHAR_LAST`, especially its `0.2583333333` top-3 cost and 77.78% known top-3 recall, under the identical visible menu,
output parser, trusted controller, and 84 fresh records? A model merely beating the original 0.38 hierarchy is no
longer enough to demonstrate incremental value.

V194 does not authorize immediate model execution, API fallback, protected access, ontology registration or pruning,
trusted mutation, action, or execution.
