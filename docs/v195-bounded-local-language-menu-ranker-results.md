# V195 bounded local language-to-menu ranker results

## Verdict

V195 is a positive, incremental development result. The frozen bounded-low-reasoning `Qwen3.8-27B-4bit` shadow
ranker passed every qualification and access gate. It improved the trusted controller beyond the strong V194
character-retrieval baseline; the gain is therefore attributable to additional model ranking quality under this finite
menu, not merely to beating the original fixed hierarchy.

This is not yet a confirmation result and does not establish unrestricted open-world recognition.

## Main result

Across all 84 observed development utterances, the model achieved:

- primary weighted top-1 recall: `0.8625000000`;
- primary weighted top-3 recall: `0.9291666667`;
- balanced top-1 recall: `0.8809523810`;
- balanced top-3 recall: `0.9404761905`;
- primary top-1 controller cost: `0.1479166667`;
- primary top-3 controller cost: `0.2141666667`; and
- balanced top-3 controller cost: `0.2119047619`.

The V194 `CHAR_LAST` comparator had primary top-3 cost `0.2583333333` and balanced top-3 cost
`0.2476190476`. V195 therefore improved the primary cost by `0.0441666667` and the balanced cost by
`0.0357142857`. The preregistered minimum primary improvement was `0.01`.

Compared with the original V190 hierarchy cost of `0.38`, the complete model-plus-trusted-controller policy reduced
expected primary clarification cost by `0.1658333333`. The relevant incremental comparison remains V194, because a
simple model-free method had already captured much of that reduction.

## Class-conditioned behavior

Balanced top-3 recall was:

- known: `0.9166666667`;
- provisional: `0.9444444444`; and
- unsupported: `1.0`.

All three truth kinds exceeded the frozen `0.75` gate. The result is not explained by matching only familiar catalog
items. However, these labels are hidden benchmark strata; the model did not receive truth kind, target contract, or
the hidden option map.

The model returned the valid explicit `INSUFFICIENT` response on `5.9524%` of observed records. Those cases safely
used the unchanged target-specific hierarchy. All 14 missing-observation controls were also insufficient without any
model generation.

## Overthinking and truncation finding

Every reasoning phase consumed the full frozen 48-token allowance, and none naturally emitted a closing `</think>`
tag within that budget. This confirms that low reasoning effort alone did not make Qwen3.8 self-terminate its hidden
reasoning quickly on this task.

The mechanical two-phase intervention worked as intended: after 48 tokens, the harness closed the thinking phase and
reserved a separate 64-token final continuation. All 84 final answers ended within the final budget, structural
validity was `1.0`, final-phase limit-hit rate was `0.0`, and mean final length was `23.3452` tokens.

The correct interpretation is therefore not that truncation was harmless. It is that a separately reserved final
channel converted deterministic reasoning-budget exhaustion into complete, parseable answers. Increasing one shared
token limit would not provide the same guarantee.

## Safety and access

The model and tokenizer each loaded once. The run made exactly 84 bounded reasoning generations and 84 final
generations, with no retries. Peak active MLX memory was `16,459,443,306` bytes and elapsed time was
`1,098.2295` seconds.

Raw reasoning and final text were hashed but neither persisted nor manually inspected. API calls, protected-language
reads, training, ontology registration, trusted-state mutation, service calls, external side effects, action, and
execution were all zero. The model only ranked a finite visible menu; trusted answers retained the full 14-contract
fallback and determined the exact terminal state. Target retention and final exactness remained `1.0`, with zero
model-caused terminal decisions.

## Decision and next step

Freeze:

`freeze_V195_positive_incremental_local_model_development_result_and_authorize_separate_confirmation_design_only`

The justified successor is a separately preregistered, independently authored confirmation population using the
unchanged Qwen snapshot, prompt, low-effort 48-plus-64 mechanical budget, exact parser, top-3 controller, costs, and
gates. No prompt repair, API comparator, additional local model, protected-language opening, ontology promotion,
authority, action, or execution is authorized by V195 itself.

