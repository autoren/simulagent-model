# V198 protected language-to-menu ranker confirmation results

## Verdict

V198 is a positive independent confirmation. The complete V195 local-model policy passed every unchanged scientific
and access gate on V197's 113 dialogue-isolated protected utterances plus 12 missing controls. It retained an
incremental advantage over the unchanged `CHAR_LAST` comparator scored on the same confirmation records.

This supports a finite, non-authoritative clarification-menu reduction claim. It does not establish unrestricted
open-world recognition or grant the model ontology or action authority.

## Main result

The unchanged bounded `Qwen3.8-27B-4bit` policy achieved:

- primary weighted top-1 recall: `0.8696428571`;
- primary weighted top-3 recall: `0.9562500000`;
- balanced top-1 recall: `0.8761061947`;
- balanced top-3 recall: `0.9557522124`;
- primary top-1 controller cost: `0.1477678571`;
- primary top-3 controller cost: `0.2087500000`; and
- balanced top-3 controller cost: `0.2088495575`.

On the same records, unchanged `CHAR_LAST` achieved primary top-3 recall `0.9194534632` and controller cost
`0.2322186147`. The model improved primary cost by `0.0234686147` and balanced cost by `0.0194690265`, exceeding the
prospective `0.01` incremental gate.

The model also cleared the absolute V195-derived ceilings of `0.2483333333` primary and `0.2476190476` balanced cost.
Relative to the fixed V190 hierarchy at `0.38`, its primary reduction was `0.17125`.

## Development-to-confirmation transfer

V195 development primary top-3 recall was `0.9291666667` and cost was `0.2141666667`. Confirmation recall increased
by `0.0270833333` and cost decreased by `0.0054166667`. Balanced recall increased by `0.0152760219` and balanced cost
decreased by `0.0030552044`.

This is not evidence that confirmation is intrinsically easier: the record distribution and per-contract counts
differ. It does show no degradation under the preregistered metrics and preserves the model's same-population
increment over character retrieval.

## Class-conditioned behavior

Balanced top-3 recall was:

- known: `0.9428571429` across 70 records;
- provisional: `1.0` across 34 records; and
- unsupported: `0.8888888889` across 9 records.

Every stratum exceeded the unchanged `0.75` gate. The unsupported estimate has the smallest sample and should not be
read as a broad open-set guarantee.

## Fail-closed and overthinking behavior

Structural validity was `0.9911504425`: one of 113 final outputs was malformed or truncated JSON. The exact parser
mapped it to `INSUFFICIENT`, so the trusted hierarchy handled it. The final-phase token-limit-hit rate was zero,
target retention and trusted exactness were both `1.0`, and there were no model-caused terminal decisions.

As in V195, every reasoning phase used all 48 tokens and none naturally closed. Mechanical closing plus the distinct
64-token final phase again mattered: mean final length was `23.6549` tokens, and every final generation stayed within
its budget. Low reasoning effort alone did not prevent reasoning-budget exhaustion.

## Access and decision

The model and tokenizer loaded once. The run made exactly 113 reasoning and 113 final generations, with no generation
for 12 missing controls and no retries. Runtime was `1,481.8825` seconds; peak active MLX memory was
`16,463,752,298` bytes. Raw model text was hashed but not persisted or manually inspected.

Unselected protected-language reads or scores, API calls, training, ontology registration, trusted mutation, service
calls, side effects, action, and execution were zero.

Freeze:

`freeze_V198_positive_confirmation_of_finite_non_authoritative_local_model_menu_reduction`

The next justified branch is model-free robustness and distribution-shift design for the now-confirmed finite-menu
interface. V198 does not authorize an API, another model, ontology promotion, authority, action, or execution.

