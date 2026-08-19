# V200 transformed `CHAR_LAST` controls results

## Verdict

V200 passes every preregistered integrity, invariance, signal, and access gate. The two V199 transformations provide
an informative deterministic baseline for the local-model robustness test.

## Order-only control

`ORDER_ONLY` reproduced canonical `CHAR_LAST` exactly after mapping option IDs back to contracts:

- top-1 contract agreement: `1.0`;
- mean top-3 contract-set Jaccard: `1.0`;
- target-inclusion disagreement: `0.0`;
- primary top-3 recall: `0.8541666667`; and
- primary top-3 cost: `0.2583333333`.

This confirms that the implementation is insensitive to list order when semantic rows and IDs are unchanged.

## Opaque-ID control

`ORDER_AND_OPAQUE_ID` retained strong nonrandom signal but exposed the expected deterministic tie-break sensitivity:

- primary top-3 recall: `0.8388888889`;
- macro top-3 recall: `0.8690476190`;
- primary top-3 cost: `0.2644444444`;
- top-1 contract agreement with canonical: `0.9880952381`;
- mean top-3 contract-set Jaccard: `0.9226190476`; and
- target-inclusion disagreement: `0.0357142857`.

The semantic scores are unchanged. Only exact score ties can be reordered by the new option IDs. This makes the
opaque condition a slightly stronger paired comparator rather than a broken baseline.

## Access and decision

The run read exactly 98 development fixtures and computed 2,352 deterministic option scores. Missing controls were
not scored and remained insufficient. Manual inspection, protected access, model/API/training use, ontology mutation,
services, side effects, action, and execution were zero. Target retention and trusted exact completion remained
`1.0`.

Freeze:

`freeze_V200_transformed_CHAR_LAST_controls_and_authorize_separate_unchanged_local_model_development_robustness_preregistration_only`

The next permitted step is a separately locked local-model development robustness run using the complete confirmed
V195/V198 two-phase policy and the V199 prospective gates. No prompt repair, model selection, protected access, or API
fallback is authorized.

