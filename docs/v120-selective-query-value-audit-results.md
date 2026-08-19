# V120 Selective Query Value Audit Results

## Outcome

V120 passed as an aggregate cost decomposition:

> `selective_query_value_boundary_derived_requires_independent_prequery_trigger`

It read only V119's frozen aggregate metrics. It did not inspect records, language, model outputs, protected
data, or prompts, and it did not define or evaluate a new policy. The frozen 0.30 clarification cost was not
changed.

All three V119 failures had posterior decision regret below the 0.7760 historical baseline. They failed only
after adding the universal query cost:

| Condition | Decision regret before query cost | Maximum affordable query cost | Total regret | Excess | Minimum zero-loss skip fraction |
| --- | ---: | ---: | ---: | ---: | ---: |
| uniform, correlation 0.25 | 0.48650 | 0.28954 | 0.78650 | 0.01046 | 3.486% |
| uniform, correlation 0.50 | 0.47745 | 0.29859 | 0.77745 | 0.00141 | 0.469% |
| strong, correlation 0.50 | 0.48657 | 0.28947 | 0.78657 | 0.01053 | 3.508% |

Thus V119's evidence has substantial decision value: it lowers decision regret by approximately
0.2895--0.2986 in the failing conditions. The fixed 0.30 cost is only slightly larger. In the idealized
zero-loss limit, skipping at most 3.51% of queries would close every observed gap.

This does not prove selective querying works. A pre-query trigger can save cost only if it identifies cases
where skipping clarification does not surrender more decision value than it saves. V120 has not found such
a trigger and cannot infer one from aggregate metrics.

Freeze V120. A next branch may preregister a model-free feasibility audit for an independently specified
pre-query signal and must charge missed-decision value explicitly. It may not mine V119 records, lower the
query cost, reuse post-query observations as a trigger, or authorize language/model evaluation.
