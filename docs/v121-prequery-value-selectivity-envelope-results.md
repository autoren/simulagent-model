# V121 Pre-query Value-selectivity Envelope Results

## Outcome

V121 passed as a necessary-condition audit:

> `value_selectivity_envelope_derived_requires_paired_fresh_trigger_validation`

It read no individual records and certified no trigger. It derived how selective a future pre-query signal
must be if it is to close V119's regret gaps without hiding lost decision value.

The queried subset's average clarification value must exceed the 0.30 query cost. Relative to the whole
population's average query value, the required lift is 3.64% for strong-prior correlation 0.50, 3.61% for
uniform-prior correlation 0.25, and 0.47% for uniform-prior correlation 0.50.

Skipping a small fraction is more demanding than those lift percentages suggest:

| Condition | Maximum skipped value at 5% skip | At 10% | At 25% |
| --- | ---: | ---: | ---: |
| strong, correlation 0.50 | 0.0895 | 0.1947 | 0.2579 |
| uniform, correlation 0.25 | 0.0908 | 0.1954 | 0.2582 |
| uniform, correlation 0.50 | 0.2718 | 0.2859 | 0.2944 |

In the two harder conditions, a 5% skipped subset may contain only about 31% of the population-average query
value. The zero-value skip calculation from V120 was therefore an optimistic lower bound, not a proposed
policy.

Aggregate results cannot show whether any pre-query observable actually isolates these cases. The next
permitted action is only an inventory of independently available pre-query signals, including provenance,
availability before clarification, mutability, dependence on the LLM candidate, and risk of leaking
post-query or outcome information. No signal may be evaluated or tuned yet.
