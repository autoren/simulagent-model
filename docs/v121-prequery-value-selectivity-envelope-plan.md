# V121 Pre-query Value-selectivity Envelope Plan

V121 derives necessary conditions for a selective clarification trigger from V120's three frozen aggregate
failures. For several fixed skip fractions it computes the greatest average query value that skipped cases
may contain while still matching the historical regret baseline. It also computes how much the queried
subset's average value must exceed the population average.

The audit reads no records and therefore cannot establish that a suitable subset or trigger exists. That
limitation is an outcome condition, not a caveat to be waived. A pass authorizes only an inventory of
pre-query signals whose provenance is independent of post-query outcomes. It does not authorize evaluating,
tuning, or deploying any trigger.
