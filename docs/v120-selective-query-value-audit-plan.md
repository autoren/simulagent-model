# V120 Selective Query Value Audit Plan

V120 reads only V119's frozen aggregate result. For every required 95%-reliable, correlation-aware
condition it separates the fixed 0.30 clarification cost from posterior decision regret. For each condition
that missed the historical 0.7760 baseline, it derives the maximum affordable average query cost and the
minimum fraction of zero-loss query skips that would close the gap.

This is not a selective policy and cannot assume that a valid pre-query trigger exists. It only determines
whether selective querying is a quantitatively coherent successor hypothesis. No record, language, model,
or protected data may be read; no cost can be retroactively discounted.
