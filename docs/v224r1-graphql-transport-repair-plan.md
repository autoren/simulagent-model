# V224r1 GraphQL transport repair plan

V224r1 changes only how the exact locked GraphQL document and variables are transported. Instead of repeated CLI form
fields with the same name, it sends a JSON request containing distinct `query` and `variables` members to GitHub's
official GraphQL endpoint. The query text, selected fields, search slices, cutoffs, exclusions, outcome definitions,
sampling, thresholds, firewall, and branch rule remain byte-identical under the V224 lock.

The already captured pinned scope-policy snapshot is reused at its exact hash. No record metadata was returned during
the failed attempt. V224r1 authorizes one repaired metadata capture followed by the original locked V224 scorer. It
does not authorize request language, model access, protected records, training, registration, mutation, action, side
effects, or execution.

