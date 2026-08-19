# V129 Complete Clarification-interface Audit Plan

## Question

V127--V128 localized a failure in the frozen candidate-specific channel: after rejecting a wrong known
candidate, it can learn that the request is declared but cannot name the correct alternative intent. V129
asks whether a complete typed answer over all eleven safe hypotheses repairs that defect in principle.

## Prospective census

Enumerate all 66 truth/candidate pairs: each of eleven frozen hypotheses crossed with each of six presented
known candidates. Weight pairs uniformly only for mechanism comparison; this is not a prevalence estimate.
No benchmark record or language is read.

A single typed answer names one of the eleven hypotheses. At 90%, 95%, and 100% reliability, test symmetric
error plus two adversarially structured regimes that put 75% of error mass on the presented candidate or
on `A00`. Compare a channel-aware Bayesian policy with a policy that incorrectly assumes symmetric errors.
The frozen V119 candidate-specific channel at 95% reliability, correlation 0.25, and the same 0.30 cost is
the direct comparator.

## Gates

At 95% reliability, the channel-aware complete interface must pass regret, known, unsupported, and
false-known gates under every prior and error bias. It may be neither worse in regret nor worse in known
accuracy than the candidate-specific comparator. The symmetric-assumed planner must remain within regret
and false-known limits under both biased regimes. Perfect answers must yield 100% known and unsupported
decisions. Full hypothesis retention, zero pair-level output, and zero execution are mandatory.

Passing establishes only abstract interface feasibility and authorizes a separately locked realization
audit. It does not establish that any human or model supplies 95%-correct typed answers and does not open
language, protected data, capability induction, richer planning, APIs, training, authority, or execution.
