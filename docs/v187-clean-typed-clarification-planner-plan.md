# V187 clean typed-clarification planner development plan

## Question

Given the identifying V186 codebook, can an exact target-blind adaptive policy select a small number of clean typed questions at lower expected cost than generic clarification and a fixed open-loop sequence?

## Frozen decision problem

Every observed V183 development identity starts from the complete 14-contract version space. The prior is the empirical frequency of the 14 target contracts among the 120 frozen observed development bindings; all contracts must have positive mass. Any protected successor must reuse this development prior unchanged.

The 164 V186 questions are collapsed only when they induce the same binary column over all 14 contracts, retaining the first question in frozen order. A clean typed answer is the dataset-provided V186 oracle bit. It is not a human answer, a model answer, or deployed evidence.

Each typed question costs 0.10 and the horizon is four questions. A singleton version space terminates exactly at no added cost. A generic trusted clarification costs 0.40 and returns the exact contract/status. Safe deferral costs 0.50 and remains insufficient. Missing observations remain insufficient at zero cost.

## Policies

Compare exact Bellman planning with the best target-blind fixed open-loop sequence, adaptive greedy weighted information gain, frozen source order with generic fallback, always-generic clarification, and immediate deferral. A target-informed minimum-certificate oracle is a non-operational lower bound.

The exact, greedy, and source-order policies may use only the current version space, frozen prior, answer history, remaining horizon, and frozen codebook. No policy may use a record ID or hidden target when choosing a deployable question.

## Gates

All observed exact-policy paths must retain the hidden target and terminate exactly through a singleton or generic trusted answer. Missing controls must remain insufficient. Exact adaptive planning must complete at least 20% of observed paths using typed questions alone, cost no more than 0.38 on average, improve by at least 0.02 over always-generic clarification and by at least 0.005 over the best fixed open-loop sequence, and exhibit at least two distinct reachable history-conditioned question choices.

Failure is retained as a clean negative or mixed boundary. Passing authorizes only a separately preregistered correlated-error protocol. It does not authorize protected-language access, a model/API run, registration, trusted-state mutation, service calls, action, or execution.
