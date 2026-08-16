# V59 budgeted root-sampled planning results

## Outcome

**Qualification: PASS**

The sealed one-shot run evaluated 24 public tasks at horizons 3, 5, and 7, with three search budgets and three replicates per task-budget cell. It accessed no audit-truth records.

## Primary findings

- High-budget exact horizon-3 root-optimal-set membership: `1.000000`.
- High-budget exact horizon-3 mean root regret: `0.000000`.
- High-minus-low budget candidate return: `0.092488`.
- Scale high-budget candidate minus observation-blind return: `0.059989`.
- Scale tasks with positive observation-contingency advantage: `0.687500`.
- Scale task-level paired lower 95% bound: `0.014608`.
- Candidate tree observation-branching rate: `0.902778`.
- Deterministic replay and budget accounting rates: `1.000000` and `1.000000`.

Failed preregistered gates: `none`.

## Claim boundary

This result concerns bounded root-sampled observation-contingent search given the frozen exact calibrated belief and symbolic simulator. It does not establish exact long-horizon optimality, approximate-belief correctness, formal or worst-case safety, human-authored language robustness, or model/adapter performance. V58 remains deferred; synthetic records do not count as human evidence.

## Integrity

The population, evaluator, single attempt, result, audit, and outcome lock are hash-bound. Candidate evaluation opened only the sealed public population. Truth-field, future-observation, and latent-conditioned-rollout access counts are all zero.
