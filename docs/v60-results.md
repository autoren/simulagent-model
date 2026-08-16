# V60 approximate-belief decision-calibration results

## Outcome

**Qualification: PASS**

The single sealed run composed the frozen V53r2 SMC² posterior with the frozen V59 search semantics on all 24 V59 public tasks. Candidate evaluation accessed no audit records, and all deployed policies were scored under the exact posterior.

## Posterior agreement at 509 outer particles

- Mean program TV: `0.000098`.
- Mean theta Wasserstein distance: `0.003757`.
- Mean binned program-theta TV: `0.030893`.
- Mean / q95 current-configuration TV: `0.001162` / `0.003028`.

## Decision and return calibration

- Horizon-3 exact optimal-set membership: `1.000000`.
- Horizon-3 mean exact root regret: `0.000000`.
- Approximate/exact-belief search root-action agreement: `0.611111`.
- Exact-belief minus approximate-belief policy return: `0.000468`.
- Scale approximate minus observation-blind return: `0.059071`.
- Scale task-level lower 95% bound: `0.004773`.
- Primary-minus-low inference-budget approximate return: `-0.002231`.

Failed preregistered gates: `none`.

## Boundary

This outcome concerns the frozen SMC² implementation, eight-template symbolic registry, 24-task population, and bounded 1,024-simulation search. It does not establish exact long-horizon optimality, general-purpose or amortized inference, formal safety, human-authored language robustness, or model/adapter performance. V58 remains deferred.

## Integrity

Normalization, simulation-budget accounting, deterministic replay, finite returns, and implementation-mutant detection were noncompensatory. Truth-field, future-observation, and latent-conditioned-rollout access counts are zero; budget accounting was `1.000000`.
