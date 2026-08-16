# V66 results: external Bayes-adaptive reward decisions

Date frozen: 2026-08-16

## Decision

V66 passed every preregistered noncompensatory gate and authorizes only independent verification
of the frozen bounded policies. It does not authorize a rerun, an infinite-horizon claim, a safety
claim, or formal verification before a separately frozen verification design.

The one immutable evaluation used the unchanged 48 public V65r3 histories, with eight records at
each prefix length from zero through five. It ran fresh pooled SMC² inference at 509 outer theta
particles per identity, 127 inner state particles, and three independent repeats. The repeat
posteriors were pooled before horizon-three planning, and the state conditional was
Rao–Blackwellized exactly for every retained static atom. No truth audit, V64/V65 evaluation result,
human record, model forward pass, or adapter-training run was accessed.

## Primary result

Every pooled-SMC² contingent policy had the same exact-posterior value as the exact
Bayes-adaptive policy, up to floating-point roundoff:

| Metric | Result | Frozen gate |
|---|---:|---:|
| Mean exact value regret | `4.63e-18` | `<= 0.005` |
| Q95 exact value regret | `2.78e-17` | `<= 0.02` |
| Maximum exact value regret | `2.78e-17` | `<= 0.08` |
| Strict root-optimal membership | `1.000` | `>= 0.85` |
| 0.005-reward epsilon membership | `1.000` | `>= 0.95` |
| Mean root-Q absolute error | `3.27e-18` | `<= 0.01` |
| Q95 root-Q absolute error | `1.39e-17` | `<= 0.04` |
| Mean SMC² self-value calibration error | `0.000199` | `<= 0.015` |
| Q95 SMC² self-value calibration error | `0.000570` | `<= 0.05` |
| Maximum SMC² self-value calibration error | `0.00222` | descriptive |

The exact Bellman/root-Q, independent policy-evaluation, and oracle-dominance residuals were at
most `5.55e-17`, `2.78e-17`, and `2.78e-17`, respectively. All beliefs and predictive branches
normalized, all four actions were retained, and every registered tie break was valid.

## Comparators and controls

The posterior-weighted known-model oracle had mean value `0.02011`, versus `0.018996` for exact and
pooled-SMC² Bayes-adaptive planning. The valid 32-point persistent posterior-sampling mixture had
mean value `0.01655`; its 64-point sensitivity value was `0.01658`. Each mixture point selected one
static identity-theta model once, retained it for the whole contingent policy, and was evaluated
under the full exact environment posterior. It was never replaced by a per-step mean transition.

Five controls were detected or dominated, exceeding the minimum of four:

- information-only EIG had mean exact-value regret `0.13579` and root disagreement `0.50`;
- myopic expected reward had mean regret `0.17429` and root disagreement `0.50`;
- the invalid mean-transition semantics was directly rejected as a valid mixture;
- shared random streams were detected by the inherited frozen mutation audit; and
- outcome/truth leakage was rejected by the implementation firewall.

Three controls were not empirically separated:

- joint-MAP certainty equivalence matched exact value and root actions;
- the first SMC² repeat alone matched exact value and root actions; and
- the persistent posterior-sampling mixture had mean regret `0.00245` and root disagreement
  `0.0482`, below the registered control-detection rule.

This nuance is load-bearing. V66 establishes that the deployed pooled posterior preserves the
frozen three-action reward policies, but it does not show that pooling is necessary for this reward
task or that Bayes-adaptive planning strictly beats MAP on these 48 histories. The model uncertainty
is decision-relevant in the benchmark construction and active acquisition is nontrivial, yet the
short reward horizon often admits the same action under several posterior approximations.

## Integrity and boundary

The run completed in `298.66` seconds. It reserved one attempt before loading the subset, wrote 48
record cells and 144 single-repeat diagnostics, produced no failure artifact, and cannot be rerun.
An independent outcome auditor reproduced every aggregate, gate, control, and prefix-length result
from the raw cells and verified all frozen policy trees.

The strongest claim is:

> On the pinned POBAX 4×3 nonterminating model with the project-authored persistent actuator family,
> the frozen three-repeat pooled SMC² posterior produced horizon-three contingent reward policies
> whose exact-posterior values and root actions matched exact Bayes-adaptive planning on all 48
> sealed public histories, while its self-estimated values remained well calibrated under the
> registered gates.

This is one externally anchored family, one source reward, one bounded horizon, and a paired reuse
of public histories. It is not independent benchmark replication, general external unknown-dynamics
transfer, infinite-horizon optimality, a safety property, natural-language grounding, or a
model/adapter result.
