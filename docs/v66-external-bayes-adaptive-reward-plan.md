# V66 preregistration: external Bayes-adaptive reward decisions

Date frozen: 2026-08-16

## Question

V65r3 established that the pooled SMC² posterior preserves one-step active-acquisition choices. V66
asks the downstream question: does that approximate belief also preserve bounded reward decisions
when actions affect both physical state and future information?

The experiment uses the unchanged 48 public histories, the pinned nonterminating POBAX 4×3 arrays,
the project-authored persistent actuator identity and theta family, and the original source reward
and discount. No truth audit or realized trajectory is used. Every policy is evaluated under the
same exact joint posterior predictive conditioned on its public history.

## Three-action Bayes-adaptive planning

For horizon three, each Bellman action receives its exact expected immediate source reward. The
planner then enumerates all six observations, updates the joint belief, and recursively selects the
next action. The static identity-theta pair remains fixed throughout the tree. The canonical action
order is `n, e, s, w`; ties within `1e-12` reward units choose the first action.

The exact planner uses the two identities, 257 theta quadrature nodes, and 11 physical states. The
approximate planner uses fresh V66 inference from the frozen V65r3 SMC² implementation: 509 outer
theta particles per identity, 127 inner particles, and three independent repeat posteriors pooled
before planning. Its dynamic state conditional is Rao–Blackwellized exactly per pooled static atom.
This is V66 inference for reward planning, not a V65r3 evaluation rerun.

## Common exact evaluation

An approximate policy's self-estimated value is not the primary outcome. Each action mapping is
executed recursively under the exact joint belief, integrating every future observation. Primary
regret is exact Bayes-adaptive value minus the exact-posterior value of the pooled-SMC² policy.

The comparison set is:

- exact Bayes-adaptive planning;
- pooled-SMC² Bayes-adaptive planning;
- a posterior-weighted model oracle that reveals the persistent static model but not physical state;
- a joint-MAP certainty-equivalent planner;
- a deterministic 32-point systematic-quadrature approximation to a valid posterior-sampling
  mixture that draws one static model once and follows its model-known POMDP policy for the full
  horizon;
- myopic expected reward;
- information-only EIG; and
- an explicitly invalid mean-transition negative control.

The oracle is an upper bound from perfect static-model information, not full-state observation. The
posterior-sampling mixture is valid precisely because the sampled model persists. Its primary
value uses 32 fixed inverse-CDF quantiles of the exact static posterior and is checked against a
64-point sensitivity diagnostic; each selected model's contingent policy is evaluated under the
full exact environment posterior before averaging. This is a bounded deterministic approximation
to the randomized policy, not exact 514-atom enumeration. Averaging transition matrices
independently at every step changes the generative model and is never a valid mixture comparator.

## Gates and boundary

The primary pooled-SMC² policy must have mean/q95/max exact value regret no larger than
`0.005/0.02/0.08`, strict root-optimal membership at least 0.85, and 0.005-reward ε-optimal
membership at least 0.95. Root Q error, self-value calibration, noninferiority margins versus MAP
and the persistent mixture, exact-reference agreement, normalization, controls, complete work, and
one-shot access are separate noncompensatory gates.

A pass authorizes only independent verification of the frozen bounded policies. It does not establish
infinite-horizon optimality, safety, independent benchmark replication, natural-language grounding,
or any model/adapter result.
