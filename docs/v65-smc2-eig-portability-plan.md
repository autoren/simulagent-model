# V65 preregistration: pooled SMC² external EIG portability

Date frozen: 2026-08-16

## Question and authorization boundary

V64 qualified a pinned, externally anchored four-action active-system-identification benchmark and
an exact one-step expected-information-gain reference. It did **not** test particle acquisition or
authorize reward planning. V65 asks the next narrow question: does the deployed equal-weight pool of
three independently randomized SMC² posterior approximations preserve the V64 exact EIG vector and
the action selected from it?

V65 is a paired portability test on a prospectively hash-selected subset of the already sealed V64
public histories. It is not a fresh benchmark population and must not be described as an independent
replication of V64. A pass authorizes only preregistration of a later Bayes-adaptive reward-decision
stage. It does not authorize that evaluation, formal verification, human-data substitution, model
access, or adapter training.

## Frozen source and family

The source model remains POBAX `4x3_nonterminating.POMDP` at commit
`a5e1d62d14e4efe783885b9d4f19cffa2a568eec`, model SHA-256
`0fa62301931960d682b02961ffd38f4dd6b8e8835bc0203f4a12f849c267d6ff`. POBAX supplies the known
initial distribution, transition arrays, observation kernel, reward tensor, and discount. The
unknown actuator family remains project-authored:

- identity is `clockwise_failure` or `counterclockwise_failure`, each with prior probability 0.5;
- theta has a scaled Beta(2,2) prior on `[0.6, 0.95]`;
- a command succeeds with probability theta and otherwise executes the identity-specific adjacent
  command;
- identity and theta are static, while the 11-state maze location is a dynamic nuisance latent;
- the complete candidate set is `n`, `e`, `s`, `w`, in that tie-breaking order.

The exact reference is the frozen V64 257-node quadrature and exact 11-state filter. V65 recomputes
that deterministic reference only for its selected histories. This is not a rerun of the V64
one-shot evaluation.

## Prospective paired subset

The source is the 192-record public V64 selection file bound in the V64 population seal by SHA-256
`844ba4355ac18f25eaa482b86cf60e7bd5a35a8d4353bf6e3bdd14e4079761ea`. It contains 32 records at
each public prefix length 0 through 5.

After the V65 candidate implementation is independently audited and frozen, the subset constructor
will partition the public records by prefix length. Within each stratum it will sort records by
`SHA256("v65|subset|6511|" + record_id)`, breaking the cryptographically negligible possibility of
a hash tie by `record_id`, and retain the first eight. The resulting 48 records will all be kept.
Only `record_id`, `prefix_length`, `initial_observation`, `actions`, and `observations` may be loaded
or copied.

The subset constructor and seal auditor may not load the V64 selection audit, V64 evaluation result,
exact EIG values, or selected actions. No record may be rejected, replaced, or reordered because of
particle behavior or exact/approximate agreement. The subset file and manifest are sealed before the
candidate evaluator is written.

## Frozen SMC² architecture

V65 preserves the qualifying V53r2/V63r1 deployment rule and its inherited settings:

- outer theta-particle budgets: 31, 127, and 509;
- primary outer budget: 509;
- three independent repeats at every budget;
- 127 inner hidden-state particles per outer particle;
- both discrete identities enumerated;
- outer and inner systematic resampling below ESS fraction 0.5;
- two particle-marginal Metropolis-Hastings rejuvenation moves after each outer resampling event;
- fixed proposal standard deviation 0.4 in the logit of scaled theta;
- no adaptive proposal.

For each identity and outer theta particle, the bootstrap filter samples source initial states,
weights them by the public reset observation, and normalizes. At each public command it propagates
the inner particles through the identity/theta transition mixture, weights the successor states with
the pinned observation kernel, and updates the likelihood estimate. The two identity-specific SMC²
normalizing-constant estimates update the 0.5/0.5 identity prior. A PMMH proposal reruns the complete
bootstrap likelihood for the reset observation and all public action-observation pairs.

Random streams are derived independently from the frozen roots and the complete tuple of stage,
record, identity, budget, repeat, outer particle, phase, tick, inner particle, purpose, resampling
ordinal, and move. Deliberately shared-stream controls must be detected.

## Pooling and approximate acquisition

Each repeat yields a normalized joint posterior measure over identity, theta, and current maze
state. The primary posterior is the equal-weight mixture of the three repeat measures. Pooling occurs
before any posterior metric, predictive distribution, EIG score, or action selection is computed.
This is load-bearing: V63 failed when repeat errors were averaged separately, whereas V63r1 qualified
the prospectively frozen pooled measure.

For one candidate command, each outer identity/theta particle first integrates over its inner state
distribution to obtain a six-outcome predictive distribution. EIG is the mutual information between
the joint static latent `(identity, theta)` and the next observation under the pooled measure. Inner
state and repeat identity are Monte Carlo structure, not information targets. Every one of the four
candidate commands must be scored. The first action in canonical `n`, `e`, `s`, `w` order within
`1e-12` nats of the approximate maximum is selected before any next outcome is available.

## Primary comparisons

The inferential unit is a selected history, with the design balanced by construction across six
prefix-length strata. At each budget, the pooled posterior is compared with the exact posterior on:

- identity total variation;
- theta Wasserstein-1 distance;
- total variation after binning theta into 16 common bins within each identity;
- current 11-state marginal total variation;
- total variation of the six-outcome predictive for each of four candidate commands;
- absolute error in all four EIG values;
- exact-optimal-set membership of the approximate selection;
- membership in the exact `0.001`-nat near-optimal set;
- exact EIG regret of the approximate selection.

The primary 509-particle pooled result must pass every noncompensatory threshold in the frozen
configuration. In particular, mean and q95 EIG-vector absolute error must not exceed `0.004` and
`0.015` nats; strict and `0.001`-nat-near-optimal membership must be at least 0.80 and 0.95; and mean,
q95, and maximum selected-action exact regret must not exceed `0.0015`, `0.006`, and `0.02` nats.
Posterior and predictive thresholds are independently gated. Primary mean EIG error and regret may
not exceed their 31-particle counterparts by more than `0.001` nats.

These thresholds tolerate Monte Carlo approximation while remaining small relative to V64's mean
one-step oracle advantages over fixed and random selection. Accuracy cannot compensate for failed
normalization, incomplete cells, access violations, mutation survival, or integrity failures.

## Mandatory single-repeat and compute diagnostics

Every repeat is retained. For every budget, the report must include repeat-specific EIG-vector
errors, strict and near-optimal membership, exact selection regret, three-repeat selected-action
disagreement, and the best-to-worst repeat regret spread. These are mandatory diagnostics but cannot
qualify a failed pooled posterior or be used to choose a favorable repeat.

For every record, budget, and repeat, the evaluator must report wall-clock time and deterministic
work counters: outer particles, inner transition draws, observation-weight evaluations, complete
history likelihood recomputations, outer and inner resampling events, PMMH attempts and accepts, and
final posterior atoms. Aggregate latency and work scaling by budget and prefix length must appear in
the frozen result. Compute has no pass threshold in V65: an accurate but expensive method remains an
accuracy pass with an explicit efficiency limitation.

## Controls and implementation audit

Before subset construction, the candidate implementation is tested only on synthetic fixtures and
non-evaluation histories. A separately written scalar scorer must agree with particle-measure
posterior, predictive, EIG, and selection computations. Analytic fixtures include normalized convex
transition mixtures, zero EIG for a static-latent point mass, invariance to storage permutation when
canonical names are preserved, and equivalence of mutual-information and expected-posterior-KL
forms.

The mutation audit must kill all registered implementation mutants, including omitted reset
conditioning, wrong actuator permutation, omitted observation likelihood, equalized identity
evidence, state-as-target information, predictive entropy in place of EIG, scoring then averaging
repeats instead of pooling measures, first-repeat-only scoring, disabled outer resampling or
rejuvenation, shared inner streams, wrong observation-action indexing, and candidate omission.

The immutable evaluation also reports eight scientific controls: score-then-average repeat EIG,
first-repeat only, state as target, MAP identity, theta mean, forced equal identity evidence, shared
streams, and attempted outcome leakage. At least six must be detected or dominated under the frozen
rules; outcome leakage and shared streams require direct integrity detection.

## Staged one-shot execution

1. Freeze this design and its source bindings. Authorization: implementation only.
2. Implement candidate SMC² and pooled EIG, independent scalar references, tests, and mutation
   audit. Freeze their hashes. Authorization: subset materialization only.
3. Apply the prospective hash rule to public V64 histories, audit the 8-per-prefix quotas and access
   log, then seal the 48 records. Authorization: evaluator implementation only.
4. Implement and mutation-test the evaluator against synthetic fixtures. Freeze evaluator, design,
   implementation, and subset hashes. Authorization: one logical evaluation only.
5. Run every record × budget × repeat cell once, pool prospectively, compute the frozen exact
   comparisons, and write one immutable result. No gate, budget, record, seed, or aggregation change
   is allowed after this point.
6. Independently audit result bindings, one-shot accounting, access counters, work counters,
   controls, and every noncompensatory gate before freezing the V65 outcome.

If posterior gates fail, repair must target the external state filter, identity evidence, outer
weighting, or rejuvenation in a newly preregistered stage without modifying V64. If posterior gates
pass but EIG fails, repair must target pooled static-latent aggregation or predictive scoring. A V65
failure cannot be repaired by dropping histories, choosing repeats, changing particle budgets, or
weakening gates.

## Claim if all gates pass

On a prospectively hash-selected, prefix-balanced subset of frozen V64 public histories, the frozen
equal-weight pool of three SMC² repeats preserves the exact one-step EIG acquisition decision within
the registered posterior, predictive, vector-error, and exact-regret tolerances. This would authorize
preregistration of external Bayes-adaptive reward decisions against oracle, MAP, and valid mixture
controls.

It would not show sequential approximate adaptive-design superiority, long-horizon Bayes-optimality,
formal safety, upstream POBAX unknown-dynamics support, natural-language grounding, human validity,
or learned-model performance.
