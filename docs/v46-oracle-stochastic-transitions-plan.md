# V46 oracle stochastic-transition foundation

## Why this is next

V42–V45 established persistent state, fixed delayed effects, and declared language composition. V46 introduces one new capability: probability mass over alternative transitions.

The first stochastic experiment must separate probability semantics from statistical estimation. Therefore support provides exact distribution-valued trajectories rather than finite samples. A pass means the architecture can represent, identify, and execute probability-weighted mechanics exactly. It does not mean the system can estimate probabilities from observed trial frequencies.

## Exact probability semantics

Probabilities are reduced rationals from `{1/4, 1/2, 3/4}`. A stochastic effect is a Bernoulli branch: apply a deterministic payload with the declared probability, otherwise do nothing. Repeated action occurrences make conditionally independent choices given the current world and pending queue.

For delayed stochastic effects, the branch is chosen when the action is evaluated and the chosen payload is placed in the hidden queue. Delivery follows the frozen V44 tick order. The executor carries exact rational mass over `(world, pending queue)` pairs and marginalizes that mass to each post-action observed world distribution. Pending events are not flushed at sequence end.

## Population and oracle contract

The 40-mechanic development population has four equal families: immediate Bernoulli mutation, delayed Bernoulli scheduling, state-conditional probabilities, and interleaved deterministic/stochastic effects. Sequences have two to four actions over two to four entities.

Initial states are complete. Each support sequence exposes the exact distribution over post-action worlds at every step. Queries require the same exact distribution. No sampled realizations are generated or consumed. Supports identify each mechanic within 20 sequences, and query structures are disjoint.

## Controls and gates

The primary system performs exact weighted version-space induction and distribution execution. A uniformized control preserves the possible outcome set but replaces nonzero probabilities with equal mass. A MAP control keeps only the modal outcome. Literal distribution lookup is the non-lifted control.

Probability mass must normalize exactly. Target retention, schema recovery, and exact trajectory-distribution match must all equal 1.000, with mean total variation exactly 0.000 across every family, sequence length, and probability value. Uniformized and MAP exact-match rates must each remain at or below 0.850; literal lookup at or below 0.950.

## Decision

- Exact weighted semantics pass and controls fail: preregister a fresh sampled-transition estimation stage with calibration and proper scoring.
- Mass or execution fails: repair exact probability semantics before sampling or language work.
- Schema recovery fails: revise the finite stochastic DSL.
- A probability ablation passes: redesign queries so probability mass is necessary.

V46 is non-final, language-free, and model-free. It is an oracle probability foundation, not empirical stochastic learning.
