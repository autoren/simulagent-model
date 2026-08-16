# V54 preregistration: exact one-step expected information gain

V53r2 passed its sealed continuous-parameter SMC², PMCMC, exact-agreement, calibration, degeneracy, correlation, and scale gates. Its frozen outcome authorizes only preregistration of exact one-step expected-information-gain selection. V54 uses that authorization without adding reward, planning, learned acquisition, particle acquisition, language, or model access.

## Scientific target

At each public history, V54 begins from the exact normalized belief

\[
p(M,\theta,s_t\mid h_t),
\]

where `M` is one of the eight V53 parameterized program templates, `theta` is the shared continuous branch probability, and `s_t` is the current world/queue configuration. The acquisition target is deliberately only `(M, theta)`. Current hidden configuration is integrated out as a nuisance variable. This prevents an intervention from receiving credit merely for exposing transient physical state while teaching nothing about the reusable mechanic.

For every legal candidate assay `a`, the primary score is

\[
\operatorname{EIG}(a)
= \sum_o p(o\mid h_t,a)
  \operatorname{KL}\!\left[
    p(M,\theta\mid h_t,a,o)
    \Vert p(M,\theta\mid h_t)
  \right].
\]

The implementation must also reproduce the equivalent prior-entropy minus expected-posterior-entropy form. Natural logarithms make all scores nats.

## One-step assay and complete candidate set

One selected experiment is an open-loop three-tick assay: apply one candidate bound action, then execute two fixed waits, observing the full world-atom panel after each tick. This is still one design decision—there is no within-assay adaptation—but it exposes one- and two-tick delayed effects instead of structurally favoring immediate mechanics.

For a record with `n` entities, the candidate set is exhaustive: `wait`, plus `pulse` and `route` for every ordered distinct actor/target pair. It therefore contains 5 candidates for two entities and 13 for three. No heuristic candidate pruning is allowed. All assays have exactly three environment ticks, three observation panels, and one cost unit, so maximizing EIG is also maximizing EIG per evidence-cost unit. The canonical intervention key supplies the deterministic tie break.

## Exact computation and independent reference

V54 reuses the already validated eight V53 program templates, scaled-Beta(2,2) prior on `[0.05, 0.95]`, 257-node Gauss–Legendre quadrature, and exact world/queue transition semantics. Reusing the model class is intentional: this experiment isolates acquisition correctness rather than claiming transfer to new mechanics.

The implementation under test accumulates predictive outcomes in batches. An independently structured scalar reference nests loops over program, theta node, hidden configuration, and exact masked outcome. Every candidate score and selected action is compared, not only the winner. Analytic fixtures cover zero-information waits, a closed-form binary Bernoulli mutual information case, identical-program zero information, and the entropy-reduction identity.

## Sealed populations

Population construction remains forbidden until implementation tests and an altered-seed audit pass and an implementation lock is frozen.

The later sealed selection population will contain 64 histories, eight per generating template. It balances 16 prior-like all-wait histories, 32 mixed informative histories, and 16 histories with a pending delayed event. Public histories and observation designs must be fresh against V49–V53, although the V53 templates are intentionally reused.

A separate 256-replication adaptive SBC population tests posterior correctness after data-dependent selection. Each policy call sees only the public pre-outcome history. The realized outcome is generated afterward from a disjoint random stream, and the posterior is updated exactly. This detects selection-conditioning bugs that ordinary passive SBC cannot expose.

## Non-compensatory gates

The primary and scalar-reference EIG values must agree for every candidate, selected actions must lie in the exact optimal set, and selected EIG regret must be at most `1e-10` nats. Predictive distributions must normalize; EIG must be finite, nonnegative up to floating tolerance, no larger than prior target entropy, and equal both information-theoretic forms.

At least one quarter of records must have an oracle EIG spread of at least `0.001` nats. Across all records, the exact selector must beat the uniform-random candidate mean by at least `0.001` nats and capture essentially all available EIG. It may never select the no-op on a record where that no-op is strictly dominated beyond the frozen tie tolerance.

Adaptive SBC rank and coverage gates are inherited in spirit from V51/V53. Truth fields, realized outcomes, candidate omissions, noncanonical tie breaks, and history/outcome random-stream collisions all have zero tolerance.

## Controls and decision boundary

Controls include uniform random selection, predictive-observation entropy, state-only information, MAP-program collapse, theta-point collapse, likelihood squaring, and an explicit attempted outcome leak. At least five must be detected or dominated by exact static-latent EIG; outcome leakage counts only if the selection firewall rejects it.

A sealed pass authorizes only preregistration of short-horizon exact Bayes-adaptive planning. It does not authorize reward optimization, planning, learned or particle acquisition, verification, language grounding, open ontology work, model access, training, or a final claim.
