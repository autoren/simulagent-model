# V56 results: symbolic and probabilistic policy verification

Decision: V56 qualifies bounded posterior-expected verification of all 48 frozen three-action policies. It authorizes only preregistration of the next definition-transfer and human-authored-language tracks.

## Sealed results

- All 25 noncompensatory gates passed: `True`.
- Policies completed: `48/48` (`{"v55": 32, "v55r1": 16}`).
- Root-action reproduction rate: `1.0`.
- Maximum root-value reconstruction error: `0.0`.
- Reachable state invariant proofs: `418139` at rate `1.0`.
- Reachable transition-support proofs: `214081` at rate `1.0`.
- Observation-totality checks: `263682` at rate `1.0`.
- Z3 unknowns / nonterminal deadlocks: `0` / `0`.
- Storm completion rate: `1.0`.
- Maximum termination-probability error: `0.0`.
- Maximum success-probability error: `3.9345748881203235e-11`.
- Maximum return error vs frozen value: `3.7388869777998934e-11`.
- Maximum return error vs independent evaluator: `3.7388869777998934e-11`.
- Integrity violations: `0`.

## Claim boundary

This verifies only bounded, three-action execution of the frozen V55/V55r1 policies under their posterior mixture. It does not establish worst-case safety, parameter-uniform guarantees, unbounded or long-horizon behavior, planner optimality outside the sealed tasks, or open-language grounding.
