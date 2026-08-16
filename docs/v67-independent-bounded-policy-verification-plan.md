# V67 preregistration: independent bounded policy execution verification

Date preregistered: 2026-08-16

## Decision being tested

V67 will test whether every frozen V66 exact Bayes-adaptive and pooled-SMC² contingent policy has
the bounded exact-posterior value already reported by V66 when executed by two new routes: a
separate scalar recursive interpreter and the standalone Storm probabilistic model checker. The
population is exhaustive: 48 records times two policy kinds, or 96 policies. No policy may be
selected, rejected, repaired, or replaced.

One V66 record was inspected before this lock solely to fix the archived policy-tree interface. That
inspection is disclosed because this is deterministic exhaustive verification, not a blinded
statistical evaluation. No policy was executed and no gate was fitted to a selected subset.

## Independent semantics

The verifier will read the pinned POBAX `4x3_nonterminating.POMDP` file with a new minimal parser;
it may not call the V62 parser. It will independently reconstruct the 257-node scaled Beta(2,2)
quadrature, the clockwise/counterclockwise persistent actuator family, reset-observation
conditioning, and the exact history filter. It may not import or call the V64 filter, V66 planner,
V66 policy evaluator, or V66 evaluator.

For each archived policy, the verifier starts from the exact joint posterior conditioned on the
public history. At each reachable policy node it uses the archived selected action, enumerates all
identity, theta, current-state, successor-state, and observation terms, and updates the exact joint
belief. Every positive-probability observation must have exactly one archived child until the third
action. Archived SMC² branch probabilities are ignored: both policy kinds are executed in the same
exact environment.

The explicit DTMC contains one state per reachable policy observation-history node plus an
absorbing `done` state. A branch probability is the independently recomputed predictive observation
probability. Its transition reward is the source expected immediate reward conditional on that
observation, multiplied by `0.95^depth`. Therefore `R=? [F "done"]` is the complete horizon-three
discounted return. `P=? [F "done"]` must be one.

## Freeze sequence

1. Audit and freeze this design. Only implementation work becomes authorized.
2. Implement the independent parser, family, scalar executor, DTMC compiler, explicit writer, and
   tests without loading or executing the 96 policies.
3. Freeze the implementation after all six analytic fixtures and all fourteen mutants pass.
4. Build all 96 bundles, bind each to the frozen V66 row and source hashes, audit every reachable
   invariant, and seal a canonical manifest.
5. Freeze a durable one-shot evaluator that refuses an unsealed or mutated bundle.
6. Reserve one attempt, run Storm on all 96 sealed models, independently reaggregate the result,
   and freeze either the success or the discrepancy without changing V66.

## Noncompensatory gates

All 96 policies and 48 policies per kind must complete. Source hashes, record bindings, root-belief
normalization, reachable-node invariants, positive-observation totality, transition normalization,
and finite results must each pass at rate one; no nonterminal deadlocks are allowed. The independent
executor must reproduce each frozen V66 policy value within `1e-10`. Storm termination error must
be at most `1e-10`, and Storm return must match the independent executor within `1e-9`. The 48
exact-minus-SMC² paired differences must reproduce V66 within `1e-10`. Every analytic fixture and
mutation control must pass. The bundle, source, tool-version, attempt, truth-access, human-data,
model-access, and adapter-training gates are all zero-tolerance.

## Claim boundary

A pass qualifies only bounded horizon-three posterior-expected execution of the 96 archived V66
policies on the pinned external family. It does not prove the V66 planning algorithm, unbounded or
infinite-horizon behavior, worst-case safety, guarantees uniform over continuous parameters,
independent benchmark replication, human-language grounding, or any model/adapter claim.
