# V67 results: independent bounded policy execution verification

Date frozen: 2026-08-16

## Decision

V67 passed every preregistered noncompensatory gate. The independent executor and Storm reproduced
the bounded exact-posterior execution values of all 96 frozen V66 policies: 48 exact Bayes-adaptive
trees and 48 pooled-SMC² Bayes-adaptive trees. This closes the registered external
identification-to-decision-to-verification sequence for the pinned horizon-three family.

The pass authorizes only preregistration of a broader multi-environment external replication. It
does not authorize a V66 or V67 rerun, a planner-optimality claim, an infinite-horizon or worst-case
safety claim, human-data simulation, model access, or adapter training.

## Verification result

The verifier independently parsed the pinned POBAX source file, rebuilt the 257-node scaled
Beta(2,2) quadrature and persistent clockwise/counterclockwise actuator family, conditioned each
public history, and executed each archived selected-action tree. It did not import or call the V62
parser, V64 filter, V66 planner, or V66 evaluator. Both policy kinds were reexecuted in the same
exact joint environment; archived SMC² branch probabilities were ignored.

Each policy was also compiled to a sealed explicit DTMC. Storm 1.13.0 checked termination and
discounted return, for 192 external property checks in total.

| Metric | Result | Frozen gate |
|---|---:|---:|
| Completed policy fraction | `1.000` | `>= 1.000` |
| Policies / per kind | `96 / 48` | `>= 96 / 48` |
| Maximum independent-to-V66 value error | `5.00e-16` | `<= 1e-10` |
| Maximum Storm-to-independent return error | `5.55e-16` | `<= 1e-9` |
| Maximum Storm termination error | `0.00` | `<= 1e-10` |
| Maximum exact-minus-SMC² paired reproduction error | `2.78e-17` | `<= 1e-10` |
| Source-policy and record binding | `1.000 / 1.000` | `>= 1.000 / 1.000` |
| Root-belief, node, branch, transition, finite rates | all `1.000` | each `>= 1.000` |
| Nonterminal deadlocks | `0` | `<= 0` |
| Analytic fixtures / mutation kills | `6/6 / 14/14` | both `1.000` |

No bundle hash, source mutation, tool-version, attempt-count, truth-access, human-record,
model-forward-pass, or adapter-training violation occurred. The sole attempt produced no terminal
failure artifact.

## Integrity nuance

One V66 row was inspected before the V67 design lock to fix the archived tree interface. This was
disclosed prospectively. The verification population was exhaustive and deterministic—no row was
selected, rejected, altered, or executed before the implementation and evaluator locks—so the
inspection cannot create a favorable subset, but V67 should not be described as blinded.

V67 verifies the execution semantics and reported values of the archived policies. It does not
independently rerun the V66 optimization search or prove that its exact policy is globally optimal.
It is posterior-expected bounded model checking, not a safety specification and not a guarantee
uniform over every continuous parameter value.

## Strongest supported claim

> On the pinned POBAX 4×3 nonterminating model with the project-authored persistent actuator family,
> an independent exact posterior executor and Storm reproduced the horizon-three discounted values
> of all 48 frozen exact and all 48 frozen pooled-SMC² contingent policies to floating-point
> precision, with complete reachable observation branches and no nonterminal deadlocks.

The appropriate next experiment is a prospectively fixed external replication across multiple
POMDP models with varied observation structure, rewards, and terminal behavior. Scalability or
longer-horizon approximation should follow only after that family-level replication preserves the
same source binding, calibration, decision, and independent-verification discipline.
