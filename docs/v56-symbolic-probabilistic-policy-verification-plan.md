# V56 preregistration: symbolic and probabilistic policy verification

V55 passed nineteen of twenty frozen gates but did not expose decision-relevant delayed effects. V55r1 preserved that failure and supplied a separately sealed confirmation in which `9/16` policies were delay-sensitive and all fourteen supplementary gates passed. The combined outcome authorizes only this preregistration.

## Claim and boundary

V56 asks whether every frozen three-action policy from V55 and V55r1 has the same bounded semantics under three independently structured views:

1. the formal executor that defines typed world, queue, stochastic, and observation semantics;
2. a separate Z3 encoding that checks the support and invariants of every reachable transition; and
3. an external Storm process that checks termination probability, terminal-goal probability, and expected accumulated reward in the compiled DTMC.

This verifies policies, not the planning algorithm in general. It is posterior-expected and bounded to three actions. The source tasks contain no preregistered catastrophe predicate or safety contract, so V56 cannot establish worst-case safety, probability-bounded safety, a guarantee uniform over the continuous parameter interval, or an unbounded temporal property. Those are deliberately outside the claim rather than added after policy selection.

## Frozen policy census

The verification census is exhaustive rather than sampled: all 32 V55 records and all 16 V55r1 records, ordered canonically by cohort and record. The compiler receives only each public record. It reconstructs the policy with the frozen primary planner and must match the root action and root value stored in the sealed source result before any model is admitted to the bundle. This is policy reconstruction, not another V55 or V55r1 evaluation; no baselines, gates, or source metrics are recomputed or reinterpreted.

## Symbolic verifier

Z3Py `4.16.0` is pinned through `z3-solver==4.16.0.0`. Its transition encoder must be structurally independent: it may consume the canonical DSL, but it may not call the belief stepper, planner, independent policy evaluator, or Storm transition compiler.

For each reachable source state, it encodes typed action binding, due-event delivery, conditional evaluation against the pre-action world, deterministic effects, the success and failure support of the one stochastic branch, delayed scheduling, next-world atoms, and queue contents. The compiler-emitted successor set is independently encoded. The XOR of the two relations must be unsatisfiable. Separate checks require complete Boolean worlds, canonical non-overdue queues, exact depth descent, total observation routing, correct terminal labels, and no nonterminal deadlocks.

Before candidate policies are accessible, an exhaustive synthetic audit covers all 256 two-entity Boolean worlds, all five actions, every unique V53/V55r1 template, and empty, due-now, and due-next queue fixtures. Five semantic mutations must all be detected.

## Probabilistic compiler and Storm

The compiled model is a DTMC because the policy has already resolved every action choice. A synthetic root samples the full joint posterior over program, theta quadrature node, world, queue, and root policy node. Each action transition follows the formal executor. The complete observed successor world selects the unique next policy node while the queue remains latent.

After the third action, a staging state is labeled `success` exactly when the public terminal goal holds and then transitions to the `done` absorber. Every action transition receives its negative action cost; the staging-to-done transition receives reward one for success and zero for failure. Storm `1.13.0` runs as a standalone executable on the explicit transition, label, and transition-reward files. Stormpy is forbidden because the official project documentation says it is incompatible with the supported Homebrew installation.

Storm checks:

- `P=? [F "done"]` equals one;
- `P=? [F "success"]` matches a direct executor enumeration; and
- `R=? [F "done"]` matches both the frozen root value and the independent recursive policy evaluator.

The explicit format uses full-precision floating-point posterior weights, so quantitative cross-checks use a preregistered `1e-9` tolerance. Logical support and type claims remain exact in Z3.

## Controls, sealing, and decision rule

Five analytic DTMC fixtures have known answers. Five probabilistic compiler mutations—dropped successor, missing observation branch, flipped success label, missing action cost, and corrupted initial mass—must all fail. Together with the five symbolic mutations, the required mutation kill rate is one.

No candidate model is built before the verifier implementation and control suite are audited and locked. The 48 policy/model directories are then hashed into one manifest and sealed. The candidate runner and output parser are audited on synthetic models without accessing that bundle and locked separately. Exactly one sealed candidate verification is permitted.

All gates are noncompensatory. A pass qualifies only bounded symbolic well-formedness and posterior-expected probabilistic verification of these 48 policies. It does not relabel V55 as a standalone pass, create a safety guarantee, or authorize a long-horizon, learned-planning, language, or model claim.
