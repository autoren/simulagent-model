# V175 certification-aware planner development results

## Verdict

V175 is a strong positive development result. Once the planner optimized the routed system's actual objective—trusted
completion only after exact consensus, otherwise deferral—the information-gathering mechanism became useful without
relaxing the safety gate.

Across all 132 V172 development states and 4,224 targets, the exact certification-aware policy achieved mean routed
risk `1.0656159932`. This was strictly below immediate deferral (`2.0`), the unchanged V167 recommendation planner
routed through the same gate (`2.1806264617`), greedy class-information gain (`1.0907624942`), uniform random query
order (`1.0871928072`), and the best fixed open-loop query subset (`1.1666666667`). It was no worse than every
operational control in every state and strictly improved over immediate deferral in all 132 states.

The non-operational target-informed certificate oracle reached `1.0416703029`, leaving a small but nonzero value-of-
target-information gap. This comparator did not receive operational authority.

## Mechanism result

The exact policy used an average of `3.9894932654` inspections and obtained trusted completion probability `2/3`.
The remaining `1/3` corresponded to provisional primitives, which were correctly deferred. This is the structural
outcome predicted by V174: the trusted alias and composition classes require full-depth certificates, while a
provisional candidate must never enter the trusted sandbox.

V173 failed because its horizon-two planner optimized classification recommendation loss. It could lower its internal
Bayes risk without ever producing an exact trusted-class certificate. V175 instead assigned stopping the actual routed
loss: zero only for a unanimous trusted version space and two for any deferral. The planner therefore gathered the
additional evidence necessary to cross the unchanged deterministic gate.

## Safety and integrity

All preregistered integrity and safety gates passed:

- all 132 states, 4,224 targets, and 29,568 target-policy scores were included;
- class-balanced priors normalized exactly;
- dynamic-program root risks reconstructed exactly in every state;
- false trusted-route probability was zero;
- provisional sandbox-entry probability was zero;
- planner commit-authorization count was zero;
- all 1,860 simulated trusted transactions reached the exact oracle final state;
- invariants, provenance, independent verification, and restart verification all passed;
- no model, API, registration, real-state mutation, real service call, side effect, or execution occurred.

The trusted route remained a deterministic consequence of the surviving version space. Neither the planner's terminal
label, the hidden target, nor the oracle comparator could authorize a commit.

## Claim boundary and next step

This is mechanism evidence on the already-used V172 development population, not fresh confirmation. The policy,
objective, cost, horizon, tie breaks, gate, and sandbox contract must now remain unchanged.

The justified successor is a separately preregistered exact-signature-disjoint population, preferably using
four-constraint states as anticipated in the post-V173 roadmap. Only after freezing that population without policy
scores should the unchanged V175 mechanism receive one fresh confirmation run. Language models remain out of scope:
V175 tests exact certification-aware control, not open-world utterance understanding.
