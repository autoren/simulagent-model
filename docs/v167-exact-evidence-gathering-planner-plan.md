# V167 exact evidence-gathering planner plan

## Purpose

V166 left 48 correct 64-candidate version spaces and no model residual. V167 asks whether two costly,
noise-free interventions can improve a later expressibility decision, and whether the best second intervention
depends on the first outcome.

This is explicitly development-informed. Two exploratory calculations over the project-authored hidden
development records were used to choose the class-balanced prior, two-query horizon, 0.1 query cost, and terminal
loss. Formal policy artifacts and scores were not persisted before this lock. V167 is therefore mechanism and
feasibility evidence, not fresh confirmation.

## Decision problem

Each case begins with the exact V166 64-candidate version space. Prior mass is one third per expressibility class
and uniform inside each class; this prevents raw candidate multiplicity from becoming an accidental prior.
An action queries one valuation whose truth value still varies. The response is deterministic and removes every
inconsistent candidate. At most two queries may be asked, each costs 0.1, and the policy may stop early.

Terminal choices are alias, composition, provisional primitive, or defer. False provisional creation costs 12,
confusing alias with composition costs 4, forcing a registered class on a provisional truth table costs 6, and
deferral costs 2. Every belief and decision remains shadow-only.

## Controls and gates

The exact Bayes-adaptive policy is compared with immediate Bayes stopping, forced MAP without deferral, uniform
random open-loop query pairs, adaptive greedy class-information gain, the optimal fixed open-loop query pair, and
an oracle-class lower bound.

All 48 cases must retain the hidden target and receive positive value from exact adaptive querying. There must be
more than one root query and at least one case whose best second action changes with the first answer. Exact Bayes
must be no worse than every non-oracle control on every case and strictly improve over the optimal open-loop pair
on at least one case. Renaming-equivalent risks must match exactly.

No evaluation population, human judgment, model, API, training, registration, trusted-state mutation, service
call, side effect, real action, or execution is allowed. Passing authorizes only a separately locked Track B
fixed-ontology reversible-sandbox design.
