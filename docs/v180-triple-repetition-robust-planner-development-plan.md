# V180 triple-repetition robust-planner development plan

## Objective

V179 proves that triple repetition corrects one raw error. V180 asks whether the resulting trusted completion remains
worth its true inspection cost.

The horizon is frozen at four measurement blocks, the first positive V179 horizon. Each block contains three raw
inspections and costs `0.3`; raw cost remains `0.1`. Stopping uses V175's actual routed loss: zero only for unanimous
trusted completion and two otherwise.

## Controls

Compare the exact robust planner with immediate deferral, V175's clean cost-0.1 policy executed unchanged but repriced
at cost 0.3, greedy information gain, the best fixed block subset, uniform random block order, and a non-operational
target-informed robust-certificate oracle. Report expected blocks and raw inspections.

All decoded paths use the V179-proven equivalence, and all trusted terminals pass through the V171 sandbox. No planner,
clean policy, oracle, target, or model can authorize a commit.

## Boundary

V176 is reused as development. Safety, benefit, and control dominance are separate. Any safe negative or mixed outcome
is retained without changing the repetition code, decoder, cost, horizon, routed loss, gate, population, or thresholds.
A positive result requires another frozen exact-context-disjoint population before confirmation. No model, API, real
sensor, registration, trusted real state, service, effect, or execution is allowed.
