# V172 trusted shadow integration population plan

## Purpose

V172 freezes the cases for the first planner-to-sandbox integration study before any integration policy or
transaction outcome is computed. It is a population experiment, not a performance experiment.

## Complete construction

The generator exhaustively crosses every three-element subset of the eight Boolean valuation indices with all
eight binary outcome assignments. This produces 448 distinct three-constraint states. Every exact 32-candidate
version space is retained. A state is marked integration-eligible only when its frozen candidate metadata contains
all three V167 prior classes; ineligible states remain visible in the population.

For every eligible state, every surviving candidate becomes a frozen simulated target case. There is no target
subsampling. Each target receives the unchanged V167 class-balanced prior weight: one third of mass per class,
uniformly divided among candidates of that class. These weights use only candidate metadata and do not use a
policy, route, transaction, or outcome score.

One implementation build exposed only structural counts and membership hashes so exact population gates could be
recorded. It ran no planner and no sandbox transaction. The formal build must reproduce all identities and counts
exactly under a separate lock.

## Boundary

The records are project-authored finite-DSL truth-table states and simulated target identities. They are not
language, human evidence, external data, or deployment cases. V172 may not run a planner, route a proposal, commit
a transaction, load a model, register a concept, mutate trusted state, call a real service, or execute an action.
A pass authorizes only separate preregistration of the trusted-only shadow integration study.
