# V153 model-free question-order comparators plan

## Purpose

V153 freezes the reference policies that any later local question-order model must beat. It reads only V152 development metadata and uses no language model. The authoritative state universe remains complete; the policy changes only the order of six registered questions.

Each of the 96 development request fixtures becomes one sequential episode when its visible language is decidable and two episodes when it is deliberately ambiguous, for 120 episodes total. An irrelevant question produces no typed selection, leaves the state at `A00`, and costs 0.3. The discriminating question produces the already frozen closed answer and an exact trusted witness.

## Comparators

- `NO_QUERY` returns safe `A00` at cost 1.0 on each decidable episode.
- `SOURCE_ORDER` asks questions in frozen catalog order.
- `SEEDED_RANDOM` uses a deterministic per-fixture shuffle under the registered seed.
- `ORACLE_ORDER` places the hidden discriminating question first and is a feasibility ceiling, not a deployable policy.

The study records correct-query rank, mean decision cost, improvement over no-query, final exact accuracy after trusted answers, irrelevant-intermediate fail-closed rate, and hypothesis retention. It must not read evaluation language or metadata and has no candidate-state proposal input.

Passing authorizes only prospective design of a local question-order protocol. It does not authorize model access, a direct-versus-reasoning comparison, evaluation access, calibration, threshold fitting, API use, training, induction, authority, action, or execution.
