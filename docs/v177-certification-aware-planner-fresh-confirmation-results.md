# V177 certification-aware planner fresh-confirmation results

## Verdict

V177 is a strong fresh confirmation of the V175 certification-aware mechanism within the fixed finite ontology.

The unchanged exact policy achieved mean routed risk `0.9956206756` on all 135 frozen V176 states and 2,160 targets.
This was strictly below immediate deferral (`2.0`), the unchanged V167 recommendation planner routed through the same
gate (`2.1828083028`), greedy class-information gain (`1.0210283544`), uniform random query order
(`1.0128928684`), and the best fixed open-loop subset (`1.0666666667`). The exact policy strictly improved over
immediate deferral in all 135 states and was no worse than every operational control in every state.

The target-informed non-operational oracle reached `0.9767385701`. The small remaining gap confirms that target
information still has value, but the oracle had no routing or commit authority.

## Behavior

The exact policy used an average of `3.2895400895` additional inspections and reached trusted completion probability
`2/3`. Provisional targets, carrying the remaining one third of class-balanced mass, were deferred. The four-constraint
population leaves at most four informative valuations, so the unchanged maximum horizon of five did not force an
extra query.

All 2,160 target certificates were valid. Their exact minimal depths were:

- 1 query: 1,344 targets;
- 2 queries: 462 targets;
- 3 queries: 36 targets;
- 4 queries: 318 targets.

This new certificate geometry differs from V174's three-constraint development population, yet the unchanged routed-
risk planner retained its advantage.

## Safety and integrity

Every preregistered safety and integrity gate passed:

- complete coverage of 135 states, 2,160 targets, and 15,120 target-policy scores;
- exact prior normalization and dynamic-program risk reconstruction;
- exact oracle-certificate validity;
- zero false trusted routing;
- zero provisional sandbox entry;
- zero planner commit authorization;
- exact final state, invariant preservation, provenance, independent verification, and restart verification for all
  1,590 simulated trusted transactions;
- zero model/API use, registration, real-state mutation, service call, external side effect, or execution.

## Scientific boundary

The exact target-context signatures are disjoint from V172/V175, so this is fresh context-generalization evidence.
Candidate identities use the same fixed ontology, so it is not unseen-concept, open-world-language, external-data, or
deployment evidence.

V177 should not be rerun or repaired. The justified next branch is a separately preregistered exact robustness study,
with controlled observation noise or bounded model misspecification introduced before population outcomes are opened.
Language models remain dormant until they can be introduced as an explicitly untrusted observation/proposal channel
with an utterance-level identifiability protocol.
