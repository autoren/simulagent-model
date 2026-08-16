# V43 results: declared sequential-language grounding

Decision: `repair_declared_language_compiler_only`.

V43 is a paired, non-final development result. It changes the exposed representation of the sealed V42 cases from symbols to declared controlled language while keeping the stateful reasoner fixed.

| Metric | Result |
|---|---:|
| State-clause exact parse (22072 clauses) | 1.000 |
| State-graph exact (1147 graphs) | 0.262 |
| Action-command exact parse (3024 commands) | 1.000 |
| Action-sequence exact | 1.000 |
| Safety abstention (280 challenges) | 1.000 |
| Compiled target retention | 1.000 |
| Compiled schema recovery | 1.000 |
| Compiled next-state exact | 1.000 |
| Compiled final-observation exact | 1.000 |
| Compiled order-counterfactual accuracy | 1.000 |
| Bag-of-actions order control | 0.000 |
| Literal language lookup | 0.000 |

All preregistered gates passed: `false`.

Interpretation: a pass establishes exact composition of the declared state/action language interface with the frozen deterministic sequential reasoner on the paired V42 cases. It does not establish open-language understanding or an independent new-mechanic replication.

Post-result integrity audit: `pass`.
