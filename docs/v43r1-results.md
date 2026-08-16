# V43r1 results: graph measurement repair

Decision: `measurement_repair_pass_preregister_deterministic_delay`.

V43's original failed outcome remains immutable. V43r1 is a preregistered correction over the same paired development data, not an independent replication.

| Metric | Result |
|---|---:|
| Original ordered-list graph exact | 0.262 |
| Repaired canonical graph exact (1147 graphs) | 1.000 |
| Duplicate-free | 1.000 |
| Semantic content mismatches | 0 |
| Comparator permutation invariance | 1.000 |
| Other V43 metrics reproduced | 1.000 |
| Other V43 gates passed | 1.000 |

All repair gates passed: `true`.

Post-result integrity audit: `pass`.
