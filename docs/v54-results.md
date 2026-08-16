# V54 results: exact one-step expected information gain

Decision: `authorize_short_horizon_exact_bayes_adaptive_planning_preregistration_only`.

V54 tests exact, one-step, open-loop assay selection for identifying `(program, theta)` while integrating out hidden dynamic state. It does not test reward planning, learned acquisition, language, or model access.

| Metric | Value |
|---|---:|
| Max candidate EIG error | 6.88338e-15 |
| Optimal-set membership | 1.000 |
| Mean oracle EIG | 0.487360 |
| Mean oracle advantage over random | 0.185013 |
| Informative record fraction | 0.844 |
| Controls detected/dominated | 6 |
| Adaptive SBC minimum p-value | 0.12527 |
| Adaptive SBC max rank z | 2.32379 |
| Adaptive SBC max coverage z | 3.72833 |

All preregistered gates passed: `true`.
Post-result audit: `pass`.
