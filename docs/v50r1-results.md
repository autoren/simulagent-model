# V50 results: history-dependent belief filtering

Decision: `history_dependent_belief_filtering_pass_preregister_supported_language_composition`.

V50 is a non-final exact symbolic development test. It tests temporal evidence retention under known value-independent masks and condition-matched scoring; it does not test language, active intervention selection, noisy sensors, continuous probabilities, or open ontologies.

| Metric | Value |
|---|---:|
| Mean complete-history suffix TV | 2.50443e-57 |
| Oracle-program suffix TV | 0 |
| Complete-history condition-matched regret | 0 |
| Oracle history-dependent query fraction | 1.000 |
| Mean oracle full-history vs latest-only TV | 0.378255 |
| Latest-only log-loss disadvantage | 0.516419 |
| Time-shuffled log-loss disadvantage | inf |
| MAP-state collapse disadvantage | inf |
| Partial-minus-full condition-matched regret | 0 |
| Raw partial-minus-full log loss (non-gating) | 0.0956573 |
| Oracle conditional-entropy gap (non-gating) | 0.0956573 |
| Calibration error | 0.000273408 |
| MAP schema recovery | 1.000 |
| Mean target-program posterior | 1.000000 |
| Probability MAE | 0 |

All preregistered gates passed: `true`.
Post-result integrity audit: `pass`.
