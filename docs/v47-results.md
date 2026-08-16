# V47 results: sampled transition-probability estimation

Decision: `sampled_estimation_pass_preregister_stochastic_language_composition`.

V47 tests finite-sample estimation from realized trajectories under a declared ontology, DSL, and finite probability vocabulary. It is non-final and language-free.

| Metric at 128 trials/intervention | Result |
|---|---:|
| Mean joint-distribution TV | 0.0000 |
| Probability parameter MAE | 0.0000 |
| MAP schema recovery | 1.000 |
| Mean target-program posterior | 1.000 |
| Held-out trajectory log loss | 0.5617 |
| Calibration error | 0.0070 |
| Improvement over uniformized | 0.0833 nats |

Mean TV learning curve: 8 = 0.0049, 32 = 0.0000, 128 = 0.0000.

All preregistered gates passed: `true`.
Post-result integrity audit: `pass`.
