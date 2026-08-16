# V48 results: stochastic language composition

Decision: `stochastic_language_composition_pass_preregister_passive_partial_observation`.

V48 is a non-final paired composition test. It does not test open language, probability language, passive partial observation, or active intervention selection.

| Metric | Language | Matched symbolic |
|---|---:|---:|
| Mean joint-distribution TV | 1.56109e-16 | 1.56109e-16 |
| Held-out trajectory log loss | 0.545981 | 0.545981 |
| Calibration error | 0.004883 | 0.004883 |
| MAP schema recovery | 1.000 | 1.000 |
| Mean target posterior | 1.000000 | 1.000000 |

Language-minus-symbolic TV: `0`.
Language-minus-symbolic log loss: `0`.
Exact trial alignment: `1.000`.
Worst mechanic: `mechanic_915e350ee46f330c` at TV `7.49321e-15`.
Worst family: `delayed_bernoulli_scheduling`; probability: `1/4`; timing: `delayed`; length: `3`.

All preregistered gates passed: `true`.
Post-result integrity audit: `pass`.
