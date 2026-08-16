# V49 results: passive partial observation

Decision: `prediction_may_pass_without_demonstrated_persistent_belief_use`.

V49 is a non-final symbolic development test under known noiseless observation masks. It does not test language, active intervention selection, noisy sensors, continuous probabilities, or open ontologies.

| Metric | Value |
|---|---:|
| Mean conditional latent-suffix TV | 4.73649e-22 |
| Oracle-program partial TV | 0 |
| Held-out conditional log loss | 0.606088 |
| MAP schema recovery | 1.000 |
| Mean target-program posterior | 1.000000 |
| Probability MAE | 0 |
| Partial-minus-full TV | 1.59957e-22 |
| Partial-minus-full log loss | 0.445646 |
| MAP-state collapse disadvantage | inf |
| History-ablation disadvantage | 0 |

All preregistered gates passed: `false`.
Post-result integrity audit: `pass`.
