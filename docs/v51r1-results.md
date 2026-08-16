# V51 results: simulation-based calibration of exact joint inference

Decision: `exact_inference_calibrated_authorize_scalable_particle_inference_preregistration`.

V51 is a non-final, prior-predictive calibration test of exact discrete inference over the mechanic program and the current world/queue configuration. It also compares two independently implemented exact paths. It does not test particle approximation, intervention selection, reward, planning, language, noisy sensors, continuous parameters, or open ontologies.

| Metric | Value |
|---|---:|
| Completed replications | 2048 |
| Normalization rate | 1 |
| Maximum exact-path TV | 6e-100 |
| Minimum primary rank chi-square p-value | 0.0568388 |
| Maximum absolute primary rank-bin z | 2.64733 |
| Maximum absolute primary coverage z | 1.8674 |
| Bug controls rejected | 2 / 3 |

## Bug-sensitivity controls

| Control | Rejected | Minimum p | Max rank z | Max coverage z |
|---|---:|---:|---:|---:|
| latest_only_query | false | 0.656433 | 2.64733 | 1.67267 |
| map_posterior | true | 0 | 43.6352 | 50.9118 |
| tempered_likelihood | true | 5.59484e-15 | 8.39841 | 5.16281 |

All preregistered gates passed: `true`.
Post-result integrity audit: `pass`.
