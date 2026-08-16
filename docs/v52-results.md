# V52 results: Rao–Blackwellized particle filtering

Decision: `repair_particle_accuracy_calibration_degeneracy_or_stream_integrity`.

V52 tests bounded particle inference only for the hidden dynamic world and delayed-effect queue. The 48 finite program/probability hypotheses and every particle's local one-step stochastic branches remain exactly enumerated. This is non-final and does not test continuous parameters, active intervention selection, reward, planning, language, noisy sensors, or open ontologies.

| Metric | Value |
|---|---:|
| Primary particle budget | 509 |
| Exact-benchmark completion | 1 |
| Mean support-program TV | 0.000290005 |
| Mean query-program TV | 0.000258424 |
| Mean probability-marginal TV | 0.000215169 |
| Mean joint-belief TV | 0.000596765 |
| Mean suffix-predictive TV | 0.00035758 |
| Mean absolute log-evidence error | 0.00343789 |
| Minimum SBC chi-square p-value | 0.0828219 |
| Maximum absolute rank-bin z | 3.61478 |
| Maximum absolute coverage z | 2.0625 |
| Unintended stream collisions | 0 |
| Independent-repeat fingerprint collision rate | 0 |
| Scale-stress normalization rate | 0.708333 |

## Budget convergence

| Particles | Mean core TV | Repeat dispersion | Mean log-evidence error |
|---:|---:|---:|---:|
| 127 | 0.00107456 | 0.000356745 | 0.0133332 |
| 31 | 0.00550778 | 0.0016489 | 0.05747 |
| 509 | 0.000343589 | 0.000137101 | 0.00343789 |

All preregistered gates passed: `false`.
Post-result integrity audit: `pass`.
