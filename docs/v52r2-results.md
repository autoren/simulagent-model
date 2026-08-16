# V52r2 results: final-joint Decimal normalization repair

Decision: `authorize_continuous_parameter_smc_squared_preregistration_with_pmcmc_reference`.

V52r2 reruns the unchanged sealed V52 populations with only final joint/configuration assembly moved into the preregistered 100-digit Decimal context. Particle filtering, likelihoods, resampling paths, budgets, seeds, exact oracle, metrics, and gates are unchanged.

| Metric | V52r2 |
|---|---:|
| SBC normalization rate | 1 |
| Scale normalization rate | 1 |
| Primary mean joint-belief TV | 0.000596765 |
| Primary mean query-program TV | 0.000258424 |
| Primary mean log-evidence error | 0.00343789 |
| Minimum SBC chi-square p-value | 0.0828219 |
| Maximum source-vs-repair exact-metric delta | 0 |
| Unintended stream collisions | 0 |

All unchanged preregistered gates passed: `true`.
Post-result repair-boundary audit: `pass`.
