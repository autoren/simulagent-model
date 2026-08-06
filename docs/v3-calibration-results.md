# V3 outcome-count calibration results

Checkpoint selection used the complete fixed validation split and constrained next-token
scores over digits 1 through 5. No test metrics were used.

| Seed | Selected step | Exact count | Balanced ID | Ambiguity F1 | Prediction counts |
| ---: | ---: | ---: | ---: | ---: | --- |
| 0 | 100 | 59.09% | 50.00% | 0.00% | {"1": 154} |
| 1 | 100 | 59.09% | 50.00% | 0.00% | {"1": 154} |
| 2 | 100 | 59.09% | 50.00% | 0.00% | {"1": 154} |

**Calibration gate: FAIL**

The gate was fixed before training: every seed must beat 50% balanced ID, at least
two of three and the mean must reach 55%, the seed range must be at most ten points,
and every selected checkpoint must predict both identifiable and ambiguous cases.
