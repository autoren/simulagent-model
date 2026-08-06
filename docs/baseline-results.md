# Deterministic baseline results

Evaluated on all 952 full-corpus test transitions using 8,145 training transitions.

| Baseline | Exact match | Changed targets | Unchanged targets | Macro field accuracy |
| --- | ---: | ---: | ---: | ---: |
| No change | 45.17% | 0.00% | 88.84% | 87.44% |
| Action majority | 58.09% | 45.30% | 70.45% | 94.19% |
| Exact-prompt lookup | 84.56% | 69.44% | 99.17% | 98.11% |
| Nearest neighbour | 75.63% | 73.08% | 78.10% | 97.55% |

Every test agent prompt already occurs verbatim in training. Exact-prompt lookup therefore
measures input duplication rather than generalization and establishes the minimum baseline a
model must beat on this split. The no-change result shows why changed and unchanged targets
must always be reported separately.
