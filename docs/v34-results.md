# V34 results: operation-focused frozen interface

Decision: `operation_interface_qualified_continue_binding_and_assembly`.

V34 is a fit/calibration-only representation diagnostic. It does not reuse V32 evaluation, open a fresh suite, or authorize V28.

## Operation classification

| Method | Fit | Calibration | Worst calibration operation | Oracle-sign compiled truth |
|---|---:|---:|---:|---:|
| Legacy hidden ridge | 1.000 | 0.621 | 0.000 | 0.659 |
| Focused hidden ridge | 1.000 | 0.997 | 0.981 | 0.997 |
| Native-logit ridge | 0.966 | 0.964 | 0.904 | 0.986 |
| Native argmax | 0.468 | 0.497 | 0.058 | 0.659 |

## Interpretation

Fit-only cross-validation selected `semanticHiddenRidge`. Its calibration operation gain over the legacy hidden-state ridge baseline is +0.376.

All registered qualification gates passed: `true`.

Post-result integrity audit: `pass`.
