# V37 results: candidate-conditioned semantic invariance

Decision: `semantic_invariance_no_material_gain_pivot_parser_or_grounder`.

V37 is a development-only semantic-interface result. It does not use V32 calibration/evaluation, V28, adapter training, or an end-to-end relational suite.

## Fresh validation

| Metric | Selected interface | Frozen V36 interface |
|---|---:|---:|
| Lexical sign | 0.786 | 0.792 |
| Outer operation | 0.978 | 0.931 |
| Compiled truth | 0.831 | 0.772 |
| Worst operation | 0.931 | 0.861 |
| Worst surface family truth | 0.556 | 0.472 |
| Distractor truth | 0.780 | 0.740 |
| Negative-composition truth | 0.944 | 0.833 |

## Selection and interpretation

Fit-only sign selection: `direct_hidden_ridge` with alpha `10.0`.

Fit-only operation selection: `direct_hidden_ridge` with alpha `1000.0`.

Compiled-truth gain over the untouched V36 interface: +0.058. All registered gates passed: `false`.

Post-result integrity audit: `pass`.
