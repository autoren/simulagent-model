# V25 results: explicit truth-hypothesis compatibility

Decision: `explicit_truth_hypotheses_insufficient_no_lora`.

V25 is an exposed-data truth-only experiment. It copied every V24 candidate assignment
unchanged, extracted three explicit assessment hypotheses per pair, and fitted one fixed
binary compatibility head. It is not a holdout or final result.

## Grounding

| Split | Atom assignment | Relation order | Truth | Exact scene |
|---|---:|---:|---:|---:|
| Fit | 0.984 | 0.968 | 0.793 | 0.101 |
| Calibration | 0.970 | 0.940 | 0.777 | 0.095 |
| Evaluation | 0.952 | 0.907 | 0.704 | 0.047 |

V24 evaluation truth was 0.878 and exact-scene accuracy was 0.203;
V25 reaches 0.704 and 0.047, respectively.

## Four-way integration

| Support graph | Query graph | Transition-set exact | Target retention | Empty version space |
|---|---|---:|---:|---:|
| oracle | oracle | 1.000 | 1.000 | 0.000 |
| frozen | oracle | 0.128 | 0.167 | 0.500 |
| oracle | frozen | 0.571 | 1.000 | 0.000 |
| frozen | frozen | 0.147 | 0.167 | 0.500 |

## Interpretation

Explicit entailment, contradiction, and unresolved hypotheses do not repair held-out truth semantics.

Calibration selected nothing. No model weights, matcher, proposal graph, ontology, DSL, or
executor changed, and no fresh benchmark was opened.

Post-result integrity audit: `pass`.
