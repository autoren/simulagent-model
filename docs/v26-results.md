# V26 results: full-depth native truth decoder

Decision: `repair_exact_graph_or_symbolic_composition_no_lora`.

V26 is an exposed-data, zero-fit development evaluation. It kept every V24 candidate
assignment fixed and selected truth status only by float32 A/B/C decoder logits from the
frozen model's final layer.

## Grounding

| Split | Atom assignment | Relation order | Truth | Exact scene |
|---|---:|---:|---:|---:|
| Fit | 0.984 | 0.968 | 0.957 | 0.612 |
| Calibration | 0.970 | 0.940 | 0.960 | 0.571 |
| Evaluation | 0.952 | 0.907 | 0.938 | 0.464 |

V25 evaluation truth was 0.704; V26 reaches 0.938.

## Four-way integration

| Support graph | Query graph | Transition-set exact | Target retention | Empty version space |
|---|---|---:|---:|---:|
| oracle | oracle | 1.000 | 1.000 | 0.000 |
| frozen | oracle | 0.545 | 0.667 | 0.250 |
| oracle | frozen | 0.917 | 1.000 | 0.000 |
| frozen | frozen | 0.519 | 0.667 | 0.250 |

## Interpretation

Native truth semantics pass, but exact graph assembly or symbolic integration remains below gate.

No head, threshold, model weight, matcher, proposal, ontology, DSL, or executor changed.
Calibration selected nothing and no fresh benchmark was opened.

Post-result integrity audit: `pass`.
