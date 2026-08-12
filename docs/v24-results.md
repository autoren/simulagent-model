# V24 results: candidate-conditioned frozen relational grounding

Decision: `factor_truth_semantics_before_fresh_benchmark_no_lora`.

V24 is an exposed-data development experiment, not a holdout or final result. The model,
layer, ontology, candidate atoms, DSL, executor, proposal count, heads, and gates were frozen
before the single feature extraction and head fit.

## Proposal and extraction

The top-three-plus-hard proposal graph retained 0.979 of evaluation-support
gold edges and 0.975 of evaluation-query gold edges.
The frozen 4B model executed 13372 candidate-conditioned forwards with zero truncation.

## Grounding

| Split | Atom assignment | Relation order | Truth | Exact scene |
|---|---:|---:|---:|---:|
| Fit | 0.984 | 0.968 | 0.982 | 0.744 |
| Calibration | 0.970 | 0.940 | 0.975 | 0.667 |
| Evaluation | 0.952 | 0.907 | 0.878 | 0.203 |

## Four-way integration on evaluation episodes

| Support graph | Query graph | Transition-set exact | Target retention | Empty version space |
|---|---|---:|---:|---:|
| oracle | oracle | 1.000 | 1.000 | 0.000 |
| frozen | oracle | 0.667 | 0.583 | 0.250 |
| oracle | frozen | 0.731 | 1.000 | 0.000 |
| frozen | frozen | 0.526 | 0.583 | 0.250 |

## Interpretation

Candidate identity transfers, but held-out truth semantics remain below the registered gate.

Calibration was report-only and selected no model, feature, threshold, regularization, or
proposal policy. Regardless of the result, V24 does not itself support a final scientific claim.

Post-result integrity audit: `pass`.
