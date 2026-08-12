# V27 results: outcome-constrained support MAP

Decision: `support_map_improves_execution_continue_match_repair_no_lora`.

V27 is an exposed-data support-only experiment. It reused all V26 query predictions,
scored 1,103 remaining support proposal edges, and selected support graphs jointly with one
shared episode program under observed-transition consistency.

## Support and integration

Evaluation exact support graphs: 0.694.
Frozen-support/oracle-query execution changed from 0.545 to 0.744;
target retention changed from 0.667 to 0.833,
and empty version spaces changed from 0.250 to 0.000.

| Support graph | Query graph | Transition-set exact | Target retention | Empty |
|---|---|---:|---:|---:|
| oracle | oracle | 1.000 | 1.000 | 0.000 |
| frozen | oracle | 0.744 | 0.833 | 0.000 |
| oracle | frozen | 0.917 | 1.000 | 0.000 |
| frozen | frozen | 0.718 | 0.833 | 0.000 |

## Search diagnostics

Mean retained graphs per support scene: 511.8; target graph branch coverage: 0.986; episode fallback rate: 0.000.

No query prediction, model weight, head, threshold, ontology, DSL, or executor changed.
No fresh benchmark was constructed.

Post-result integrity audit: `pass`.
