# V22 development results: typed relational oracle foundation

Decision: `authorize_relational_language_grounding_development`.

This is an open development result, not a sealed final evaluation. It authorizes work on
the relational language-grounding interface only; it does not authorize final data, new
model weights, or a matched neural challenger yet.

## Oracle result

- 24 mechanics and 312 relational queries;
- exact lifted schema recovery: 1.000;
- exact transition-set match: 1.000;
- identifiability accuracy: 1.000;
- permutation consistency: 1.000;
- distractor consistency: 1.000; and
- literal graph lookup transition-set match: 0.038.

## Relational families

| Family | Schemas recovered | Complete episodes | Queries | Exact |
|---|---:|---:|---:|---:|
| `unary_selection` | 6/6 | 6/6 | 78 | 1.000 |
| `relation_conditioned` | 6/6 | 6/6 | 78 | 1.000 |
| `two_hop_composition` | 6/6 | 6/6 | 78 | 1.000 |
| `existential_aggregation` | 6/6 | 6/6 | 78 | 1.000 |

## Search scaling

| Outcome bits | Candidates | Median support | Median search seconds | Maximum executions |
|---:|---:|---:|---:|---:|
| 1 | 24 | 2.0 | 0.002356 | 96 |
| 2 | 552 | 3.5 | 0.103954 | 3864 |

## Audit and firewall

- structural/metamorphic audit: pass;
- complete false facts are distinct from explicit unknown facts;
- fit/evaluation programs and registered structural query axes are disjoint;
- no target program or structured oracle-graph field appears in agent inputs;
- zero V21-final record or model-result reads; and
- zero model forwards, linear fits, adapter runs, or V22-final constructions.
