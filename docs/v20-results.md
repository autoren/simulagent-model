# V20 results: calibrated grounding–program uncertainty

Decision: `preserve_only_negative_probabilistic_development_result`.

V20 reuses the saved V19 features and frozen deployment heads. It performs zero model
forward passes, feature extractions, linear fits, adapter runs, or final-suite reads.

## Calibration

| View | Unique calibration prompts | Threshold | Calibration coverage | Development coverage | Development mean set size |
|---|---:|---:|---:|---:|---:|
| `novel_ontology` | 24 | 0.2660 | 0.958 | 0.963 | 1.000 |
| `supported` | 8 | 0.0011 | 1.000 | 1.000 | 1.000 |

## Novel Ontology

| Condition | Episode macro | Complete | Target retained | Empty | Excess outcomes/query |
|---|---:|---:|---:|---:|---:|
| `hard_support_hard_query` | 0.740 | 28/40 | 0.750 | 0.225 | n/a |
| `oracle_support_probabilistic_query` | 0.919 | 29/40 | 1.000 | 0.000 | 0.090 |
| `probabilistic_support_oracle_query` | 0.734 | 28/40 | 0.750 | 0.225 | 0.629 |
| `probabilistic_support_probabilistic_query` | 0.740 | 28/40 | 0.750 | 0.225 | 0.608 |

## Supported

| Condition | Episode macro | Complete | Target retained | Empty | Excess outcomes/query |
|---|---:|---:|---:|---:|---:|
| `hard_support_hard_query` | 1.000 | 40/40 | 1.000 | 0.000 | n/a |
| `oracle_support_probabilistic_query` | 1.000 | 40/40 | 1.000 | 0.000 | 0.000 |
| `probabilistic_support_oracle_query` | 1.000 | 40/40 | 1.000 | 0.000 | 0.000 |
| `probabilistic_support_probabilistic_query` | 1.000 | 40/40 | 1.000 | 0.000 | 0.000 |

## Preregistered checks

- `novel_empty_posterior_bounded`: pass
- `novel_episode_macro_nonnegative_gain`: pass
- `novel_excess_outcomes_bounded`: fail
- `novel_target_retention_nonnegative_gain`: pass
- `supported_complete_episodes_preserved`: pass
- `supported_credible_target_retention`: pass
- `supported_episode_macro_preserved`: pass
- `supported_no_empty_posterior`: pass

The hard V19 result is unchanged. V20 cannot authorize LoRA and is eligible for the
sealed V21 suite only according to the preregistered supported-preservation decision.
