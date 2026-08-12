# V5 frozen 0.8B representation-probe results

## Decision

The frozen Qwen3.5-0.8B representation gate passes decisively. A class-balanced float32 logistic head over the calibration-selected hidden representation reaches 96.15% mean validation balanced accuracy on full input and 91.64% after removing history and memories. The generative A/B vocabulary interface, not the absence of linearly accessible signal, was the immediate V4 bottleneck.

This is not yet evidence of semantic epistemic reasoning. The strong no-history result is compatible with static-field, action-template, or scenario-family shortcuts. The next scientific requirement is a newly generated untouched challenge holdout with entity renamings, paraphrases, evidence-rung minimal pairs, and held-out mechanics. LoRA behind the discriminative head is eligible, but should not be interpreted without that holdout.

## Results

| Input | Seed | Selected feature | C | Calibration balanced | Validation balanced | AUC | Group-bootstrap 95% interval |
| --- | ---: | --- | ---: | ---: | ---: | ---: | --- |
| full | 0 | `layer_06_mean` | 10 | 90.56% | 96.15% | 1.000 | 90.00%–100.00% |
| full | 1 | `layer_06_mean` | 10 | 90.56% | 96.15% | 1.000 | 89.86%–100.00% |
| full | 2 | `layer_06_mean` | 10 | 90.56% | 96.15% | 1.000 | 90.00%–100.00% |
| no_history | 0 | `layer_06_mean` | 1 | 82.57% | 91.64% | 0.980 | 78.29%–100.00% |
| no_history | 1 | `layer_06_mean` | 1 | 82.57% | 91.64% | 0.980 | 78.29%–100.00% |
| no_history | 2 | `layer_06_mean` | 1 | 82.57% | 91.64% | 0.980 | 79.14%–100.00% |

The three seeds exercise the stochastic optimization order of the float32 SAGA fits; they do not create three independent pretrained Qwen models.

Full input makes 7 errors across 2 of 19 validation contexts. The no-history variant makes 13 errors across 3 contexts, including 2 contexts for which every record is wrong. This concentration is why record-level performance alone is not enough to establish shortcut-resistant generalization.

## Firewall and limitations

- Features were extracted from 1,037 training, 181 calibration, and 154 validation records.
- Layer, pooling, regularization, and threshold were selected only on calibration.
- Validation contains 19 context groups and is used only for frozen evaluation.
- No prompt was truncated; source hidden states were bfloat16 and probe inputs/weights were float32.
- Validation errors are reported by context group because record-level accuracy can hide group concentration.
- V3 test records read: 0.
- The extraction path reads only `agent_input` plus labels needed for supervised fitting; metadata and hidden-state inputs exclude target outcomes, mechanic labels, empirical support, and scenario IDs.
- Because 0.8B already passes the representational gate, 4B and 9B frozen extraction is not required to answer the capacity question and is deferred.

## Subsequent locked challenge

The next shortcut-resistant evaluation was frozen before scoring and failed. On 120 new base cases
from 63 contexts, the unchanged probe reached 49.58% balanced accuracy and 0.457 AUC. The held-out
power-trip mechanic was a constant-identifiable 50.00%, while short-start relock reached 52.28%.
Entity-renamed and paraphrased surfaces were also near chance. See
`docs/v5-challenge-results.md`. The 96.15% result should therefore be interpreted as strong
within-distribution label signal, not a transferable epistemic representation.
