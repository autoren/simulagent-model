# V106 Open-World Development Benchmark Result

## Outcome

After one zero-access bookkeeping repair (the design audit expected 14 future model gates although 15
were frozen), V106 passed every scientific pipeline gate. The repair changed no split, baseline,
metric, cost, model condition, or threshold. It was frozen before any development language was read.

The hash-only development split contains 128 calibration and 128 evaluation records, exactly 32 per
class in each subset. Retrieval trained on 2,149 MASSIVE training utterances spanning all 12 declared
intents. Calibration selected a known threshold of `0.75` and an unsupported threshold of `0.50` from
124 frozen threshold pairs.

Development-evaluation results:

| Baseline | Exact decision | Status macro F1 | Known intent exact | Novel scenario exact | False-known acceptance | Mean regret |
|---|---:|---:|---:|---:|---:|---:|
| Complete safe enumeration / abstain | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.125 |
| Ask always | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.125 |
| Identifier grammar | 0.453 | 0.393 | 0.406 | 0.000 | 0.250 | 3.781 |
| Character n-gram retrieval | 0.508 | 0.539 | 0.359 | 0.563 | 0.063 | 2.117 |
| Oracle | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 |

The grammar was strong only on its intended surface case: 81.25% exact for familiar known requests and
100% for the withheld unsupported scenario, but 0% on unfamiliar known and novel-valid records.
Retrieval generalized more broadly: status recall was 35.94% for known, 65.62% for novel, and 75% for
unsupported, with 56.25% exact novel-scenario routing. Its broader predictions nevertheless incurred
more costly errors than cautious abstention, so ask-always is the best frozen non-oracle baseline by
mean regret (`1.125`).

This is an important decision-level distinction: 50.78% exact classification did not make retrieval the
best policy under asymmetric error costs. The future model must achieve mean regret at most `1.375`—the
best deterministic regret plus the frozen `0.25` margin—in addition to the separately frozen semantic,
calibration, selective-risk, and controlled-abstention gates.

## Boundary and next step

The source archive and selected development file were each read automatically once. No utterance was
manually inspected. The protected test was not opened, and there were zero model loads, generations,
API calls, training runs, service calls, or side effects.

Freeze V106. The next authorized stage is a separate implementation audit followed by exactly one
development-only run of the pinned local Qwen3.8-27B 4-bit model on the 128 evaluation records and 64
missing-observation controls. It remains a shadow proposer with no pruning, belief, action, tool, or
execution authority. The protected test remains sealed unless that model passes every development gate.
