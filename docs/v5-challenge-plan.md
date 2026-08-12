# V5 shortcut-resistant frozen-probe challenge plan

## Locked method

The challenge uses the already selected Qwen3.5-0.8B frozen probe without retraining or
recalibration: full input, layer 6 mean pooling, float32 SAGA logistic head, `C=10`, seed 0, and
threshold `-1.1556417346000671`. `configs/v5-frozen-probe-lock.json` hashes the exact probe,
source result, feature metadata, system prompt, and V4 dataset manifest. The challenge evaluator
may run once and must refuse to overwrite its result.

## New source worlds

The holdout is generated from Simulagent commit `d42e0266b59f60af6c7550ab69eb9f3a1c77a18c`
using fixed seeds 8101–8103. Neither generated family appeared in V4:

- `powertrip`: a held-out delayed generator reversal mechanic.
- `relockshort`: a short-start structural variant of the known relock family.

Each mechanic contains forced, announced, upstream-announced, consequence-announced,
procedure-announced, and unobservable variants with matched trap/control worlds. All ambiguous
records are retained and paired with an identifiable action from the same observation context.
The challenge also retains simulator-derived evidence-contrast groups when structurally matched
contexts change identifiability across evidence variants.

Every selected base record has three input surfaces: canonical, entity-renamed, and paraphrased.
They share one binary label and one bootstrap context group. Surface copies are never counted as
independent contexts.

## Pre-score audit

- 120 base records and 360 surface records.
- 63 original context groups.
- 57 ambiguous and 63 identifiable base records.
- 74 short-start relock and 46 power-trip base records; both mechanics contain both classes.
- Two evidence-contrast groups. This small axis is diagnostic and cannot establish broad
  evidence-rung generalization by itself.
- Zero exact prompt overlap and zero source-scenario overlap with V4 development.
- V3 test records read: 0.
- Frozen challenge dataset SHA-256: `ddf04fcf163d05db68d8a72d13094d958cccc22c8c6bf6a4e6df12e04e6778ec`.

## Preregistered gates

The frozen challenge passes only if all conditions hold:

1. Canonical balanced accuracy is at least 75%.
2. Each mechanic reaches at least 65% canonical balanced accuracy.
3. Entity-renamed and paraphrased balanced accuracy each reach at least 70%.
4. Prediction agreement between canonical and each transformed surface is at least 85%.
5. Across canonical evidence contrasts, ambiguous scores exceed identifiable scores in at least
   75% of cross-label comparisons.

Balanced accuracy and AUC are also reported with context-group bootstrap intervals. Error
concentration, calibration threshold transfer, score shifts, complete surface-triplet accuracy,
and complete evidence-group accuracy are diagnostic. No threshold, layer, pooling rule,
regularization value, transformation, record selection rule, or gate may change after scoring.
