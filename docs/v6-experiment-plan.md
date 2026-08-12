# V6 shortcut-resistant corpus and mechanic-transfer plan

## Scientific question

V5 reached 96.15% balanced accuracy inside its generated development distribution but fell to
49.58% on a locked shortcut challenge. V6 asks whether training the same smallest frozen
representation on paired surface views and evidence interventions produces a transferable linear
boundary on a genuinely new mechanic. LoRA remains ineligible unless every preregistered gate
passes.

## Corpus and split contract

- Development mechanics: short-start relock and power-trip, seeds 9101–9106.
- Untouched mechanic: mirror rejection, seeds 9301–9303. This simulator mechanic was implemented
  for V6 and is absent from every development split and prior dataset.
- Each base input has canonical, entity-renamed, and paraphrased views with one shared binary label
  and an explicit same-label invariance group.
- The model-facing target contains only `ambiguous` and the surface-invariance relation. Exact
  outcome counts, possible transitions, empirical support, oracle traces, and privileged dynamics
  are absent.
- Connected context/evidence components are split before surface expansion. No context, prompt, or
  evidence group crosses train/calibration/holdout.

The final corpus has 143/49/76 base records and 41/14/39 context groups for
train/calibration/holdout. Ambiguity rates are 48.25%, 55.10%, and 50.00%. Development contains 18
training and 8 calibration evidence-intervention groups spanning both development templates.
After aggregating observationally identical hidden worlds, these development interventions are
label-preserving and therefore supervise invariance rather than a label flip. The untouched
mirror-rejection holdout contains 16 evidence groups, including four label-changing groups; their
score direction is a preregistered transfer diagnostic.

Leakage audit: zero prompt or scenario overlap with V4 development and the V5 challenge; zero
cross-split contexts or prompts; zero privileged target fields; V3 test records read: 0.

## Fixed baseline

The protocol fixes Qwen3.5-0.8B-4bit, full input, layer 6 mean pooling, a class-balanced float32
SAGA logistic head with `C=10` and seed 0, and complete-triplet training. No layer, pooling rule,
regularization value, model size, or input variant is selected in V6. The decision threshold is
fit once on canonical calibration records. Development feature extraction reads only train and
calibration. The resulting probe and all development artifacts must be hash-locked before the one
permitted mirror-rejection evaluation.

Protocol lock SHA-256:
`9664acde7fa491b5d22c48c2fc4c2c1266c283472ffdd65eb6d4a9568c748164`.

Dataset SHA-256:
`1fd29263b8aa262b2153aacb8d753fc942b43b1f6a869ec6c215c8690a61186d`.

## Preregistered LoRA gates

All gates must pass:

1. Canonical development calibration balanced accuracy is at least 75%.
2. Untouched-mechanic canonical balanced accuracy is at least 70%.
3. The context-group bootstrap lower bound for holdout balanced accuracy is at least 55%.
4. Entity-renamed and paraphrased holdout balanced accuracy are each at least 65%.
5. Canonical prediction agreement with each transformed surface is at least 85%.
6. Complete surface-triplet accuracy is at least 60%.
7. Holdout canonical balanced accuracy improves by at least 15 points over the locked V5 challenge
   reference of 49.58%.
8. Ambiguous scores exceed identifiable scores in at least 75% of label-changing holdout evidence
   comparisons.

Failure of any gate is a no-go for LoRA on this corpus. No threshold, record selection rule, gate,
or transformation may change after feature extraction begins.
