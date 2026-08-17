# V72 shared-support active-sensing oracle preregistration

V72 begins a new program after the frozen negative V71 result. It does not modify V71, open V71's five protected models, or reinterpret an engineered success as external evidence. Its first purpose is narrower: prove that the existing exact joint-belief planner can represent a shared-support sensor-codebook problem in which information is genuinely control-relevant.

The positive oracle has two independent hidden variables: machine condition `A/B` and codebook `canonical/reversed`. `calibrate` reveals the codebook reference while preserving condition; `inspect` then reveals condition through that codebook; `repair_A/B` succeeds only for its matching condition. Calibration after inspection is deliberately uninformative, so the useful order is `calibrate → inspect → repair`. Both codebooks assign positive probability to both labels everywhere, eliminating fallback by construction.

At horizon three, the exact planner must uniquely calibrate first, inspect after either calibration label, and select both repair actions across reachable two-observation histories. A MAP point-model planner must instead inspect first because it treats its selected codebook as known. Exact-minus-MAP and exact-minus-posterior-sampling normalized regret must each reach `0.05`.

The negative-control oracle retains identical dynamics and sensors but gives `repair_A` a condition-independent dominant reward. Exact and point controls must agree and have regret no larger than `1e-12`. This control prevents a test harness that reports an advantage merely whenever latent uncertainty exists.

Oracle evaluation is an implementation test, not scientific evidence. Passing authorizes only metadata-level discovery of fresh external active-sensing environments. Candidate policies, values, regrets, optimal actions, or EIG remain forbidden until a separate source census, structural partition, and evaluator lock exist.
