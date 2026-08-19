# V107 Local Open-World Model Development Plan

V107 runs exactly one already cached model: Qwen3.8-27B 4-bit. It receives the frozen V105 visible typed
catalog and no demonstrations. The corpus consists of the 128 V106 development-evaluation utterances and
64 missing-observation controls; calibration and protected-test records are excluded. Missing controls
contain only the sentinel and never expose their source utterance.

Each response is validated by the exact V105 contract. Invalid output becomes zero-confidence abstention
without retry. The raw output is sealed for automatic verification and never manually inspected. Model
confidence is scored as emitted, with no post-hoc calibration. All classification and regret metrics are
computed on the 128 observed records; missing-observation abstention is reported separately.

The model is a shadow semantic proposer. Its output cannot remove any of the 17 safe hypotheses, alter
capability state or posterior, select an action, call a tool, or execute anything. All reported action
costs are counterfactual. Passing every development and access gate authorizes only a separate protected-
test preregistration; failure closes the model branch without opening the test.
