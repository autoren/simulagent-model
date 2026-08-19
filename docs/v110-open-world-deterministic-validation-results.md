# V110 Deterministic Novelty, Abstention, and LLM Validation Result

## Outcome

V110 completed the prospectively hash-split secondary development analysis and failed the primary
noncompensatory gate set. The 128 frozen V109 observed records were divided into 64 calibration and 64
evaluation records, exactly 16 per class in each subset. No new model output was generated.

The direct LLM reproduced V109's central tradeoff on evaluation: 71.88% exact decisions and 100% exact
known-intent accuracy, but zero novelty recall, 21.88% false-known acceptance, seven unsafe known shadow
proposals on novel or unsupported requests, ECE `0.2406`, and mean regret `1.375`.

Calibrated confidence abstention selected a `0.99` threshold from 13 frozen candidates. It reduced false-
known acceptance and unsafe known shadow proposals to zero, reduced ECE to `0.0313`, and achieved mean
regret `1.0391`, better than ask-always's `1.125`. The price was severe: decision coverage fell to 14.06%,
overall exact accuracy to 12.5%, known accuracy to 3.13%, and novelty recall remained zero. Confidence can
therefore suppress risky action but does not identify valid novelty.

Character retrieval selected known and unsupported thresholds of `0.8` and `0.3`. It and the fixed
deterministic novelty override reached 100% novelty recall with zero false-known acceptance, but novelty
precision was only 29.63%, known accuracy was 31.25%, unsupported recall was zero, ECE exceeded `0.23`,
and mean regret was `2.1875`. The rule moved from under-detecting novelty to over-detecting it.

The primary conservative LLM-plus-validation policy had the same evaluation outcome as retrieval. Under
the selected thresholds, retrieval classified enough records as novel that the rule's novelty-acceptance
branch dominated before exact agreement could help. It passed novelty recall, exact novel-scenario routing,
false-known acceptance, missing-evidence abstention, hypothesis retention, and zero-execution gates, but
failed novelty precision, known and unsupported accuracy, macro F1, calibration, selective risk, and regret.

No non-oracle policy jointly achieved useful novelty precision and recall. The oracle remained exact with
zero regret, confirming that the hidden typed ground truth and scoring path can represent the desired
behavior. The negative result is therefore about available evidence and decision rules, not an impossible
evaluation contract.

## Boundary and decision

V110 read only the frozen development artifact and official training archive automatically. It exposed no
source language or raw model response, read no protected-test language, loaded no model, and made no API,
training, service, tool, or external-effect call. All policies retained the complete safe hypothesis set,
and actual execution count was zero.

Freeze V110 as nonqualifying. Do not retune thresholds, switch the primary policy, open the protected test,
or proceed to typed capability induction or the richer sequential decision problem. The next justified
step is a preregistered, aggregate-only evidence-separability audit: measure whether deterministic features
not used by V110—such as per-intent retrieval score, top-two margin, scenario concentration, and LLM/
retrieval disagreement type—contain enough signal for any simple frozen gate to satisfy both novelty
precision and recall. If not, close this evidence interface and require a genuinely new contrastive or
multi-turn evidence source rather than another post-hoc threshold.
