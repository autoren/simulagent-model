# V147 closed-alternative scoring results

## Verdict

V147 is a clean negative development result. Exact likelihood scoring over all 14 registered certificate alternatives removed free-form generation and almost eliminated structural failure, but it did not make the pinned local model reliably identify ambiguity or the catalog boundary. The branch fails its preregistered gates and closes without rescoring, prompt or alias changes, threshold fitting, a rerun, or test use.

The result is project-authored synthetic development evidence only. It is not V146 test evidence, external transfer, calibrated probability, unrestricted open-world understanding, authority, action, or execution evidence.

## Execution integrity

The prospectively frozen run completed all 144 V146 development fixtures and exactly 2,016 fixed candidate-sequence scores. It used one tokenizer load and one pinned local Qwen3.8-27B 4-bit model load. It produced zero generated tokens, retries, test scores, API calls, training runs, service calls, side effects, or executions. No prompt, response, reasoning trace, final text, or conversation was persisted in a scoring record.

The run took 1,535.67 seconds and peaked at 17,141,042,608 active bytes. All access gates passed. One fixture had an exact maximum-score tie and failed closed to `A00`; the remaining 143 had a unique registered maximum.

The V146 test partition received zero scores. Because two of its rows were displayed during pre-preregistration implementation inspection, the entire partition is retired as future evidence. Neither this result nor any successor may present it as blind test evidence.

## Main measurements

| Measurement | Result | Gate | Pass |
|---|---:|---:|---:|
| Certificate structural validity | 99.31% | 100% | No |
| Deterministic final-output validity | 100% | 100% | Yes |
| Overall final exact accuracy | 77.78% | at least 95% | No |
| Compatible-set exact accuracy | 74.31% | at least 95% | No |
| True-option retention | 88.89% | at least 99% | No |
| Ambiguity sensitivity | 50.00% | at least 98% | No |
| Decidable specificity | 82.50% | at least 98.5% | No |
| Conditional proposal correctness | 100% | at least 98.5% | Yes |
| Full six-stage group accuracy | 16.67% | at least 80% | No |
| False-known rate on non-known truths | 15.28% | at most 5% | No |
| Sequential mean decision cost | 2.6521 | at most 0.7 | No |
| Improvement over no query | 0.2750 | at least 0.25 | Yes |
| False-known on right-hand sequential truths | 45.83% | at most 5% | No |
| Safe non-known on right-hand sequential truths | 54.17% | at least 95% | No |
| Authoritative true-hypothesis retention | 100% | 100% | Yes |
| Actual execution count | 0 | 0 | Yes |

Language-class final accuracy was 100% on familiar known requests, 95.83% on both unfamiliar-known and clarified-known requests, 43.75% on novel-valid requests, 62.50% on clarified novel-valid requests, 87.50% on unsupported requests, 62.50% on clarified unsupported requests, and 54.17% on insufficient-evidence requests.

The most direct unsafe pattern was ambiguity collapse: of 24 `A00` truths, 13 remained `A00`, while 11 were converted to known choices (`K21` four times, `K22` twice, and `K23` five times). Conversely, every valid sufficient certificate was conditionally correct. The system therefore performed well after it selected the right singleton form, but the model was not reliable at deciding when singleton evidence existed.

## Score diagnostics

Softmax over the 14 sequence scores had a descriptive ten-bin ECE of 0.0766 and multiclass Brier score of 0.3521. These are candidate-relative diagnostics, not calibrated probabilities.

At a relative-score threshold of 0.9, coverage was only 39.58% and selective accuracy was 96.49%. That is insufficient to establish the required high-reliability open-world policy, and the threshold was not prospectively authorized as a decision rule. V147 therefore does not support post-hoc threshold selection.

## Interpretation

V144 showed that free-form certificate generation had a serious completion problem: 24 reasoning traces did not close. V147 removed that problem almost completely, increasing structural validity from 83.33% to 99.31%. Nevertheless, final exact accuracy fell from 97.22% to 77.78%, false-known errors rose from 0% to 15.28%, and sequential cost rose from 0.3417 to 2.6521.

The comparison should not be read as a pure method ranking because V144 and V147 use different fresh controlled populations. It does establish the intended mechanism boundary: generation failure was not the only obstacle. When forced to rank a complete closed list, the model strongly recognized familiar capabilities but frequently treated semantic resemblance as catalog membership. Fixed alternatives make every answer expressible; they do not supply the missing evidence that distinguishes a known capability from a nearby novel or unsupported one.

## Decision and next boundary

Freeze the decision:

`closed_alternative_scoring_fails_development_gates_close_without_test_or_tuning`

V147 authorizes no test run, rescoring, rerun, prompt or alias modification, confidence-threshold fitting, API condition, training, induction, authority, action, or execution. The next direction must be separately justified model-free. The most defensible candidate is a prospective selective-evidence architecture in which the LLM remains only a recall-oriented proposer, deterministic typed evidence or clarification is required before accepting catalog membership, and any calibration rule is designed before a wholly new blind population is authored. V147 scores may not be mined to construct that rule.
