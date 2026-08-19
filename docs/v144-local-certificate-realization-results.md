# V144 pinned local certificate development realization results

## Outcome

V144 completed the exact preregistered development realization but did not qualify:

```text
local_certificate_realization_fails_development_gates_close_without_test_or_tuning
```

The run generated one thinking-enabled response for each of the 144 locked V142 development fixtures. It used one load of the pinned local Qwen3.8-27B 4-bit model, made zero test-fixture generations, retries, API calls, service calls, training runs, or executions, and persisted no raw response, final text, or thinking trace.

## Main finding

The certificate interface exposed a sharp separation between semantic proposals that completed and ambiguity cases that did not complete.

- All 116 valid sufficient certificates proposed the correct choice.
- All 4 valid insufficient certificates were correct.
- However, 24 responses reached the 1,024-token ceiling without closing the prompt-opened thinking trace.
- Twenty of those 24 incomplete responses were ambiguous fixtures.
- The other four incomplete responses had `K12` truth: two known-familiar and two known-clarified cases.

Consequently, observable certificate-level ambiguity sensitivity was only `4/24 = 16.67%`. The deterministic finalizer safely mapped every incomplete ambiguous response to `A00`, so final ambiguous decisions were all correct. That safety result belongs to the controller, not to successful model certificate realization.

The conditional 100% proposal accuracy is also selection-conditioned: it proves that every completed sufficient certificate was correct, but it does not establish what the 24 incomplete traces would eventually have produced.

## Preregistered metrics

| Metric | Result | Gate | Pass |
|---|---:|---:|:---:|
| Certificate structural validity | 83.33% | at least 95% | No |
| Deterministic final-output validity | 100% | exactly 100% | Yes |
| Overall final exact accuracy | 97.22% | at least 95% | Yes |
| Every language-class final accuracy | minimum 91.67% | at least 95% | No |
| Compatible-set exact accuracy | 83.33% | at least 95% | No |
| Certificate true-option retention | 83.33% | at least 99% | No |
| Ambiguity sensitivity | 16.67% | at least 98% | No |
| Decidable specificity | 96.67% | at least 98.5% | No |
| Conditional proposal correctness | 100% | at least 98.5% | Yes |
| Full six-stage group accuracy | 87.5% | at least 80% | Yes |
| False-known rate on non-known truths | 0% | at most 5% | Yes |
| Candidate-attracted fraction of errors | 0% | at most 50% | Yes |
| Sequential mean decision cost | 0.3417 | at most 0.7 | Yes |
| Sequential improvement over no query | 0.8250 | at least 0.25 | Yes |
| Sequential false-known on right truth | 0% | at most 5% | Yes |
| Sequential safe non-known on right truth | 100% | at least 95% | Yes |
| Thinking-trace presence | 100% | at least 95% | Yes |
| Authoritative true-hypothesis retention | 100% | exactly 100% | Yes |
| Actual execution count | 0 | 0 | Yes |

The two language classes below their gate were `known_familiar` and `known_clarified`, each at 91.67%. All unfamiliar-known, novel-valid, unsupported, novel-valid-clarified, and unsupported-clarified final accuracies were 100%. The `insufficient_evidence` final accuracy was also 100%, but its certificate validity and compatible-set accuracy were each only 16.67% because deterministic fallback—not a valid model certificate—produced 20 of its 24 safe final abstentions.

## Sequential result

The fail-closed controller still produced a useful decision policy:

- every ambiguous initial decision queried;
- mean sequential cost was 0.3417 instead of 1.1667 without querying;
- overall improvement was 0.8250;
- every right-side decision avoided a false-known action;
- the worst family improvement was 0.45.

This is evidence that deterministic fallback plus clarification can be safe and valuable on this synthetic development population. It is not evidence that the LLM reliably supplied the intended evidence certificate.

## Interpretation and boundary

V144 rejects the current local certificate realization at the frozen 1,024-token ceiling. It does not justify claiming that the task is semantically impossible for the model, because completed certificates were exceptionally accurate. It also does not justify a larger ceiling or retry: those would be new experimental conditions selected after observing this outcome.

The V142 test split remains sealed and unauthorized. The branch closes without prompt tuning, response mining, retries, rerunning V144, increasing the token budget, changing models, or moving to an API. Capability induction, model authority, action, and execution remain closed.
