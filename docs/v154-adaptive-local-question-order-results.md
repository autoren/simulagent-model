# V154 adaptive local question-order results

## Verdict

V154 is a clean negative development qualification with a useful model-usage result. Neither the primary direct condition nor the prospectively triggered bounded-low-reasoning challenger passed every question-order gate. The sealed V152 evaluation split must remain unopened, and both conditions close without retry, rerun, reprompting, budget changes, threshold fitting, calibration, or tuning.

Direct decoding was clearly better. All 96 outputs were structurally valid, 80 placed the discriminating question first and 16 placed it second. Query top-1 accuracy was 83.333%, MRR 0.9167, mean correct-query rank 1.1667, and sequential mean cost 0.34. Cost improvement over safe no-query abstention was 0.66. Direct passed structural validity, mean-rank, cost, improvement, final-safety, fail-closed, retention, no-candidate, and zero-execution gates, but missed the preregistered 90% top-1 and 0.95 MRR thresholds.

Because direct failed, the locked adaptive rule triggered bounded low reasoning. That condition performed worse: 78 fixtures ranked the correct question first, 14 second, and 4 fifth. Top-1 accuracy was 81.25%, MRR 0.8938, mean rank 1.3125, sequential cost 0.375, and improvement over no-query 0.625. Five final responses were invalid JSON, giving 94.792% structural validity. It failed structural validity, top-1, MRR, mean-rank, and cost-sharpness gates.

Both conditions remained architecturally safe. Every one of the 240 condition-specific sequential episodes ended in the exact state after the trusted closed answer. All 46 irrelevant intermediate answers failed closed to `A00`, authoritative hypothesis retention was 100%, candidate-state proposal fields were absent, and execution was zero.

Freeze the decision:

`local_question_order_conditions_fail_development_gates_close_without_evaluation_or_tuning`

## What the reasoning intervention established

The official pinned Qwen3.8 template defaults thinking to `xhigh`; V154 explicitly requested `low`. Because that template setting supplies only a brevity instruction, V154 also imposed a hard 48-token reasoning phase and a separate reserved 64-token final continuation.

The mechanical intervention worked as a completion safeguard but did not improve the task. All 96 reasoning phases consumed the full 48-token budget and none produced a natural `</think>` close. The runner therefore forced the close in every case. No final continuation reached its 64-token ceiling, so the five invalid JSON outputs were not caused by final-token truncation. Direct outputs averaged 31.56 tokens with zero limit hits and zero invalid records; bounded low reasoning averaged 48 reasoning tokens plus 29.26 final tokens and introduced five invalid records.

The proper interpretation is:

> Reserving a final-answer budget prevents an overlong reasoning trace from swallowing the output, but low reasoning does not improve—and here degrades—a small registered question-ordering task that direct decoding already handles efficiently.

This directly addresses the earlier Qwen overthinking concern. A reasoning budget is useful as a systems safeguard when thinking is enabled, but enabling thinking is not automatically beneficial. For this task, direct decoding should remain the preferred local configuration.

## Relation to V151r1

Removing state proposals and confidence improved the interface but did not cross the strict ranking gates. V151r1 direct query ranking had top-1 accuracy 83.333%, MRR 0.9141, mean rank 1.1875, and cost 0.3525. V154 direct produced the same top-1 rate, slightly higher MRR at 0.9167, slightly better mean rank at 1.1667, and lower cost at 0.34. The simplified prompt therefore delivered a small efficiency improvement and perfect syntax, but not the preregistered jump in first-question accuracy.

This is evidence that candidate-state generation was unnecessary overhead, not evidence that prompt simplification solves semantic question selection. The model still confuses which of two closely related registered questions is most discriminating on about one-sixth of requests.

## Access and recovery boundary

The study completed exactly 96 direct generations and, after direct failed, 96 bounded reasoning plus 96 bounded final generations. It used one tokenizer/model load, no retries, no closed-answer or evaluation generations, no manual raw-output inspection, no persisted raw prompts, reasoning traces, or final responses, no API or training, no services or side effects, and zero execution.

Only normalized rankings and resource/structural diagnostics were persisted. The complete run used 288 generation calls and stayed within every access gate.

## Claim boundary and next direction

V154 is project-authored synthetic development evidence. It is not evaluation evidence, external language transfer, calibrated confidence, human-answer evidence, or permission for capability induction, authority, action, or execution.

Do not reopen V154 by changing the prompt, reasoning effort, reasoning or final budget, parser, fallback, thresholds, quantization, model, API, or training condition. The V152 evaluation split remains sealed.

The current model has nevertheless demonstrated real heuristic value relative to the frozen baselines: direct cost 0.34 is far below source/random cost 1.05 and close to the oracle cost 0.3, while mistakes affect interaction cost rather than final correctness. The failed gates say that this heuristic is not reliable enough for the stronger preregistered ranking claim.

The next branch should therefore not search for another decoding tweak on the same language. A scientifically distinct successor should examine whether a deterministic or hybrid retrieval layer can select questions from explicit lexical/typed features, with the LLM retained only as a non-authoritative tie-breaker if model-free feasibility justifies it. Any such branch requires fresh development language and a prospective model-free protocol; evaluation, APIs, training, authority, action, and execution remain closed.
