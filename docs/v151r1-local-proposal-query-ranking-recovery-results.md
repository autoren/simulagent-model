# V151r1 recovered local proposal and query-ranking results

## Verdict

V151r1 is a clean negative development qualification with a positive architectural
result. The recovered Qwen3.8-27B-4bit realization did not qualify as a sufficiently
reliable semantic proposal or query-ranking component, so the V149 evaluation split
must remain unopened and this realization must close without retry, rerun, reprompting,
threshold fitting, or tuning.

The trusted closed-answer architecture nevertheless behaved exactly as intended.
Across 120 sequential episodes, final exact accuracy after a trusted answer was 100%,
false-known outcomes were 0%, all 21 irrelevant intermediate answers failed closed,
and authoritative hypothesis retention was 100%. The non-authoritative model proposals
reduced mean decision cost to 0.3525, below the preregistered maximum of 0.45, and
improved over safe no-query abstention by 0.6475, above the required 0.55.

Those safety and cost results do not rescue the model-utility claim. Compatible-state
retention was 84.375% rather than the required 97%; exact compatible-set accuracy was
57.292% rather than 90%; ambiguity sensitivity was 50% rather than 95%; query top-1
accuracy was 83.333% rather than 90%; and query mean reciprocal rank was 0.9141 rather
than 0.95. Decidable specificity and top-1 state accuracy also failed their 95% gates.
The model's self-reported confidence was strongly overconfident and remains diagnostic
only: its 10-bin ECE was 0.4604 and its Brier score was 0.4494.

Freeze the decision:

`recovered_local_proposal_query_ranking_fails_development_gates_close_without_evaluation_or_tuning`

## Recovery integrity

The original prospectively locked V151 process was externally interrupted after 59
generation attempts. Fifty-eight normalized, non-authoritative artifacts had been
durably written; the 59th attempt had no persisted response; and 37 fixtures had never
started. Before any semantic output inspection, V151r1 froze a no-retry recovery:

- retain the 58 durable artifacts byte-for-byte;
- represent the interrupted fixture by the registered invalid fail-closed proposal;
- generate each of the 37 never-started fixtures exactly once under the original model,
  prompt, decoding, validation, semantic metrics, and qualification gates;
- use one additional model and tokenizer load and no retries;
- never read the closed-answer or evaluation language with the model.

The completed census therefore contains exactly 58 retained outputs, one technical
fail-closed record, and 37 newly generated outputs. Accounting finished at 96 model
generation attempts, two model loads, two tokenizer loads, one generation attempt per
fixture, zero retries, zero API calls, zero evaluation or closed-answer generations,
zero manual raw-response inspection, zero persisted raw responses, zero training,
zero services, zero side effects, and zero execution.

The single interrupted case cannot explain the failed gates. Even if it were changed
post hoc to a perfect response, compatible-state retention could reach at most 85.417%,
exact compatible-set accuracy at most 58.333%, and query top-1 accuracy at most 84.375%.
All would remain far below their frozen thresholds. No such replacement is authorized;
these bounds only establish that the negative semantic result is robust to the technical
interruption.

## What the model did and did not contribute

The strongest positive finding is economic rather than authoritative. Query ranking was
often useful enough to lower interaction cost: mean correct-query rank was 1.1875, which
passed the maximum-rank gate, and the sequential cost and improvement gates both passed.
The model therefore supplied imperfect but sometimes useful hints to a trusted
controller.

The hints were not dependable enough to prune the state space or decide catalog
membership. The compatible state was retained on only half of the insufficient-evidence
fixtures, and exact candidate sets were never produced for the nominally novel-candidate
class. Eighty-four percent of semantic errors were attracted toward candidates, above
the permitted 50%, which is consistent with the project's recurring similarity-driven
membership failure. Familiar known requests had exact candidate sets, but their useful
question was ranked first only 66.667% of the time. The failure is thus not output syntax:
95 of 96 records were structurally valid, with the sole invalid record forced by the
technical recovery contract.

The proper interpretation is:

> A local LLM can reduce the expected number of trusted clarification questions while
> remaining safely non-authoritative, but this realization cannot reliably identify the
> compatible open-world state set or consistently choose the best first question.

## Claim boundary and next direction

V151r1 is project-authored synthetic development evidence. It is not evaluation
evidence, external transfer, human-language deployment evidence, calibrated confidence,
or evidence that the model may define capabilities, satisfy witnesses, authorize known
membership, act, or execute.

The V149 evaluation split remains unopened. No V151/V151r1 retry, rerun, prompt change,
reasoning-effort change, threshold fit, calibration fit, API comparison, training,
induction, authority, action, or execution is authorized from this outcome.

The result closes the current model-utility realization. Any successor must be a
prospectively distinct scientific question using fresh development language or a frozen
mechanical intervention. A bounded low-reasoning challenger may later test whether
direct decoding caused avoidable proposal errors, but it must use a new lock and fresh
population; it must not reopen or tune against V151r1. The more important architectural
direction remains trusted typed clarification with a model used only to rank questions,
because that is the component whose mistakes degrade cost rather than safety.
