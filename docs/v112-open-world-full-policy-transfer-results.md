# V112/V112r1 Fresh Full-Policy Transfer Result

## Execution and recovery

V112 prospectively selected 192 previously unused, externally authored MASSIVE validation records: 48
each for familiar known, unfamiliar known, novel-valid, and unsupported requests. Selection occurred before
language extraction and had zero identifier overlap with either the earlier V101 development set or its
protected test. The pinned Qwen3.8-27B model completed all 192 observed and 48 language-free control
generations in 37.7 minutes, with one model load, no retries, and 17.01 GB peak active memory.

The original runner encountered a mechanical `TypeError` during aggregation after all 240 fixtures had
been durably written: the ask-always helper's unused record argument was omitted. V112r1 froze hashes for
every preserved fixture, passed the record argument, and reran aggregation only. It performed zero new
model loads or generations and changed no population, language, response, policy, threshold, confidence,
metric, gate, or decision rule.

## Novelty-evidence result

The V111 signal transferred on fresh language. A typed LLM `ABSTAIN` was treated as evidence for a novel
candidate, while the actual shadow decision remained `ASK_FOR_CLARIFICATION`. It achieved:

- novelty precision: `0.7021` (33 true positives among 47 signals);
- novelty recall: `0.6875` (33 of 48 novel-valid requests);
- novelty F1: `0.6947`;
- non-novel false-positive rate: `0.0972` (14 of 144);
- novelty-probability ECE: `0.0494`;
- novelty-probability Brier score: `0.1237`.

All four preregistered novelty-evidence gates passed. This supports a precise claim: in this controlled
open-set transfer, model abstention contains calibrated evidence that a request may express valid novelty.
It does not prove that abstention means novelty in unrestricted language, nor does it identify or authorize
a new schema.

## Full-policy result

The frozen policy accepted a known shadow decision only when the LLM's exact intent agreed with deterministic
nearest-intent retrieval, accepted typed unsupported proposals as shadow rejections, and asked otherwise.
Compared with the direct LLM, it improved mean regret from `1.1276` to `0.8047`, better than ask-always's
`1.125`; false-known acceptance fell from `0.1458` to `0.0625`; and confidence ECE fell from `0.2616` to
`0.0235`. Top-80% error was `0.2208`.

It retained 100% of safe hypotheses, abstained on all 48 missing-observation controls, achieved 93.75%
unsupported recall with 95.74% precision, and made zero actual executions. Its observed exact decision
accuracy was 62.5% and status macro F1 was 59.83%. Novel exact-scenario accuracy remained zero by design:
at this pre-induction stage, novelty evidence asks rather than inventing a capability.

The full policy failed two of seventeen noncompensatory gates:

- known exact-intent accuracy was `0.78125`, below the required `0.80` by two correct records;
- top-confidence-80% error was `0.2208`, above the allowed `0.20` by roughly three retained errors.

Every other quality and access gate passed. The overall policy therefore does not qualify, even though its
novelty detector and expected decision utility transferred.

## Decision boundary

Freeze V112/V112r1 as a positive fresh novelty-evidence result and a negative full-policy result. Do not
lower thresholds, alter the validation rule, open the protected test, or proceed to schema induction. The
next policy must be preregistered and evaluated on another genuinely new population; it should address
known-request over-abstention and selective ranking without sacrificing the transferred novelty signal,
false-known safety, calibration, or regret.

No protected-test language was read. No API, adapter training, real service, action authority, capability
creation, tool execution, or external side effect occurred.
