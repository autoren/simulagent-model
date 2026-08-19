# V114 Record-Disjoint Rescued-Policy Transfer Result

## Outcome

V114 is a clean negative transfer result. The frozen Qwen3.8-27B 4-bit condition completed all 240
generations exactly once: 192 new human-authored MASSIVE test-partition records and 48 missing-observation
controls. The population was selected before language extraction and had zero identifier overlap with V101,
including its sealed protected records, or V112. There was one model load, no retries, no API or training,
no manual language or raw-response inspection, no protected-language read, and zero execution, service
calls, or external effects.

The primary preregistered decision is
`novel_evidence_nontransfer_close_abstention_signal_beyond_V112`. Neither the frozen V112 policy nor the
V113-rescued policy passed all seventeen absolute gates. V114 therefore does not authorize protected-set
access, typed induction, richer sequential planning, another model, training, actions, or execution.

## Novel-capability evidence did not extend beyond V112

Typed `ABSTAIN` remained informative, but missed two noncompensatory thresholds by one false positive:

| Metric | V114 | Gate |
| --- | ---: | ---: |
| Precision | 68.75% | at least 70% |
| Recall | 68.75% | at least 50% |
| Non-novel false-positive rate | 10.42% | at most 10% |
| F1 | 68.75% | reported |
| ECE | 0.0526 | at most 0.15 |

The confusion counts were 33 true positives, 15 false positives, 15 false negatives, and 129 true
negatives. Reducing the false-positive count from 15 to 14 would have crossed both failed gates, but the
thresholds are frozen and may not be relaxed. V112 remains a valid positive result on its own fresh
validation population; V114 shows that the claim does not transfer unchanged to this second, record-
disjoint test-partition population.

## Neither full policy qualified

| Metric | Frozen V112 policy | V113-rescued policy | Gate |
| --- | ---: | ---: | ---: |
| Exact decisions | 58.33% | 59.38% | at least 60% |
| Known exact intent | 73.96% | 76.04% | at least 80% |
| Top-80% error | 27.27% | 25.97% | at most 20% |
| False-known acceptance | 9.38% | 10.42% | at most 10% |
| Mean regret | 0.9688 | 1.0026 | at most 1.125 |
| Unsupported precision | 97.62% | 97.62% | at least 80% |
| Unsupported recall | 85.42% | 85.42% | at least 80% |
| Confidence ECE | 0.0349 | 0.0362 | at most 0.15 |

Both policies retained the full safe hypothesis universe and executed nothing. Their mean regret remained
better than ask-always's 1.125, but favorable average cost cannot compensate for the failed novelty,
known-grounding, exact-decision, and selective-error gates.

## Paired rescue evidence was unfavorable

The same model response and retrieval result fed both policies. There were 22 eligible LLM/retrieval
disagreements, exceeding the preregistered minimum of eight, but the rule triggered only three times rather
than the required four. All three triggers were concentrated in the calendar scenario:

| Paired result | Count |
| --- | ---: |
| Wrong to correct | 2 |
| Correct to wrong | 0 |
| Wrong to wrong | 1 |
| Correct to correct | 0 among changed records |

The rule therefore had 66.67% rescue precision rather than the required 75%. It improved known accuracy by
2.08 percentage points and selective error by 1.30 points, but increased false-known acceptance by 1.04
points and mean regret by 0.0339. The harmful trigger changed an already-wrong novel-valid case into a more
costly false-known acceptance. Novel evidence itself remained exactly unchanged.

Because the upstream novelty gates failed, the formal rescue-mechanism status is
`not_interpretable`. Independently, the preregistered paired preservation gates for false-known acceptance
and regret failed, the minimum trigger opportunity failed, and rescue precision failed. The V113 rescue
rule should therefore not be retained as a supported repair or retuned on V114.

## Scientific interpretation

V114 narrows the result to a useful boundary:

> A typed local-model abstention is sometimes evidence of an undeclared valid intent, but it is not stable
> enough across two record-disjoint MASSIVE partitions to serve as the sole novelty discriminator. A small
> retrieval-agreement rescue can recover known requests while simultaneously making a novel request more
> costly, so static known-intent repair does not solve the open-set boundary.

This is controlled open-set intent transfer from one source distribution, not unrestricted open-world
understanding. The next branch must introduce genuinely new contrastive or multi-turn evidence, rather than
mine V114, relax its thresholds, open the protected set, scale the model, ensemble outputs, or begin schema
induction. Any successor remains shadow-only and keeps the LLM permanently non-authoritative.
