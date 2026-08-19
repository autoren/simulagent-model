# V132 Local Complete-answer Realization Results

## Outcome

V132 completed the exact locked condition and is a decisive negative realization result:

> `local_complete_answer_does_not_realize_V130_boundary_close_one_pass_local_realization_branch`

The pinned local Qwen3.8-27B 4-bit model produced 264 of 264 answers with one model load, one deterministic
generation per fixture, no retries, and 100% structural validity. Exact eleven-way answer accuracy was
74.24%, far below V130's 97.25% boundary. The one-sided 95% Wilson lower bound was 69.59%, below the frozen
95% gate.

Accuracy by truth kind was:

| Truth kind | Exact answer accuracy |
| --- | ---: |
| Declared known | 80.56% |
| Valid undeclared composite | 55.56% |
| Unsupported | 66.67% |
| Missing / insufficient | 100.00% |

No semantic subgroup met every frozen requirement. Per-choice accuracy ranged from 25.00% for `N03` to
95.83% for `K01`, apart from the perfect `A00` controls. False-known answers occurred on 18.33% of non-known
truths, exceeding the 10% ceiling.

## Error mechanism

The result is not explained by blindly copying the preliminary candidate. There were 68 incorrect answers;
only 5.88% selected the presented candidate, well below the modeled 75% candidate-attraction stress.
Abstention accounted for 30.88% of errors, also within its frozen bound. The dominant failure is broader
semantic confusion among exact known, valid-undeclared, unsupported, and insufficient choices.

The weakest choice-level results were `N03` at 25.00%, `K03` and `N02` at 54.17%, and `U00` at 66.67%.
Those aggregate results motivate a separate source-label identifiability audit: SGD defines novelty by an
unseen service-intent pair, and some unseen services reuse intent names found in declared services. That
possibility must be tested from frozen schema metadata rather than inferred from individual model errors.

## Downstream decision result

Using the frozen V130 policies at 97.25% assumed answer reliability did not rescue the empirical answers.
Across all nine prior/error-model conditions:

- mean regret was 1.7072--1.7773, above the 1.1667 ceiling;
- exact known action probability was 68.75%--80.56%, failing at least one condition;
- unsupported correctness was 66.67%, below 80%;
- false-known probability was 17.50%--18.33%, above 10%.

Every downstream performance gate failed. Complete hypothesis retention and zero execution still passed.

## Access and safety

- Selected SGD current-turn language records: 240
- Missing-observation controls: 24
- Manual language or raw-response inspection: 0
- Local model loads: 1
- Local model generations: 264
- API calls or training runs: 0
- Real service calls, side effects, or executions: 0
- Peak active memory: 16,623,987,800 bytes
- Runtime: 1,704.00 seconds

Freeze V132 without retries, prompt revision, threshold changes, larger/API models, or error mining. The
complete menu fixed the abstract interface's expressivity, but this local one-pass source did not supply the
required evidence strength. This result closes the current one-pass local realization branch and does not
authorize human-equivalence claims, induction, richer planning, APIs, training, authority, or execution.
