# V115 Two-Pass Contrastive Catalog-Fit Results

## Outcome

V115 is a valid negative development result. The pinned local Qwen3.8-27B 4-bit model completed the
prospectively locked condition: 192 new balanced MASSIVE test-partition records and 48 controlled missing-
observation fixtures, two generations per fixture, 480 generations total, one model load, temperature zero,
and no retries. Both passes had 100% structural validity and all controls abstained in both passes.

The second pass did not make novel-capability evidence reliable. It explicitly selected `N00` nine times:
five true positives and four false positives. Novel precision was 55.56%, recall 10.42%, F1 17.54%, non-
novel false-positive rate 2.78%, ECE 0.2213, and Brier score 0.2358. Precision, recall, and calibration failed
their frozen gates.

The correct locked decision is:

> `contrastive_evidence_negative_close_two_pass_single_model_branch`

No independent-source confirmation, schema induction, protected-test access, or richer sequential planning
is authorized by V115.

## What the contrastive pass did

Aggregate pass-two verdict counts were:

| True class | C00 confirm | O00 other known | N00 undeclared-valid | U00 unsupported | A00 insufficient |
| --- | ---: | ---: | ---: | ---: | ---: |
| known familiar | 44 | 1 | 1 | 0 | 2 |
| known unfamiliar | 35 | 3 | 2 | 0 | 8 |
| novel valid | 8 | 20 | 5 | 1 | 14 |
| unsupported | 1 | 0 | 1 | 42 | 4 |
| missing controls | 0 | 0 | 0 | 0 | 48 |

The key failure is visible in the novel-valid row. For 28 of 48 novel-valid requests, the review selected
either the proposed known capability or another declared capability; it explicitly recognized only five as
valid but absent from the catalog. In other words, a second look at the same request and catalog mostly
reinforced semantic assimilation to the closest known capability rather than detecting catalog absence.

This is not an interface-validity failure. Both passes produced valid machine-readable outputs on every
fixture, and pass two remained good at ordinary catalog tasks: 81.25% exact known-intent accuracy, 87.5%
unsupported recall, and 97.67% unsupported precision. But it had 30.21% false-known acceptance, only 10.42%
novel-scenario accuracy, and poor confidence calibration.

## Policy comparison

| Shadow policy | Exact decision | Known exact | False-known | Top-80% error | Mean regret |
| --- | ---: | ---: | ---: | ---: | ---: |
| frozen V112 policy from pass one | 60.94% | 76.04% | 5.21% | 24.03% | 0.7760 |
| V115 two-pass confirmation policy | 60.42% | 77.08% | 7.29% | 24.68% | 0.9375 |

The two-pass policy gained 1.04 percentage points of known accuracy, but slightly reduced overall exact
accuracy, increased false-known acceptance by 2.08 points, increased selective error, and increased mean
regret by 0.1615. It therefore failed the inherited combined-policy qualification as well as the new
contrastive-evidence gates.

The first direct typed pass still achieved 86.46% known-intent accuracy on this population. That reinforces
the corrected project interpretation: supported-request grounding under an unambiguous interface is not the
main bottleneck. The unresolved problem is deciding whether a plausible semantic interpretation belongs to
the declared capability catalog at all.

## Scientific boundary

V115 rules out a specific repair:

> Asking the same frozen model to challenge its own candidate against the complete catalog, in a second
> static no-retry pass, does not create reliable novel-capability evidence.

It does not show that interactive clarification or externally grounded evidence is useless. The two passes
saw the same user request and the same capability catalog; they did not receive a new answer, observation,
or consequence that could distinguish competing hypotheses.

The justified successor is therefore a separately preregistered multi-turn development branch in which the
system—not the LLM—selects a deterministic, typed clarification query and evaluates the answer as new
evidence. Before any model run, a language-free oracle feasibility audit must show that the proposed answer
channel can actually separate known, novel-valid, and unsupported hypotheses and can improve a frozen
clarification policy. A negative feasibility audit should close this path without generation.

## Safety and access

- Fresh development-language reads: 1
- Protected-test language reads: 0
- Manual language or raw-response inspection: 0
- Local model loads: 1
- Local model generations: 480
- API calls: 0
- Adapter-training runs: 0
- Real service calls or external side effects: 0
- Actual executions: 0
- Safe-hypothesis retention: 100%
- Peak active memory: 17,601,963,080 bytes
- Runtime: 6,161.87 seconds

Freeze V115 without retrying, retuning, or mining individual records. Keep the model permanently
nonauthoritative and keep protected access, induction, capability registration, richer sequential planning,
APIs, training, actions, and execution closed.
