# V187: Clean typed-clarification planner development result

## Bottom line

V187 is a clean and informative negative result. The V186 binary codebook can identify all 14 capability contracts, but it is not cost-effective from an uninformative 14-contract start under the frozen costs. Exact adaptive planning, greedy information gain, and the best fixed open-loop policy all choose immediate generic trusted clarification at cost 0.40.

## Frozen setting

- Initial version space: all 14 contracts.
- Prior: frozen frequencies of the 120 observed development targets; all 14 contracts have positive mass.
- Raw V186 questions: 164.
- Distinct binary partitions after exact column deduplication: 25.
- Typed-question horizon: 4.
- Typed question: 0.10.
- Generic trusted clarification: 0.40.
- Safe deferral: 0.50.
- Clean answers: deterministic dataset-provided oracle bits.

No utterance or dialogue language was read. Protected language remained sealed. No model, API, training, registration, trusted-state mutation, service call, side effect, action, or execution occurred.

## Policy comparison

| Policy | Mean cost | Mean questions | Typed-only completion | Final exactness |
|---|---:|---:|---:|---:|
| Exact adaptive | 0.4000 | 0.0000 | 0.0000 | 1.0000 |
| Best fixed open loop | 0.4000 | 0.0000 | 0.0000 | 1.0000 |
| Greedy information gain | 0.4000 | 0.0000 | 0.0000 | 1.0000 |
| Always generic | 0.4000 | 0.0000 | 0.0000 | 1.0000 |
| Frozen source order | 0.6250 | 3.5833 | 0.3333 | 1.0000 |
| Immediate deferral | 0.5000 | 0.0000 | 0.0000 | 0.0000 |
| Target-informed oracle | 0.1000 | 1.0000 | 1.0000 | 1.0000 |

Every exact-policy terminal state was safe: the hidden target was retained and generic trusted clarification returned the exact contract. All 12 missing controls remained insufficient at zero cost.

The scientific gates did not pass. Exact adaptive planning had no typed-only completions, no improvement over always-generic clarification, no improvement over the best fixed open-loop policy, and no history-dependent question choices.

## Why the binary channel failed economically

The failure is not identifiability. V186 proved that every pair is separable. The problem is information economics.

The target-blind policy begins with 14 possibilities. No single frozen binary question isolates enough prior mass to justify paying 0.10 before a possible 0.40 generic fallback. Four binary questions already cost as much as the exact generic answer, and the restricted partitions do not create a cheaper partial strategy. Therefore the Bellman-optimal action is generic clarification at the root.

The target-informed oracle is much cheaper because every contract has a direct intent-concept confirmation question. But that oracle chooses the right question using the hidden target. Its 0.10 cost is evidence that a reliable external proposal or a higher-bandwidth clarification could be valuable, not evidence that the current target-blind policy can obtain that value.

Source order demonstrates the danger of asking questions merely because they are semantically valid: it spends 0.625 on average and still needs generic clarification for two thirds of targets.

## Decision and successor

Freeze V187 without correlated-error, protected, or model follow-up. Robustness wrappers cannot make a clean policy cost-effective; triple repetition would only raise its cost.

Do not repair this result by changing the 0.10/0.40 costs, extending the horizon, or opening protected language. The next justified study is a text-free clarification-channel economics frontier:

1. compute unrestricted and frozen-codebook decision-tree lower bounds;
2. identify the exact binary-question break-even cost relative to generic clarification;
3. determine whether a finite multiway typed question or a separately validated non-authoritative proposal is required to approach the target-informed gap;
4. only then preregister a new clean channel design.

This successor must keep source annotations as simulated oracle answers and must not grant language or models authority.
