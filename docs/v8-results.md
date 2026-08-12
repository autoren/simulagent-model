# V8 results: structured contrastive transition reasoning

## Verdict

V8 validated the central diagnostic claim but did not satisfy the advancement
criteria for an untouched final mechanic.

The frozen Qwen representation transfers the direction of dense causal
evidence interventions extremely well: every leave-one-mechanic-out pair test
on every surface scored 1.0. The learned heads do not yet transfer a reliable
absolute decision boundary or a complete determinant ledger to every unseen
mechanic. V8 therefore stops before final-mechanic construction and before
LoRA.

## What was run

V8 used six exposed simulator-derived mechanics, three surface variants, dense
matched label-flip and same-label interventions, an explicit non-answer-bearing
transition schema, and no test/final data. The compiled corpus contained 6,480
records and 1,080 intervention groups, including 576 label-flip groups.

All pre-model shortcut gates passed. The role-scrubbed metadata, unigram,
character, and length baselines were at chance under their hard gates. The
full role-aware character baseline was reported separately and reached at most
0.6354 AUC with 0.5 balanced accuracy.

The following locked stages were then executed:

1. frozen layer-6 mean and pair-difference LOMO probes;
2. an additive action-conditioned structured head;
3. a no-training ledger-derived decision diagnostic; and
4. one normalized query-conditioned relational head, as the preregistered final
   development attempt.

## Gate results

| Locked stage | Min cell BA | Mean cell BA | Min pair direction | Min status macro F1 | Min decisive determinant | Decision |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Frozen diagnostic | n/a | n/a | 1.000 pair-difference; 0.750 pointwise direction | n/a | n/a | Advance to structured head |
| Additive structured head | 0.400 | 0.880 | 1.000 | 0.700 | 1.000 | Fail |
| Ledger-derived decision | 0.400 | 0.806 | 1.000 | 0.700 | 1.000 | Fail |
| Query-conditioned relational head | 0.625 | 0.896 | 1.000 | 0.145 | 1.000 | Fail and stop V8 development |

The relational head improved the worst absolute cell from 0.400 to 0.625, but
the locked minimum was 0.650. Its worst absolute cell was Mirror Power Trip on
the entity-renamed surface. Its status macro F1 collapsed on held-out Beacon
Calibration (0.145), despite 1.0 accuracy in selecting the most sensitive row
within ambiguous records. This is exactly why both absolute and structured
worst-cell gates were preregistered.

## Interpretation

The positive result is narrow but strong:

- causal pair direction generalized across all six mechanics and all three
  surfaces for every trained head;
- the direction held separately for unresolved-to-known-true and
  unresolved-to-known-false pairs;
- action-conditioned heads consistently identified the decisive row within
  ambiguous records; and
- mean absolute performance was high, reaching 0.896 balanced accuracy for the
  relational head.

The negative result is equally important:

- relative ordering does not imply a transferable absolute classifier;
- converting row-status logits directly into the record decision did not fix
  the boundary shift;
- explicit multiplicative action–role–evidence interactions improved the worst
  balanced accuracy but damaged unseen-mechanic status semantics; and
- further threshold or loss-weight tuning on these six folds would now be
  repeated adaptation to observed development failures, not a clean test.

The data support the claim that the frozen representation contains a highly
transferable local intervention direction. They do not support the claim that
the current learned head performs robust, absolute epistemic transition
classification on unseen mechanics.

## Recommended next approach

Do not proceed to LoRA or to a final-mechanic score with the current head.

For a product system, keep the deterministic simulator as the authoritative
identifiability engine. The transition schema and evidence ledger are already
structured, so enumerating compatible hidden assignments and comparing their
transitions is exact, auditable, and simpler than asking a language-model head
to relearn that algebra.

For the research question, the next version should be neuro-symbolic:

1. use the model only to ground unstructured observations into determinant
   identities and resolved values;
2. apply a deterministic transition-table evaluator to derive
   outcome-sensitive versus outcome-invariant uncertainty; and
3. evaluate evidence grounding and transition resolution separately.

Before another learned end-to-end head is attempted, add more exposed
development mechanics balanced by transition operator, determinant arity,
class prevalence, and invariant branches. Use operator-family holdouts in
addition to mechanic holdouts. A true token-level action-to-evidence attention
model is justified only when the evidence itself is natural language; for the
current structured JSON task, a symbolic evaluator is the more faithful
approach.

## Data firewall and reproducibility

Every result reports zero reads of V3 test records, prior holdouts, Tone Drift,
V7 model outputs, and untouched V8 mechanics. No new final mechanic was created
or evaluated, no adapter was trained, and Qwen remained frozen.

Key result hashes:

- frozen diagnostic: `8b1d2879e2ea157b713627669b94da53b1696ec758a57edb4223735dbaa92ad1`
- additive structured head: `2ff494a6bb0cd6a97d4f9b83f3b49717a3e0eaa2db8caca9973cbfb51cd8fea8`
- ledger-derived decision: `ea04e817233e9327c55d36f24df4f0cac2aee6267f7eae8df582e1ca1b98971f`
- query-conditioned relational head: `60c4050c77dc31b021b312bb796593b04e53e3047ebe589b9dc852faedc61c77`

The TypeScript suite (27 tests), Python suite (50 tests), and TypeScript build
all passed before the final locked run.
