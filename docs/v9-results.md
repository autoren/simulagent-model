# V9 results: neuro-symbolic natural-language evidence grounding

## Verdict

V9 validates the neuro-symbolic decomposition but does not authorize a final
mechanic or LoRA.

The deterministic half is complete: allowed determinant-value sets reproduce
the simulator oracle exactly. The frozen Qwen representation plus linear heads
also transfers evidence attribution and temporal status well. Its remaining
failure is semantic polarity—especially zero-shot negation and unseen operator
families—which prevents reliable allowed-value ledgers and matched-pair
decisions.

## Symbolic audit

The locked TypeScript evaluator enumerates every assignment compatible with the
allowed value sets, looks up the complete transition table, deduplicates
transition codes, and declares the action identifiable iff one code remains.

Against all 6,480 exposed V8 records and all 18 mechanic-by-surface cells it
produced:

- identifiability mismatches: 0;
- possible-transition-count mismatches: 0; and
- compatible-assignment-count mismatches: 0.

The independent Python implementation also reproduced all 2,160 V9r2 symbolic
targets exactly.

## Corpus and pre-model audit

V9 generated 2,160 natural-language records from 90 semantic contexts, with:

- 1,608 training and 552 calibration-context records;
- six mechanics;
- four language-template families;
- two operator families containing three mechanics each;
- three lexical surfaces;
- 360 intervention groups, including 192 label-changing groups; and
- current, unknown-current, stale-only, and conflicting-current evidence.

The first locked corpus was correctly rejected before model access. A synthetic
hexadecimal scene identifier reached 0.619 balanced accuracy in one
held-out-mechanic shortcut cell against a ceiling of 0.55. V9r2 removed exactly
that line and shifted the saved evidence offsets without changing any evidence,
target, split, or model gate.

V9r2 then passed with:

- zero synthetic identifiers;
- zero duplicate or cross-split prompts;
- zero malformed spans or symbolic mismatches;
- zero determinant identifiers or literal target labels in observation prose;
- maximum metadata-only match balanced accuracy: 0.512; and
- maximum evidence-position-only match balanced accuracy: 0.512.

Report-only character baselines showed that the corpus contained genuine
linguistic signal: context grounding balanced accuracy 0.864, allowed-value
accuracy 0.794, and temporal accuracy 1.000.

## Frozen evaluation

The frozen `mlx-community/Qwen3.5-0.8B-4bit` layer-6 mean extraction encoded
2,986 unique prompts representing 42,000 determinant–evidence candidates. Token
lengths ranged from 120 to 180; none were truncated. Three fixed linear heads
predicted evidence match, allowed values, and temporal status over 13 locked
folds.

| Fold | Span accuracy | Allowed values | Temporal | Symbolic BA | Complete flip pairs | Complete ledger |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Context | 1.000 | 0.993 | 1.000 | 1.000 | 1.000 | 0.980 |
| Beacon Calibration | 0.999 | 0.821 | 0.999 | 0.786 | 0.583 | 0.632 |
| Generator Tuning | 0.974 | 0.698 | 0.979 | 0.958 | 0.917 | 0.479 |
| Hatch Traversal | 1.000 | 0.562 | 1.000 | 0.771 | 0.542 | 0.562 |
| Mirror Power Trip | 0.943 | 0.601 | 0.965 | 0.792 | 0.703 | 0.189 |
| Mirror Rejection | 0.922 | 0.654 | 0.943 | 0.917 | 0.833 | 0.417 |
| Pressure Hatch Relock | 0.802 | 0.520 | 0.880 | 0.727 | 0.615 | 0.078 |
| Binary-partition operators | 0.860 | 0.617 | 0.943 | 0.700 | 0.526 | 0.285 |
| Multiway-partition operators | 0.840 | 0.518 | 0.796 | 0.695 | 0.500 | 0.174 |
| Inspection Report | 0.974 | 0.738 | 0.984 | 0.899 | 0.889 | 0.594 |
| Operator Log | 0.942 | 0.312 | 0.942 | 0.745 | 0.815 | 0.029 |
| Questioned Claim | 0.960 | 0.148 | 0.960 | 0.781 | 0.741 | 0.022 |
| Technical Summary | 0.989 | 0.765 | 0.989 | 0.950 | 0.963 | 0.551 |

The hard gates passed for every span axis, temporal accuracy, and downstream
symbolic balanced accuracy. They failed for:

- minimum allowed-values accuracy: 0.148 versus 0.650; and
- minimum complete label-flip-pair accuracy: 0.500 versus 0.600.

No mean can override these worst-fold failures.

## Interpretation

V9 separates three capabilities that V8 partially conflated:

1. **Evidence attribution:** The model generally identifies which sentence is
   about the queried determinant. This transfers strongly across contexts,
   mechanics, templates, and operator families.
2. **Temporal epistemic status:** The model recognizes current, unavailable,
   stale, and conflicting evidence with a worst-fold accuracy of 0.796.
3. **State polarity:** The model does not robustly infer whether the selected
   evidence establishes the active state, inactive state, or both.

The Questioned Claim template is the clearest diagnostic. It expresses the true
state by rejecting the opposite claim. With that template held out, span
accuracy remained 0.960 and temporal accuracy remained 0.960, while value
accuracy collapsed to 0.148. The representation found the right evidence but
the linear value head largely failed to compose negation with the mentioned
state.

Operator-family transfer shows a related problem. Evidence location remained
above 0.84, but value accuracy was only 0.617 for binary partitions and 0.518
for multiway partitions. Mechanic-specific state language still transfers
unevenly.

The deterministic layer prevents many grounding errors from becoming final
errors when the mistaken value is transition-invariant: worst-fold symbolic
balanced accuracy still reached 0.695 and passed its gate. That robustness is a
benefit of the neuro-symbolic architecture, but it cannot justify inaccurate
grounding ledgers or failed causal pairs.

## Decision and next research step

The current method is not eligible for final-mechanic evaluation or LoRA. All
firewall counters remain zero, and neither was run.

For production, the deterministic evaluator is ready whenever it receives a
correct allowed-values ledger.

For a future development version, focus only on the demonstrated grounding
bottleneck:

- create multiple exposed negation and claim-rejection constructions so a new
  negation construction can be held out without making negation itself absent
  from training;
- balance active/inactive paraphrases within every mechanic and surface;
- add explicit contradiction, negation-scope, and temporal-update minimal
  pairs;
- hold out state-lexicon families separately from transition-operator families;
  and
- compare token-aware or NLI-style frozen heads before considering LoRA.

This should be a new locked development corpus. Retuning the current value head
against the observed Questioned Claim failure would not be a clean test.

## Reproducibility and firewall

Key hashes:

- V9 symbolic audit: `6e49e22329a498c6160bd46232f4ad449b6d50274aa1b6affe268a384de7d961`
- V9r2 dataset: `cff130b8efa713630ad915207bc18d8e43fb77d149e86eb72365fe06b71e2c78`
- frozen feature artifact: `3b42d70d4b5b1e679a68b10a1d0f9e1c676c4a49053e4e3adda04cd9858327fe`
- frozen evaluation result: `efb4f4a71f77a834fccacb2e396caf3d057fbdfb822f34457cb6ec09c1dfe3aa`

No Tone Drift, V3 test, prior holdout, V7 output, untouched V8 mechanic, or
final V9 mechanic was read. No adapter was trained.
