# V30 direction: fresh signed-fact language interface

## Decision

Retain V28 as the selected structured inference architecture. Do not continue the V29
posterior-graph rule, do not add another score-weight or branch-budget sweep, and do not start
LoRA from the current exposed benchmark.

The next experiment should isolate the language representation on a newly generated,
surface-only development suite. Its primary challenger should convert each evidence clause to a
canonical signed fact—predicate, ordered arguments, and positive/negative/unknown polarity—then
use deterministic symbolic comparison to align that fact with the declared atom candidates.
This separates linguistic polarity normalization from candidate matching and program induction.

## Evidence for the pivot

- V28 marginal program MAP selects the target program in 11/12 evaluation episodes and passes
  frozen-support/oracle-query execution at 0.756, target retention at 0.917, and empty version
  spaces at 0.000.
- V29's exact posterior-marginal graph rule reduces exact support graphs from 0.694 to 0.611 and
  support/oracle execution from 0.756 to 0.577. Further posterior decoding is contraindicated.
- In V28, 25/36 evaluation support scenes are exact and 31/36 have exact assignments.
- Six of the eleven non-exact scenes have exact assignments. Nine of twelve wrong truth labels
  occur on the correct evidence/candidate edge.
- The native full-depth truth decoder is already wrong for eleven of those twelve labels.
- Eight of twelve truth errors are `negated_opposite`; eight occur in held-out `eval_d`
  surfaces. The dominant residual is polarity normalization, not program search.

## Proposed V30 protocol

Construct a fresh language-only development benchmark before accessing any new model result.
It is not a final relational benchmark and cannot support a world-model claim.

Freeze:

- new paraphrase families and generator hashes;
- disjoint fit, calibration, and sealed evaluation surface banks;
- unary and ordered-relation atoms, positive/negative/unknown states, and all established
  semantic operators;
- extra double-negation, contrastive-clause, inverse-relation, and distractor cases;
- entity-count and sentence-length strata;
- exact signed-atom, polarity, assignment, and scene-level metrics;
- one frozen primary challenger and decision thresholds before evaluation.

The primary challenger should emit a constrained canonical structure such as:

```json
{"predicate":"linked","arguments":["source","target"],"polarity":"negative"}
```

Candidate alignment and truth status should then be derived symbolically. The V26 direct
A/B/C decoder is the registered baseline. A deterministic parser for generator templates may be
reported only as an oracle ceiling, not as the language-system result.

## Gates and branches

- Require at least 0.98 atom value accuracy, 0.95 relation-order accuracy, and 0.80 exact scene
  accuracy on the sealed surface evaluation before reintegrating the challenger with V28.
- If signed extraction passes, freeze it and run one V28 integration replay, then construct a
  fresh relational benchmark protocol.
- If it fails mainly on polarity while a stronger frozen NLI/signed-fact model passes, replace
  the grounder without changing the symbolic system.
- If all frozen signed-fact challengers fail on the fresh suite, a narrowly scoped grounding
  adapter or LoRA becomes scientifically justified. Train only the language interface and keep
  the DSL, executor, program search, and integration gates frozen.
- If a tuned interface merely memorizes surface templates or harms novel paraphrases, stop the
  tuning branch rather than expanding end-to-end training.
