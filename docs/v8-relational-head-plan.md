# V8 normalized query-conditioned relational head preregistration

## Authorization

This is the single Stage 5 development experiment anticipated by the V8
analysis when a simpler structured head is insufficient. The locked additive
head and its locked ledger-derived decision both retained perfect paired
direction but failed the same worst-cell absolute balanced-accuracy gate. This
shows that the information is present but that additive pooled features and
their class boundaries do not transfer absolutely across every mechanism and
surface.

No final mechanic has been constructed or read. This stage remains confined to
the six exposed V8 development mechanics.

## Locked architecture

Reuse the locked float32 layer-6 mean record and component embeddings. The Qwen
backbone remains frozen and no component is re-extracted.

Project the record, candidate action, transition table, determinant role, and
evidence statement to 32 dimensions. Layer-normalize the summed
record/action/table projection to form the query, and separately normalize each
role and evidence row. For every row, concatenate:

1. query,
2. role,
3. evidence,
4. query × role,
5. query × evidence, and
6. role × evidence.

A shared linear layer plus GELU maps that explicit relational representation to
one row vector. Two shared row heads predict the five determinant statuses and
a binary outcome-sensitivity logit. The record ambiguity score is the maximum
row-sensitivity logit, reflecting the simulator definition that any sensitive
unresolved determinant makes the record ambiguous.

This is a low-parameter query-conditioned row head, not a generic attention
block. The multiplicative terms test the specific action–role–evidence binding
that an additive head could not express cleanly.

## Training and losses

Run the same six leave-one-mechanic-out folds. Train only on the `train` records
of the other five mechanics. Use the same structured components, matched
label-flip pairs, and three-surface groups as Stage 4.

Train full-batch Adam for exactly 300 steps with learning rate 0.003 and seed 0
plus fold number. No hyperparameter search is permitted. Inverse-frequency
weights are computed only within each training fold. The locked loss is:

`0.5 * determinant CE + 1.0 * row-sensitivity BCE + 1.0 * record BCE + 2.0 * paired logistic + 0.1 * surface variance`.

The decision threshold is the native binary-logit boundary, zero. It is not
fitted or changed per mechanic. This makes absolute calibration part of the
training objective rather than a post-hoc fold-specific correction.

## Evaluation and hard gates

Evaluate each omitted mechanic on canonical, entity-renamed, and paraphrased
surfaces. Retain the original Stage 4 hard gates unchanged:

- minimum cell balanced accuracy at least 0.65;
- mean cell balanced accuracy at least 0.75;
- minimum cell pair-direction accuracy at least 0.85;
- minimum cell determinant-status macro F1 at least 0.65; and
- minimum cell decisive-determinant accuracy at least 0.75.

Report ambiguity F1, AUC, margins, resolved-true/false pair direction, exact
ledger accuracy, losses, and saved head hashes in addition to the gates.

## Stop rule and firewall

If every gate passes, the method becomes eligible for a separately
preregistered, one-shot evaluation on one newly constructed final mechanic. If
any gate fails, stop V8 model development and do not construct or evaluate that
mechanic.

This lock permits one six-fold relational-head run, zero additional component
extractions, zero adapter runs, and zero final-mechanic evaluations. Tone Drift,
V3 test records, prior holdouts, V7 outputs, and any new/untouched V8 mechanic
remain prohibited.
