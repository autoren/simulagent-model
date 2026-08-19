# V127 Fresh SGD Typed-constraint Selectivity Feasibility Plan

## Question

V126 showed that nearest-neighbour similarity ranks clarification value in the wrong direction. V127 asks
whether a mechanistically different source of evidence could work: exact typed slot-name compatibility
between the current dialogue state and the six declared known intent schemas.

This is an oracle upper-bound audit. The source-authored frame annotations are not available to a deployed
policy and cannot be treated as parser or LLM output. A positive result would establish only that realizing
an accurate typed parser is worth testing. A negative result would close this slot-signature mechanism
before any model or parser run.

## Prospective design

Before reading the archive, select 576 unused SGD test turns from the text-free V124 inventory, balanced at
192 exact-known, 192 novel-valid, and 192 unsupported records. Exclude every V125 evaluation identifier.
Use the unchanged 11-choice V125 catalog and the unchanged V119 simulated clarification channel.

For each selected frame, automatically read slot names only from its accumulated state and non-intent USER
actions. Never access utterance fields or slot values. A declared known schema is compatible exactly when
the observed set is nonempty and contained in that schema's required-plus-optional slot set. Skip the query
only for one unique compatible known schema; otherwise query using a deterministic least-inconsistent known
candidate. There is one rule, no thresholds, no fitting, and no selection.

## Gates and boundaries

The trigger must skip 5%--95% of cases with at least 95% shadow-action precision. Under every frozen prior
and correlation condition it must meet the inherited regret, known, unsupported, and false-known gates;
queried cases must be worth at least the 0.30 query cost, skipped cases at most that cost, and selective
regret may not exceed query-all. Complete hypothesis retention and zero execution are mandatory.

No record-level evidence, language, slot values, model output, or action may be emitted. Passing authorizes
only a separately locked typed-parser realization design—not a language/model run, protected access,
capability induction, richer planning, API use, training, authority, or execution.
