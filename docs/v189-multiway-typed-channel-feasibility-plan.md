# V189 multiway typed-channel feasibility plan

## Question

Can a finite categorical clarification reduce the number of interactions enough to close V188's binary bandwidth gap, and is any gain robust to a conservative bit-equivalent cost rather than dependent on an assumed cheap categorical answer?

## Questions and population

Reuse the frozen 14-contract development prior. Construct exactly three categorical questions from allowed semantic payload fields: declared domain, normalized intent concept, and transactionality. The full question set contains all three; a coarse control excludes the globally identifying intent menu.

No utterance language is read. Answers are deterministic dataset-provided contract attributes, not human, model, or deployment evidence.

## Prospective pricing sensitivity

The binary anchor remains 0.10. For a question with `k` reachable answer categories, conservative bit-slot cost is:

`o + (0.10 - o) * ceil(log2(k))`

where per-turn overhead `o` is fixed on the grid 0.00 through 0.09 in steps of 0.01. This decomposition always prices a binary question at 0.10. At zero overhead it charges purely for worst-case answer discrimination; positive overhead gives multiway answers credit for reducing turns.

An additional optimistic lower-bound scenario prices the answer at `0.10 * H(answer | current version space)`. It is a bound, not an operational estimate.

Generic trusted clarification remains 0.40 and the horizon is two categorical turns.

## Controls and interpretation

Compare exact adaptive, best fixed open loop, coarse-only exact adaptive, and always-generic clarification. Every terminal route must be singleton or generic, retain the target, and be exact.

Strict improvement under zero-overhead bit-slot pricing is robust theoretical multiway value. Improvement only with positive turn overhead or entropy pricing is conditional feasibility and requires external cost/UI evidence before any clean successor. If the global intent menu drives the result while exact and open loop tie, report menu compression—not adaptive planning.

No language/model run, protected access, registration, authority, action, or execution is authorized.
