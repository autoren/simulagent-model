# V128 Fresh SGD Typed-relation Feasibility Results

## Outcome

V128 failed the preregistered oracle-feasibility gates:

> `oracle_typed_relation_support_infeasible_close_annotation_signature_family`

The one-time run built six frozen support sets from 4,881 source-authored known training frames and evaluated
432 records disjoint from both V125 and V127. It accessed no utterance fields or slot values and emitted no
record-level evidence. Typed relations were present for 301 records (69.68%).

The unique-support rule skipped 102 records (23.61%): 95 known and seven novel-valid. Skip precision was
93.14%, an improvement over V127's 87.21% but below the locked 95% gate. Across all nine prior/correlation
conditions, skipped clarification value was 0.4600--0.5336, still above the 0.30 cost. Selective regret was
0.9856--1.0557: better than the 1.1667 ask-always control, but worse than query-all in every condition.

The selective policy also reached only 72.09%--72.56% exact-known probability, below the 80% requirement.
Unsupported correctness remained 88.05%--93.13% and false-known probability stayed below 3.41%, but those
gates cannot compensate for incorrect known candidates and unsafe skips. The queried subset had very high
clarification value because a wrong known candidate is costly and the frozen candidate-specific channel
cannot name an alternative exact known intent after rejecting it.

V128 is an oracle upper-bound failure. A real act/slot parser or LLM would add extraction error and cannot
rescue the frozen rule by implementation quality alone. Freeze the result and close the annotation-signature
family: do not mine individual tokens, add frequency thresholds, combine it post hoc with V126 similarity,
or build a parser for this trigger. The current evidence supports universal clarification over selective
querying whenever the simulated channel is available. A successor should revisit the clarification
interface itself—especially its inability to recover a different exact known intent—not search for another
cheap pre-query proxy. Language/model runs, protected access, capability induction, richer planning, APIs,
training, authority, and execution remain closed pending a separately locked interface audit.
