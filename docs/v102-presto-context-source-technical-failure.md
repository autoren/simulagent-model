# V102 PRESTO Source Technical Failure

V102 did not reach its scientific source gates. The single authorized archive download completed and
passed the frozen byte-size, MD5/ETag, generation, and last-modified checks. Both locked dev/test JSONL
members were then parsed automatically, but inventory construction stopped when a `seeded_notes` entry
did not satisfy the official README's strict assumption that both `name` and `text` are strings.

The exact exception was `ValueError: seeded note is invalid`, raised by the frozen structural validator.
No input, target, argument, context, seeded value, or other language was emitted or manually inspected.
There were zero model loads, generations, API calls, training runs, service calls, and external side
effects. Because the verified response bytes were not persisted before the parser ran, no source artifact
survived the failed process.

This is a technical parser-schema incompatibility, not evidence that PRESTO lacks eligible paired context
dependencies. Freeze V102 without a scientific feasibility conclusion. A fresh repair stage may retain
the exact source identity, dependency rule, counts, diversity thresholds, and access boundary while
treating non-string optional context leaves as absent rather than invalid. It must never stringify or
emit those leaves, and it must persist the verified archive before parsing so another structural error
does not require another network retrieval.
