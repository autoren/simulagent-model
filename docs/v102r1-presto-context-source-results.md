# V102r1 PRESTO Context-Dependency Source Result

## Outcome

V102r1 successfully repaired and exercised the technical parser, but the unchanged scientific source
gate failed. The official PRESTO archive again matched its frozen 415,990,813-byte identity and is now
persisted at SHA-256 `1fc671692cceb31fbda17e351e47f2cc52ee8779042f92dc26674cc0cca2167f`.

The two locked members contained 276,665 records: 82,547 development and 194,118 test. Of these, 3,665
development and 8,614 test records were en-US with `metadata.context == "human"`. The repaired parser
ignored 39,035 non-string optional context leaves without coercing or emitting them, confirming that the
repair addressed a real schema condition.

## Scientific result

The exact preregistered dependency rule admitted zero candidates. Consequently:

- development eligible candidates: 0 (required 64);
- protected-test eligible candidates: 0 (required 64);
- total eligible candidates: 0 (required 256);
- previous-turn-dependent candidates: 0 (required 64);
- seeded-state-dependent candidates: 0 (required 64);
- dependency source kinds: 0 (required 2);
- semantic root functions among eligible records: 0 (required 8).

Identifier disjointness, pair identity, human-only provenance, zero synthetic context, and text-free
inventory checks passed, but they cannot compensate for the failed availability gates.

## Boundary and interpretation

The archive was parsed automatically; no input, target, target argument, context, token, seeded value,
or prompt was emitted or manually inspected. There were zero model loads, generations, API calls,
training runs, service calls, and external side effects.

Freeze V102r1 as a negative result for this exact copy-based context-dependency construction. It does
not show that context is unimportant in PRESTO, nor does it assess abstention or any LLM. It shows that
the frozen target-argument delimiter and contiguous-copy criterion does not yield a viable paired source.
Do not select a PRESTO population or revise thresholds under V102r1. A successor may first run a separate,
text-free structural target-syntax census against the persisted archive, with all diagnostic features
locked before reading it; any materially different dependency definition must then be preregistered as
a new scientific construction.
