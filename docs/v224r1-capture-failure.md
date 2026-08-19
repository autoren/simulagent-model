# V224r1 repaired-capture failure

The V224r1 JSON transport repair reached GitHub's GraphQL endpoint but the monolithic record page—up to 100 issues,
each with timelines plus nested pull requests, reviews, and file metadata—received HTTP 502 before any census artifact
was written. GitHub's GraphQL rate ledger increased by 65 points during the attempt. The implementation did not log a
per-page success counter, so it makes no exact claim about how many safe metadata responses were held transiently in
memory before the error.

The following facts are exact:

- no record metadata, preliminary census, release metadata, release index, deep audit, summary, or result artifact was
  persisted;
- no selected GraphQL field contained an issue/pull title, body, comment, review text, or commit message;
- no task language or protected research record was persisted or exposed to the research process;
- the pinned scope-policy snapshot remains the only capture artifact and its hash is unchanged; and
- no model, training, registration, mutation, service action, side effect, or execution occurred.

The failure shows that the originally implemented query performed deep provenance retrieval too early. The frozen V224
scientific design already specifies preliminary enumeration followed by hash-selected deep auditing. A valid repair
may execute those phases separately while preserving the exact record frame, outcome definitions, selection seed,
limits, thresholds, and decision rule.

