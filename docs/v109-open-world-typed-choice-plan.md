# V109 Single-Choice Typed Interface Plan

V109 is a fresh interface condition, not a retry or rescore of V107. It uses the 128 V106 calibration-
membership records that were excluded from V107 model generation, plus the same 64 language-free missing-
observation controls. The protected test remains unopened.

The interface compiles the complete 17-hypothesis universe into 12 known-intent choices, three visible-
scenario novelty choices, one unsupported choice, and one insufficient-evidence choice. Each has exactly
one unique code. The model returns only `choice_id` and `confidence`; deterministic code validates that
single code and expands it into the frozen V105 decision representation. Aliases, bare intent names,
qualified intent names outside the choice code, extra fields, retries, and post-hoc repairs are forbidden.

The study separately gates serialization mechanics and semantic quality. This matters because V108 showed
that canonicalizing the visible alias repaired known grounding but also exposed false-known decisions on
novel requests. If the new interface is mechanically reliable yet semantic gates fail, the appropriate
successor is not another formatting tweak: it is a sequential clarification problem in which asking and
acting have explicit costs. No V109 outcome directly authorizes protected-test access, API use, training,
planning authority, tool use, or execution.
