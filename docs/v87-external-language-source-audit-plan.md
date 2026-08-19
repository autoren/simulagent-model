# V87 External-Language Source Audit Plan

V87 closes the source, license, mapping, and leakage gate required by the frozen V86 outcome before
any externally authored dialogue payload or model is accessed. The audit uses only official repository
documentation, immutable commit identifiers, archive status, file sizes, and Git blob identifiers.
This metadata exposure is disclosed rather than described as discovery-clean source selection.

Three public task-oriented dialogue sources are compared under noncompensatory gates. Schema-Guided
Dialogue (SGD) is the only eligible source: its paid-crowd-worker user language is linked at each user
frame to a named service and a typed state containing active intent, requested slots, and slot values.
The same official repository supplies a machine-readable schema and an explicit CC BY-SA 4.0 license.
Taskmaster-1 has strong human-language provenance and a CC BY 4.0 license, but its span/API-argument
annotations do not provide the required turn-level active-intent and accumulated-state contract.
The audited MultiWOZ reference repository is conservatively excluded because its metadata does not
establish one self-contained normalized schema-to-user-frame contract without opening versioned data
and implementation details.

If the design audit passes, it authorizes downloading only `dev/schema.json` and
`dev/dialogues_001.json` from SGD commit `e852981ae34990f4358979625854259302feaa78`.
The bytes must reproduce their registered Git blob identifiers before parsing. Parsing is code-only:
no utterance may be manually inspected before a future subset lock, no language score may be computed,
and no model may be loaded. Calendar and messaging service prefixes, plus the four V84 synthetic
families, are excluded before any future selection to reduce semantic leakage from earlier work.

V87 does not yet create a benchmark. A passing inventory may support a separate preregistration with a
fixed record population, deterministic hash selection, genuine `NONE` cases, exact typed targets,
non-executable provenance, and frozen local-model access. It never authorizes an API dependency,
adapter training, live service transaction, or tool execution.
