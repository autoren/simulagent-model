# V87 External-Language Source Audit Results

V87 passed its narrow source-feasibility gate. The frozen audit compared three public task-oriented
dialogue sources using only repository documentation and file metadata. The Schema-Guided Dialogue
(SGD) dataset was the sole source satisfying every noncompensatory requirement: independently authored
human user language, an explicit dataset license, machine-readable service schemas, and deterministic
turn-level links from utterances to service, active intent, requested slots, and slot values.

The selected source is the archived official SGD repository at commit
`e852981ae34990f4358979625854259302feaa78`, released under CC BY-SA 4.0. V87 acquired only the pinned
`dev/schema.json` and `dev/dialogues_001.json` files. Their exact byte sizes and Git blob identifiers
matched the preregistration. Derived source metadata retains the dataset name, revision, URLs, license,
and share-alike designation.

The one-shot code-only inventory found 128 dialogues and 825 user turns. After the frozen service-family
exclusions, all 825 turns remained structurally eligible: 757 had an active intent and 68 had the genuine
`NONE` label. The eligible population spans three services (`Flights_3`, `Restaurants_2`, and
`RideSharing_1`) and seven service-intent labels. The structural record index has SHA-256
`ebdb2204bf437782fe6d4a4af8f6489905f1363cc00cd56289255a0f7446f2a2`.

No utterance or slot-value text was emitted by the inventory, no record was manually inspected, and no
local model, API model, adapter, live service, tool action, or external side effect was used. Taskmaster-1
remains a useful future comparator but was not eligible for this exact experiment because its audited
span/API-argument format lacks the required turn-level active-intent and accumulated-state contract.
The audited MultiWOZ reference repository was conservatively excluded because its metadata did not
establish one self-contained normalized schema-to-user-frame contract without opening versioned data
and implementation details.

This result authorizes only a separate preregistration that selects a sealed, non-executable external
language shadow subset from the frozen structural index. Selection must occur by a fixed hash rule before
utterance extraction or scoring, preserve genuine `NONE` cases, retain CC BY-SA attribution, and keep all
language permanently non-executable. It does not yet authorize a model call, prompt choice, tuning,
training, API dependency, or execution authority.
