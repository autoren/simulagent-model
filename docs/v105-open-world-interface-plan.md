# V105 Typed Open-World Interface Plan

## Visible catalog

Compile a public catalog from MASSIVE training annotations only. It exposes exactly 12 declared intents
inside `calendar`, `iot`, and `recommendation`, plus the slot types observed for each declared intent.
The two hidden valid intents and the withheld `email` scenario must not appear anywhere in the catalog or
prompt.

## Proposal contract

Given a visible catalog and an observation, a proposer returns exactly four JSON keys: `status`,
`known_intent`, `novel_scenario`, and `confidence`. `KNOWN` requires one listed intent; `NOVEL` requires
one visible scenario; `UNSUPPORTED` and `ABSTAIN` require null intent and scenario. Any parse, key, type,
range, or invariant failure maps deterministically to zero-confidence `ABSTAIN`.

The complete safe universe always contains the 12 known intent hypotheses, one novel-capability
hypothesis per visible scenario, `UNSUPPORTED`, and `INSUFFICIENT_EVIDENCE`: 17 hypotheses total. The LLM
may propose or rank but can never delete a hypothesis, alter authoritative state or posterior, select an
action, or execute anything.

## Insufficient-evidence control

Hash-select 64 already frozen identifiers per role and pair each with `observation_available = false`.
No utterance is exposed in this condition. The deterministic runtime response is abstain-and-ask; a model
may be evaluated only in shadow. This is an explicit missing-observation intervention, not naturally
occurring external-language evidence.

## Boundary

V105 reads only the local source archive's training annotations and text-free population identifiers.
It does not read selected development language or protected-test language. Passing authorizes only the
next prospective lock for baselines, metrics, costs, calibration, and one local model condition.
