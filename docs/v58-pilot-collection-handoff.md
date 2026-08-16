# V58 pilot collection handoff

Status: ready for a human collection coordinator, but packet release and language collection are not authorized by the current locks.

## Sealed pilot packets

- `data/v58-human-authored-known-ontology-language/author-packets/pilot_writer_slot_00.json`
- `data/v58-human-authored-known-ontology-language/author-packets/pilot_writer_slot_01.json`

Each packet contains 60 text-free prompts: 12 each for distractor scope, argument reversal, contrastive focus, inverse relation, and direct relation. The five evaluation-only construction families have no pilot prompts. The packet files must remain byte-identical to `configs/v58-author-packet-seal.json`.

## Coordinator checklist before a separate release authorization

1. Recruit two adult human pilot writers who are not evaluation writers, validators, adjudicators, or candidate developers.
2. Assign each person to one pseudonymous pilot slot. Keep the real-identity mapping outside this repository and outside candidate-development access.
3. Explain that submissions must be written personally without a language model, paraphraser, or other generative writing tool.
4. Obtain the per-submission research-consent, right-to-contribute, no-generative-assistance, and CC-BY-4.0 attestation.
5. Use `configs/v58-human-submission.schema.json` for each submission. Do not add target, oracle, validator, or candidate fields.
6. Store pilot submissions in a new location; never edit the sealed packet files.
7. Recruit two independent validators per submission and a separate adjudicator. Use the validation and adjudication schemas in `configs/`.
8. Keep candidate outputs hidden from writers, validators, and the adjudicator. Keep evaluation packets unreleased.
9. After collection, run the frozen structural validator and a new preregistered pilot population audit. Do not write or tune the candidate before pilot collection is sealed.

## Stop conditions

Stop collection and do not seal a pilot population if any writer used generative assistance, consent or license provenance is incomplete, identity roles overlap, a packet hash changes, validation records are not independent, disagreements remain unadjudicated, or evaluation packets/text become visible to candidate development.

This handoff does not claim that any human text has been collected, does not authorize packet release, and does not authorize candidate development or evaluation.
