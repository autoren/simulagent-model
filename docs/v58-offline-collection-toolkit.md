# V58 offline pilot collection toolkit

The toolkit is frozen before human text exists. It does not authorize collection. Its command-line entry points require a separate `v58_pilot_release_lock` that must bind the current author-packet seal and explicitly authorize pilot—but never evaluation—packet release and collection.

## Components

- `python/render_v58_pilot_form.py` verifies a release lock and sealed pilot packet, then creates one self-contained offline HTML form. The form has no network or model dependency and downloads schema-conforming JSONL locally.
- `python/v58_pilot_intake.py` verifies the release lock again, binds submissions to all 120 sealed pilot prompts, validates attestations and quotas, and later validates two independent reviews plus required adjudications. It prints only aggregate counts and errors, never submitted text.
- `configs/v58-human-submission.schema.json`, `configs/v58-human-validation.schema.json`, and `configs/v58-human-adjudication.schema.json` remain the exchange contracts.

## Future coordinator sequence

1. Obtain a separately audited pilot-release lock after the two human writers, validators, adjudicator, consent procedure, and external identity-map custody are confirmed.
2. Render each of the two pilot forms to an external coordinator-controlled location. Do not place completed forms or downloaded JSONL in candidate-development storage.
3. Concatenate the two writer JSONL files without changing records and run submission intake. A passing receipt requires exactly 120 unique, packet-bound submissions and all per-writer/per-family quotas.
4. Collect two blinded reviews per submission. Record disagreements in the adjudication schema using a third, disjoint person.
5. Run review intake. A passing receipt requires at least 0.90 raw verdict agreement, complete review/adjudication structure, and 120 finally accepted prompts.
6. Only a subsequent pilot-population audit and seal may authorize candidate development. Evaluation packets remain unreleased throughout.

The toolkit does not infer human authorship. It enforces workflow, attestation, role, packet, quota, and record integrity; real coordinator oversight remains required.
