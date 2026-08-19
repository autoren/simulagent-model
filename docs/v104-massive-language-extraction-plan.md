# V104 MASSIVE Selected-Language Extraction Plan

## Purpose

V104 materializes language only for the 512 identifiers frozen by V101. It reopens the persisted V100
archive locally once, revalidates each selected record against the frozen text-free candidate inventory,
and writes separate development and protected-test JSONL artifacts. No selected identifier may be
replaced and no unselected utterance may be emitted.

## Record contract

Each output record contains its frozen identifiers, role, source partition, class, schema-visibility
status, authoritative scenario and intent, raw and annotated utterances, parsed typed slots, and the
reconstructed current-utterance/intent overlap count. Known classes must map to declared schema intents,
novel-valid to hidden intents inside catalog scenarios, and unsupported to the completely withheld
scenario. Slot types and familiarity must reconstruct exactly from source language.

## Boundary

Automatic extraction emits exactly 256 development and 256 protected-test records, 64 per class in each.
Neither artifact may be manually inspected in this stage. The protected test remains sealed until the
prompt, response grammar, deterministic controls, metrics, and noncompensatory gates are prospectively
locked. Passing does not authorize model loading, API use, training, posterior integration, planning,
or execution.
