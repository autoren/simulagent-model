# V101 MASSIVE Population Plan

## Purpose

V101 selects the MASSIVE development and protected-test records while the source remains text-free. It
uses only the already frozen V100 candidate identifiers, structural class labels, source partitions,
scenarios, and intents. It does not reopen the archive or extract any utterance.

## Selection

Select 64 records from each of the four V100 classes independently in the official validation partition
and again in the official test partition. Within each split and class, first hash-select the locked
minimum from every represented scenario: 12 for each known-class scenario and 16 for each novel-valid
scenario. The unsupported class has one structurally withheld scenario and needs no extra balancing.
Fill the remaining class quota by a second independent hash order. The salt and all quotas are frozen
before any selected identifier is computed.

The resulting development and protected-test populations must each contain 256 records and be identifier-
disjoint. Each known class must cover all three catalog scenarios, novel-valid must cover both hidden-
intent scenarios, and unsupported must cover the one withheld scenario. Per split, known-familiar must
cover at least six intents, known-unfamiliar at least eight, novel-valid both hidden intents, and
unsupported all four source intents.

## Boundary

The selection artifact may contain only candidate/source identifiers, split, class, scenario, intent,
overlap count, and slot-type count. It may not contain an utterance, annotated utterance, normalized
token, slot value, or prompt. Passing V101 authorizes only a new, separately locked automatic extraction
stage. It does not itself authorize archive reopening, selected-language extraction, manual inspection,
model loading, API use, training, posterior integration, planning, or execution.
