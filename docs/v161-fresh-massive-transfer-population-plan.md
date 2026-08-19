# V161 fresh MASSIVE transfer-population plan

## Purpose

V160 proved that a deterministic finite grammar can route perfectly when the request explicitly uses its
registered alias syntax. It did not test whether independently authored language can be translated into
that kind of finite codebook. V161 creates a new, text-free external transfer population from the pinned
MASSIVE 1.1 source.

This stage selects identifiers only. It does not extract an utterance, define or score an interface, run a
model, or evaluate novelty.

## Source and contamination boundary

The source is the immutable V100 MASSIVE candidate inventory. It contains structural identifiers,
partition, scenario, intent, class, intent-token-overlap count, and slot-type count, but no utterance,
annotated utterance, tokens, or slot values.

All 512 identifiers selected by V101 are removed before ranking. Those records have already been used by
the V104–V119 research sequence and cannot provide fresh transfer evidence. V161 must have zero identifier
overlap with V101.

## Four externally grounded classes

- `known_familiar`: declared catalog intent with current-utterance intent-token overlap;
- `known_unfamiliar`: declared catalog intent without that overlap;
- `novel_valid`: one of two valid source intents hidden from the catalog; and
- `unsupported`: an utterance from the completely withheld email scenario.

The source annotations provide independently checkable structural truth. Any future missing-observation or
unknown-alias controls must be added and labeled separately; they are not silently mixed into the external
source classes.

## Prospective selection

Use salt `simulagent-v161-fresh-massive-transfer-v1` and select 48 records per class in each role:

- `development_transfer` from MASSIVE validation;
- `protected_transfer` from MASSIVE test.

Selection first enforces the locked per-scenario minima, then fills by hash order using only candidate ID,
class, role, and scenario. The resulting 384 records must be role-disjoint and text-free.

## Noncompensatory gates

- exact source inventory and V101 exclusion identities;
- exactly 512 excluded candidate IDs;
- at least 48 unused candidates in every role/class pool;
- exactly 48 selected records per role/class, 192 per role, and 384 total;
- zero overlap with V101 and between transfer roles;
- three-scenario coverage for both known classes, two for novel-valid, and one for unsupported;
- minimum intent coverage of 6/8/2/4 for familiar/unfamiliar/novel/unsupported;
- no train records or language-bearing fields;
- zero archive reopen, utterance extraction or inspection, model/API calls, training, services, side
  effects, or execution.

## Claim boundary

Passing establishes only a fresh, text-free population of unused external identifiers for future controlled
open-set transfer. It is not language, interface, model, calibration, planner, or safety evidence. Language
extraction requires another prospective lock, and the protected role remains sealed beyond that stage.

