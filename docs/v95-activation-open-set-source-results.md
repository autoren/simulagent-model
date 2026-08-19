# V95 Activation-Turn Open-Set Source Result

## Verdict

V95 is a clean negative source-feasibility result and a strong partial validation of the redesigned
class construction. The current-turn, intent-activation rules solved both substantive V94 failures, but
the fresh shard contained only three eligible services. Retaining the preregistered three-service
catalog therefore left no fourth service to withhold as unsupported.

The scientific source gate failed, so V95 stops before population selection, language extraction,
manual inspection, or model access.

## What passed

The pinned shard produced 1,044 structurally valid source records and 251 annotated intent activations.
The catalog contained `Hotels_1`, `Hotels_4`, and `Music_1`, with six supported pairs. The locked
service-stratified partition successfully hid exactly two pairs from two different services while
retaining four declared pairs:

- hidden: `Hotels_4::ReserveHotel`, `Music_1::LookupSong`;
- declared: both `Hotels_1` pairs, `Hotels_4::SearchHotel`, and `Music_1::PlaySong`.

Every non-unsupported class passed its preregistered count and coverage gates:

- known familiar: 80 records across three services;
- known unfamiliar: 38 records across three services;
- novel valid: 64 records across exactly two services;
- insufficient evidence: 46 records across three services.

This confirms that restricting non-`NONE` examples to intent activations and using only the current turn
for lexical overlap produces a viable familiar/unfamiliar split without relying on continuation turns.

## What failed

The source required at least four eligible fresh services: three catalog services plus one completely
withheld unsupported service. Only three eligible fresh services existed after the prospectively frozen
exclusions. The algorithm correctly protected the minimum catalog size, selected zero unsupported
services, emitted zero unsupported candidates, and failed the eligible-service, unsupported-count,
unsupported-candidate, and unsupported-coverage gates.

No class threshold should be relaxed and no previously exposed service should be reintroduced. Doing so
would change the meaning or freshness of the benchmark.

## Access and claim boundary

The inventory downloaded the pinned 2,680,136-byte shard once and automatically tokenized current turns
only to compute overlap counts. It emitted no utterance, token, slot value, history, or prompt. Manual
utterance inspection, model loads, model generations, API calls, adapter training, real service calls,
and external side effects were all zero.

V95 is source-feasibility evidence only. It is not novelty, abstention, calibration, posterior, planning,
or execution evidence.

## Correct successor

Freeze V95 unchanged. A fresh successor may use two independently pinned untouched shards: one supplies
the already validated three-service catalog construction and the other supplies one or more completely
withheld unsupported services. Both sources and their disjoint roles must be locked before either new
payload is opened. The successor must retain activation-only non-`NONE` cases, current-turn lexical
separation, two service-stratified hidden pairs, at least three declared pairs, and the existing class
count and coverage gates.
