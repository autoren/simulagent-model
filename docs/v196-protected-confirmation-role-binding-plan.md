# V196 protected confirmation role binding plan

## Purpose

V195 passed on fresh development language. V196 chooses the confirmation population without opening confirmation
utterances. It records why a newly sampled all-`dev` role is unavailable and binds a stricter subset of the SGD role
that V183/V184 froze and sealed before V195.

## Source decision

After excluding every V183 and V191 dialogue, one contract has zero unused `dev` dialogues. Mixing source partitions
after seeing V195 would add an avoidable contract-by-partition confound. Instead, use the pre-existing V183 protected
role, but exclude every protected record whose dialogue also appears in V183 development or V191. Where multiple
protected records share one remaining dialogue, choose exactly one by a fixed salted identifier hash.

This metadata-only rule leaves 113 observed records across all 14 contracts and 12 pre-existing missing controls. It
must retain at least two observations per contract, contain 70 known, 34 provisional, and 9 unsupported records, and
have zero dialogue overlap with either development source.

## Boundary

V196 may read source identifiers, roles, hidden contract labels, candidate metadata, and artifact envelopes. It may
not read or emit protected utterances, slots, frames, predictions, scores, or outcomes. It runs no model or policy.

A pass authorizes only a separate lock for the unchanged V195 policy on the selected protected records. It does not
authorize immediate protected-language access, model generation, API use, ontology changes, authority, action, or
execution.

