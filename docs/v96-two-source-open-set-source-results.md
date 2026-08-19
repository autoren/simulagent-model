# V96 Two-Source Activation Open-Set Source Result

## Verdict

V96 is a clean negative result. Prospectively assigning one untouched shard to the catalog role and a
second untouched shard to the unsupported role did not provide sufficient service diversity. The
activation logic, role separation, and text-free boundary worked correctly, but the source topology did
not match the assumed shard-level division of labor.

The source gate failed, so V96 stops before population selection, language extraction, manual
inspection, or model access.

## Result

The catalog shard produced 499 structurally valid records and 102 intent activations, but only
`Events_1` was an eligible unexposed service. It therefore supplied one service, two supported pairs,
one hidden pair, one declared pair, 26 familiar-known candidates, eight unfamiliar-known candidates,
68 novel-valid candidates, and 48 insufficient-evidence candidates.

The unsupported shard produced 1,355 structurally valid records and 380 intent activations, but no
service was eligible after the prospectively frozen freshness exclusions. It supplied zero unsupported
services and zero unsupported candidates.

Only the activation/current-turn boundary, source-role disjointness, familiar count, novel count,
insufficient-evidence count, text-free inventory, and zero-access gates passed. The catalog, hidden-pair,
declared-pair, unfamiliar, unsupported, and associated coverage gates failed.

## Interpretation

V96 falsifies the assumption that two arbitrarily consecutive SGD shards can be assigned whole to the
catalog and unsupported roles. The shards are too service-concentrated. This is not evidence against the
five-class benchmark itself: V95 already showed that its catalog portion is feasible when three services
co-occur. It is evidence that role assignment must occur at the service level over a prospectively
locked aggregate source pool, not at the individual-shard level.

## Access and claim boundary

Both pinned shards were downloaded once, totaling 5,421,858 bytes. The inventory automatically
tokenized current turns but emitted no utterance, token, slot value, history, or prompt. Manual
utterance inspection, model loads, model generations, API calls, adapter training, real service calls,
and external side effects were all zero.

V96 is source-feasibility evidence only. It is not novelty, abstention, calibration, posterior, planning,
or execution evidence.

## Correct successor

Freeze V96 unchanged. A successor may pin an aggregate pool of untouched shards before any payload is
opened, pool their source-annotated activation records, and then hash-partition services into catalog
and unsupported roles. It must retain all V95 class rules and thresholds. It cannot pool already opened
V93-V96 language, weaken freshness exclusions, inspect language, or select services after seeing
language-derived features.
