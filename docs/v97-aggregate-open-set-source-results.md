# V97 Aggregate-Pool Open-Set Source Result

## Verdict

V97 is a clean negative result that closes the SGD development split as a source of fresh service-level
open-set evidence. The aggregate implementation worked, but the prospectively frozen freshness list
already contained every one of the 17 services in the development schema. Consequently, no service was
eligible for either the catalog or unsupported role.

The source gate failed, so V97 stops before population selection, language extraction, manual
inspection, or model access.

## Result

All twelve pinned shards were downloaded exactly once and passed byte-size and Git-blob identity checks.
The aggregate pool contained 15,330 structurally valid source records and 4,725 source-annotated intent
activations. Per-shard valid record counts ranged from 988 to 1,565.

The complete eligible-service set was empty because the frozen `previouslyExposedServices` list equals
the complete development schema service set. The partition therefore selected zero catalog services,
zero unsupported services, zero hidden or declared pairs, and emitted zero benchmark candidates. Only
the aggregate-shard, service-role disjointness, activation/current-turn boundary, text-free inventory,
and zero-access gates passed.

## Interpretation

The result is not evidence that the five-class benchmark is infeasible. V95 already showed that its
catalog construction works. V97 shows that repeated dev-split source audits have exhausted every service
namespace that can honestly be called fresh at the service level. More dev dialogue shards cannot repair
that problem.

The design audit should have checked whether any unexcluded schema service remained before authorizing
payload downloads. This omission did not compromise the result or expose language, but a successor audit
must make schema-set viability a pre-download noncompensatory check.

## Access and claim boundary

The twelve pinned payloads totaled 45,767,156 bytes. The inventory automatically tokenized current turns
but emitted no utterance, token, slot value, history, or prompt. Manual utterance inspection, model
loads, model generations, API calls, adapter training, real service calls, and external side effects
were all zero.

V97 is source-feasibility evidence only. It is not novelty, abstention, calibration, posterior, planning,
or execution evidence.

## Correct successor

Freeze V97 unchanged and do not reuse any SGD development service as fresh open-set evidence. A
successor must first metadata-pin and structurally audit a genuinely new typed schema split or a
different independently authored dataset. Before any dialogue payload download, it must prove that at
least four eligible service namespaces remain outside all prior exposures and that the source supplies
machine-checkable intent and `NONE` annotations. Only then may it reuse the validated V95 activation and
current-turn class construction.
