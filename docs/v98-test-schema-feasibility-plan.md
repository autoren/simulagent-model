# V98 SGD Test-Schema Feasibility Plan

## Purpose

V97 exhausted every service in the SGD development schema. V98 tests whether the separately published
official SGD test schema contains enough genuinely new typed service families to support the controlled
open-set benchmark. This is a schema-only stage: no test dialogue payload may be opened.

Freshness is defined at the service-family level, using the identifier prefix before the final numeric
version suffix. A test service such as `Banks_1` is not fresh if any development service from the
`Banks` family has already been exposed. At least four test services from four service families absent
from the complete development schema must remain, and each must define at least two typed intents and
one slot.

## Boundary

The pinned test schema may be downloaded once and parsed automatically. The inventory may emit service
identifiers, family identifiers, and intent/slot counts, but not intent names, descriptions, slot names,
surface tokens, or dialogue language. Passing authorizes only preregistration of a pinned test-dialogue
source pool and its text-free activation census. It does not authorize dialogue download, selected
language extraction, manual inspection, model access, API use, training, planning, or execution.
