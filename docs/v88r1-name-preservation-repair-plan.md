# V88r1 Mechanical Name-Preservation Repair Plan

V88r1 is a single mechanical retry of the execution-inconclusive V88 census. The first attempt stopped
after one unobserved and unscored generation because the scorer row omitted the generic harness's
separate fixture `name` field. No scientific output was available when this repair was registered.

The only change is to copy `record["name"]` unchanged into the scorer result before the harness identity
check. The source, selected 48 records, corpus bytes, prompts, local model, decoding, parser, scorer,
controls, thresholds, non-deployable provenance, and all no-execution rules are inherited byte-for-byte
from the original V88 implementation lock.

The retry may load the local model once and generate once for each of the same 48 records. Including the
failed attempt, the disclosed cumulative ceiling is two model loads and forty-nine generations. There is
no API, training, manual language inspection, live service call, or side effect. A malformed, failed, or
negative result cannot trigger another retry.
