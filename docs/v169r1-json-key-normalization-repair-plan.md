# V169r1 JSON key-normalization repair plan

The V169 population and every scientific gate passed. Its nominal verifier compared a JSON-loaded summary, whose
object keys are strings, with an in-memory `Counter` projection whose class-coverage keys are integers. The only
mismatch is `{"1", "2", "3"}` versus `{1, 2, 3}`; JSON serialization necessarily maps both to the same object.

V169r1 preserves the original run, failed audit, membership, eligibility, artifacts, gates, and decision. It
independently reconstructs V169 and compares all persisted values after canonical JSON round-trip normalization.
It creates no nominal V169 outcome and performs no planner scoring, model access, registration, action, or
execution. Passing authorizes only a separate locked run of the unchanged V167 planner.
