# V108 Typed Interface Forensics Plan

V108 performs no generation and reads no selected language. It automatically reparses the exact frozen
V107 raw responses, joins them only to V101 text-free scenario/intent ground truth and the V106
evaluation membership, and emits aggregate categories without raw text or individual identifiers.

The diagnostic tests one preregistered ambiguity in V105's catalog: every intent exposes both a fully
qualified `intent_id` and a shorter `intent`, but `KNOWN` validation accepts only `intent_id`. A narrowly
defined counterfactual canonicalizer may replace an exact, uniquely mapped short intent with its qualified
ID and may remove a redundant scenario only when it exactly matches that resolved known intent. It cannot
change status or confidence, infer from language or gold, retry, or regenerate.

This does not alter V107. Its original metrics, gates, nonqualification, and sealed protected test remain
authoritative. If at least 75% of invalid observed outputs become structurally valid and counterfactual
known and overall exact accuracy clear the frozen thresholds, the successor should test a fresh
constrained typed serialization interface. Otherwise the next branch should move to sequential
clarification rather than treating formatting as the main bottleneck.
