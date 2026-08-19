# V113 Historical Known-Disagreement Rescue Census Plan

V112's novelty evidence transferred, but its full policy missed known accuracy by two records and the
top-confidence error gate by roughly three retained errors. V113 treats the full V112 transfer population
as historical policy-design evidence. It generates no new model output and makes no additional transfer
claim.

Only requests for which the typed LLM proposed `KNOWN` and deterministic retrieval chose a different exact
intent are eligible for rescue. Nine fixed, simple rule families use the proposed-intent retrieval score,
nearest score, their gap, LLM confidence, and scenario agreement. A rescued item emits the original typed
known intent with a frozen 0.75 action confidence. LLM abstentions, unsupported decisions, the novelty flag,
all safe hypotheses, and every V112 metric and gate remain unchanged.

Candidates are selected against all seventeen original V112 quality gates. If none passes, this rule family
closes. If one or more pass, one deterministic rule may be frozen for a separately locked evaluation on a
new disjoint population. V113 cannot authorize protected-test access, induction, another model run, APIs,
training, action authority, or execution.
