# V107 Local Open-World Model Development Result

## Outcome

The pinned local Qwen3.8-27B 4-bit condition completed exactly once: one model load and 192 independent,
zero-temperature, no-retry generations over 128 observed development-evaluation records and 64 controlled
missing-observation records. Runtime was 2,351 seconds (39.2 minutes) and peak active memory was 17.32 GB.

The condition is nonqualifying. It passed 16 of 25 total quality/access checks, including all access
checks, confidence ECE, false-known acceptance, mean regret, regret relative to the best deterministic
baseline, and controlled abstention. It failed structured validity, macro F1, observed exact decisions,
known exact intent, novel recall and exact scenario routing, unsupported precision, and top-confidence
selective error.

Key development results:

- all-generation structured validity: 68.23% (`131/192`);
- observed-language structured validity: 52.34% (`67/128`);
- observed exact decision accuracy: 35.16%;
- status macro F1: 46.91%;
- exact known-intent accuracy: 0%;
- exact novel-scenario accuracy: 40.63%;
- unsupported recall: 100%, precision: 78.05%;
- novel recall: 40.63%, precision: 76.47%;
- false-known acceptance: 0%;
- 10-bin confidence ECE: 0.0766;
- top-confidence 80% error: 56.31%;
- mean shadow regret: `1.0977`, which is `0.0273` lower than ask-always's `1.125`;
- controlled missing-observation abstention: 100% (`64/64`);
- permanent true-hypothesis retention: 100%.

The class pattern is sharp. All 32 unsupported requests were correctly rejected. Thirteen of 32 novel-
valid requests were correctly routed as novel. No familiar or unfamiliar known request was resolved to
an accepted exact declared intent. The exact V105 validator found status-field invariant violations in
31/32 familiar-known, 21/32 unfamiliar-known, and 9/32 novel-valid responses, but none in unsupported or
missing-observation cases. Every invalid response became zero-confidence abstention as preregistered.

This gives a clean interpretation. The local LLM demonstrated useful open-world caution: it was excellent
at recognizing a wholly withheld scenario and at abstaining when no utterance existed. It also found some
hidden valid capabilities and slightly improved decision cost over always asking. But it did not reliably
ground ordinary supported requests to the exact typed catalog, and its confident subset remained too
error-prone. Low regret alone is therefore not sufficient to place it in the decision path.

## Boundary and decision

The protected test was never opened. There were zero API calls, adapter-training runs, manual utterance
inspections, real service calls, tool executions, or external side effects. Every model output remained a
non-authoritative shadow proposal; the complete 17-hypothesis universe was never pruned.

Freeze V107 as a useful negative development result and close this model branch. Do not run the protected
test, loosen the parser or gates after seeing the result, retry malformed rows, add an ensemble, substitute
an API model, or train an adapter under this protocol. The next scientific branch should study a revised
typed grounding interface or a richer sequential information-gathering problem under a fresh prospective
lock, while preserving deterministic abstention and complete hypothesis retention.
