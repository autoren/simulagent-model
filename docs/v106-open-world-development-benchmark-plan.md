# V106 Open-World Development Benchmark Plan

V106 is a development-only deterministic benchmark. It hash-splits the 256 frozen MASSIVE development
records into 32 calibration and 32 evaluation records per class without using language. Only after this
design is frozen may code automatically read development utterances. The protected-test artifact remains
unopened.

The comparison includes complete safe enumeration, ask-always, a fixed identifier-token grammar, a
character n-gram TF-IDF nearest-neighbor baseline trained only on the 12 declared MASSIVE training
intents, and an oracle. Retrieval's two thresholds are selected only on the calibration half by a frozen
cost function and tie-break. Evaluation reports status, exact intent/scenario, selective calibration,
false acceptance, and counterfactual decision-regret metrics. All actions are shadow calculations.

The 64 frozen development missing-observation controls are evaluated separately. Their authoritative
runtime result is always abstain-and-ask; no source utterance is supplied in this condition. Complete
enumeration retains every safe hypothesis and therefore cannot lose the truth even when a proposer fails.

Passing V106 means only that the deterministic benchmark and safety controls are valid. It authorizes a
separate implementation audit and one development run of the already local Qwen3.8-27B 4-bit snapshot.
This single high-capacity challenger replaces a small/large ensemble: prior capacity experiments did not
justify extra model complexity, and the new question is whether one capable model adds semantic utility
over deterministic controls. No API, adapter training, protected-test access, action authority, or real
execution is authorized.
