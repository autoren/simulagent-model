# V111 Existing-Evidence Novelty Separability Audit Result

## Outcome

V111 completed the preregistered aggregate-only census over the exact frozen V110 calibration and
evaluation membership. It enumerated 1,343 rules from ten fixed families over twelve registered features.
No model was loaded, no new response was generated, and no protected-test language, individual feature,
record identifier, source utterance, or raw model response was persisted.

Exactly one rule met the joint calibration constraints. The selected rule was `llm_abstain_only`: treat the
frozen local model's typed `ABSTAIN` output as evidence that a request may express valid novelty. On the 64
calibration records it found 8 of 16 novel-valid cases with 72.73% precision, 50% recall, and a 6.25%
non-novel false-positive rate.

The unchanged rule transferred to the separate 64-record evaluation subset. It found 8 of 16 novel-valid
cases, produced two false positives among 48 non-novel cases, and achieved:

- novelty precision: `0.80`;
- novelty recall: `0.50`;
- novelty F1: `0.6154`;
- non-novel false-positive rate: `0.0417`.

These values pass all three frozen noncompensatory gates. The result is deliberately narrow: the model did
not correctly label these cases as novel. Its reluctance to commit was statistically useful evidence of
novelty. `ABSTAIN` therefore must not be converted directly into an authoritative `NOVEL` decision. It may
only trigger retention of novel hypotheses or an information-gathering branch in a subsequently locked
policy.

The evaluation-label oracle found 13 feasible registered rules. Its best rule used a low retrieval score
and low top-two intent margin, reaching 71.43% precision and 62.5% recall with an 8.33% false-positive
rate. This is a diagnostic upper bound only; it was selected using evaluation labels and cannot authorize
or alter the transferred rule.

## Boundary and decision

V111 establishes that the frozen single-turn evidence is not completely devoid of novelty signal. It does
not establish a complete open-world classifier, exact novel-schema identification, calibrated downstream
action, or unrestricted open-world understanding. Half of the novel-valid cases were still missed, and the
selected signal alone does not distinguish valid novelty from every other reason to defer.

Freeze the positive separability diagnosis. The next authorized step is a separate preregistered full
development policy that uses `ABSTAIN` only as non-authoritative novelty evidence, retains the complete safe
hypothesis universe, and evaluates status accuracy, false-known acceptance, calibration, risk-coverage, and
exact-planner regret jointly. Keep the protected test sealed. Do not begin schema or mechanic induction,
sequential planning, API use, adapter training, capability creation, action authority, or execution.
