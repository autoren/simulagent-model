# V153 model-free question-order comparators results

## Verdict

V153 is a positive model-free comparator result over 96 development request fixtures and 120 sequential episodes. The oracle placed the discriminating question first on every episode, giving mean rank 1.0, mean decision cost 0.3, and improvement of 0.7 over safe no-query abstention.

Both fixed source order and the preregistered seeded-random order had mean rank 3.5 and mean cost 1.05, slightly worse than the 1.0 no-query cost. Source order placed exactly 20 episodes at each rank from one through six. Seeded random produced rank counts 21, 19, 18, 21, 23, and 18.

All three question-asking policies achieved 100% exact final states after the trusted answer. All 600 irrelevant intermediate questions failed closed to `A00`, every policy retained the complete authoritative hypothesis universe, and execution remained zero. No candidate-state proposal field was present. The census projected development metadata without conversation text and read no evaluation language; it used no model, API, or training access.

The future empirical target is now explicit. A local question-order heuristic adds value only if it substantially beats rank 3.5/cost 1.05 and approaches the oracle rank 1.0/cost 0.3 while final correctness remains guaranteed by the typed-answer firewall.

The passing decision is:

`freeze_question_order_comparators_authorize_local_question_order_protocol_design_only`

No model run or evaluation access is authorized by this result.
