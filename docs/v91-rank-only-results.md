# V91 Rank-Only Structured-LLM Results

V91 is a clean negative utility result and a positive safety-boundary result. The pinned Qwen3.5 4B
model completed its one allowed load and 64 deterministic generations on the fresh sealed population.
There was no new weight download, API call, retry, training run, manual utterance inspection, pruning,
early stopping, belief update, action selection, service call, or external side effect.

The safety architecture worked exactly as intended. All 64 model responses parsed as JSON and used only
allowed intent identifiers. Deterministic completion preserved the full schema intent set on all 64
records, retained `NONE` on all 64, and preserved the authoritative-state fingerprint on all 64. The
model was permanently non-deployable and non-executable. Independently, all 480 permutations of the
five V79 hypotheses preserved the exact posterior-aware root action, optimal-action set, Q values, and
policy value. There were zero action mismatches, zero execution-certificate violations, and maximum
absolute numerical error `1.7763568394002505e-15`.

The ordering itself did not qualify:

- active top-1 accuracy was `25/32 = 0.78125`;
- `NONE` top-1 accuracy was only `5/32 = 0.15625`;
- overall top-1 accuracy was `30/64 = 0.46875`;
- overall top-2 recall was `38/64 = 0.59375`;
- mean reciprocal rank was `0.66667` and mean gold rank was `1.9375`;
- only `10/64 = 0.15625` raw outputs contained the requested complete permutation.

The best preregistered non-oracle deterministic control was the identifier exact-match grammar. Its MRR
was `0.75260` and mean gold rank was `1.60938`, so the model's MRR improvement was `-0.08594` and its
mean-rank reduction was `-0.32813`. The grammar recognized only `4/32` active records at rank one, but
it correctly prioritized all 32 `NONE` records; the model's active strength did not compensate for its
failure to distinguish inactive states. Lexical overlap reached `0.75` active top-1 but ranked no `NONE`
record first. These complementary deterministic errors are descriptive only; no unregistered ensemble
or revised control was evaluated.

Freeze V91 without prompt changes, threshold relaxation, record replacement, retry, deterministic-model
ensemble, larger model, API comparator, or adapter training. The local model is not justified as an
authoritative candidate generator or as a search scheduler. Retain complete deterministic schema
enumeration, mandatory `NONE`, immutable state, and exact posterior-aware planning. The rank-only
canonicalizer remains a useful fail-closed architectural pattern, but there is no evidence that this
model adds enough ordering utility to deploy or integrate it.
