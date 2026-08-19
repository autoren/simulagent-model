# V126 SGD Retrieval-selectivity Results

## Outcome

V126 failed the preregistered transfer gates:

> `retrieval_status_band_fails_cross_dataset_selectivity_close_current_trigger_inventory`

The run automatically processed all 4,881 locked training and 576 locked evaluation utterances in memory.
The frozen retrieval bands assigned 478 records to clarification, 66 to high-similarity known, and 32 to
low-similarity unsupported. Thus the trigger skipped 98 records, or 17.01%, without any fitting.

The skip was pointed in the wrong direction. Across all priors and correlations:

- queried records had average clarification value from 0.4810 to 0.5461, above the 0.30 cost;
- skipped records had average clarification value from 1.9171 to 1.9650, more than six times the cost;
- selective regret ranged from 1.1820 to 1.2361;
- query-all regret ranged from 0.8988 to 0.9609;
- ask-always regret was 1.1667.

The selective policy was therefore worse than query-all in every condition and even worse than ask-always.
It retained acceptable unsupported correctness (at least 89.15%) and false-known safety (at most 6.33%),
but known exact probability missed 80% under all uniform-prior conditions and reached only 77.55%--78.98%.

The same-skip-fraction oracle diagnostic achieved 0.8392--0.8908 regret. This shows that saving 17% of queries
is not intrinsically impossible; frozen retrieval status bands simply do not rank clarification value. Raw
retrieval alone was also poor: 23.44% known exact, 13.02% unsupported correctness, and 1.3863 regret.

This is a cross-dataset finite-population result under a simulated 95%-correct clarification channel, not a
claim about real human or model clarification. It nevertheless closes the only LLM-independent semantic
signal family found in V122 for the current trigger role. V126 may not be retuned, mined, or used to fit new
thresholds. Language models, protected access, induction, APIs, training, authority, and execution remain
closed. Any successor requires a genuinely new pre-query evidence mechanism, not another threshold on the
same nearest-neighbor similarity.
