# V201r1 elapsed-time verification repair plan

V201's scientific run completed, but its first independent freezer rejected two exact-comparison checks. The cause is
localized: the evaluator copied `elapsed_seconds` into its scientific summary, then the runner performed one final
progress write a few milliseconds later. The final access ledger therefore has a slightly larger elapsed time.

V201r1 is preregistered after that failure and may repair only comparison semantics for this one volatile top-level
field. It must prove that:

- the original failed audit has exactly two false checks: evaluation-summary and result reconstruction;
- the rebuilt and persisted summaries differ only in `elapsed_seconds`;
- the final access time is no more than 0.01 seconds later and never earlier;
- replacing the rebuilt volatile time with the persisted aggregation-time value makes the summary exact;
- the original result derives exactly from the persisted aggregation summary and final access gates; and
- all fixtures, normalized proposals, scored rows, scientific metrics, qualification gates, the negative decision,
  model outputs, and source artifacts remain byte-for-byte unchanged.

No language or raw response is inspected. No model, policy, parser, score, or gate is rerun. A pass preserves V201's
negative presentation-invariance finding and authorizes only a roadmap update plus a separate model-free
decision-sufficiency design.

