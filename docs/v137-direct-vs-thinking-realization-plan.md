# V137 Direct-versus-thinking Realization Plan

## Question

Does Qwen3.8-27B's thinking mode materially improve the controlled open-world boundary that the frozen V132
direct condition could not realize, when ambiguity and clarification are observable by construction?

## Locked comparison

Use one pinned Qwen3.8-27B 4-bit snapshot on the untouched V135 synthetic test split. The 100 fixtures contain
twenty complete five-stage groups: forty clear requests, twenty genuinely ambiguous requests, and forty
clarification-resolved conversations. Run exactly two deterministic conditions on every fixture:

- direct inference with thinking disabled and 32 output tokens;
- thinking-enabled inference with a 512-token budget.

The system and semantic prompt, catalog, candidate, conversation, temperature, and one-sample/no-retry rule
are otherwise identical. Direct output must be exact JSON. Thinking output must end in exact JSON following a
closed thinking block. Raw answers and reasoning traces are never persisted or manually inspected; only the
parsed typed choice, validity, hashes, token counts, trace-presence flag, and runtime are retained.

## Qualification

Each condition is assessed independently. It must reach 95% overall, clear, ambiguous-abstention, and
clarified accuracy; 99% structural validity; at least 80% fully correct five-stage groups; no more than 5%
false-known answers on non-known truths; and bounded candidate-attracted error. Its counterfactual sequential
policy must cost at most 0.70 on the two latent sides of every ambiguous group on average, improve at least
0.25 over not asking, keep right-truth false-known decisions at most 5%, and produce safe non-known decisions
at least 95% of the time. Thinking must actually emit a thinking trace on at least 95% of fixtures; direct
must emit none.

All outputs remain non-authoritative shadow evidence with the complete safe universe retained and zero
execution. A passing condition authorizes only a separately designed external-language transfer study. No
outcome opens V134, permits retries or prompt repair, establishes independence or human equivalence, or
authorizes APIs, induction, training, authority, actions, or execution.
