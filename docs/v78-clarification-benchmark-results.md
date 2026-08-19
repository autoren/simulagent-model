# V78 Clarification-and-Tool-Use Benchmark Results

## Verdict

V78 is a clean model-free benchmark-design failure with a narrowly localized
cause. Nineteen of twenty preregistered gates passed. The sole failure was
`unknown_has_safe_unknown_continuation`.

The result does support the core bounded mechanism:

- ambiguous intent began with `ask_recipient`, used both focused questions,
  and reached certified interpretation-specific execution;
- MAP interpretation incurred normalized regret `0.2134415432` on the ambiguous
  fixture, while act-immediately incurred `0.0569600617`;
- the clear fixture executed `execute_schedule_chen` immediately, and ask-always
  incurred normalized regret `0.0656319796`;
- the dominant control chose `safe_preview`, with zero MAP and posterior-sampling
  regret;
- every exact policy respected complete-belief execution certification;
- transition, observation, belief, and shared-support checks all passed;
- no model, API, adapter, human record, real tool, or external side effect was
  accessed.

It does not authorize local-model inference because the benchmark failed its
noncompensatory gate vector.

## Why the unknown continuation gate failed

The unknown-heavy fixture correctly began with clarification and never executed
an irreversible action on an unknown-indicating branch. However, after repeated
`*_other` observations, the horizon-one policy selected another focused question
instead of `safe_preview` or `abstain`.

This is not evidence that repeated questioning is genuinely optimal. It exposes
a missing terminal utility in the finite-horizon model:

- asking costs `-1.0` or `-1.2`;
- safe preview costs `-2.5`;
- abstention costs `-5.0`;
- reaching the horizon with the task still unresolved costs `0.0`.

At the final decision step, the benchmark therefore rewards asking and then
silently expiring more than explicitly choosing a safe resolution. The gate was
designed to catch exactly this failure mode and did so.

## Audit status

The durable harness wrote all four raw fixture artifacts before gate aggregation.
An independent implementation recomputed every exact root action, value, root
Q-value, MAP regret, posterior-sampling regret, act-immediately regret, and
ask-always regret. It reproduced the complete gate vector and confirmed that the
unknown safe-continuation gate was the only failure.

V78 is frozen. Its priors, rewards, channels, population, gates, evaluator, and
outcome must not be changed or rerun.

## Correct successor

A separately named successor may add an explicit terminal utility for an active,
unresolved belief at horizon exhaustion. That is a semantic correction to the
decision problem, not a retrospective relaxation of the V78 gate. The terminal
penalty must be preregistered before any successor policy is computed and should
make unresolved silent expiration worse than explicit abstention.

Even if that successor passes, it should authorize only a frozen local-model
candidate-generation protocol. It would still not establish natural-language
understanding, calibrated LLM probabilities, safe real-tool execution, human
usability, or API-model necessity.
