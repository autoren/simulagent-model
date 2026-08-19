# V90 Local Capacity and Generation Results

V90 is a clean negative capacity study. All four pinned local conditions completed their one allowed
model load and 48 independent generations on the same fresh, sealed SGD population. There were no API
calls, retries, adapter-training runs, manual utterance inspections, service calls, or external side
effects. Every model remains an offline, non-executable shadow component.

| Condition | Active intent coverage | Intent-set exact | NONE-only exact | State-key recall | State-key exact | Elapsed | Peak active memory |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3.5 4B, 4-bit | 0.8750 | 0.8125 | 0.7917 | 0.9701 | 0.2292 | 81.6 s | 3.53 GB |
| Qwen3.5 27B, 4-bit | 0.7917 | 0.8958 | 1.0000 | 0.7771 | 0.2500 | 370.6 s | 16.74 GB |
| Qwen3.8 27B, 4-bit | 0.8750 | 0.8750 | 1.0000 | 0.9531 | 0.2500 | 441.3 s | 16.74 GB |
| Qwen3.8 27B, 8-bit | 0.8750 | 0.9167 | 1.0000 | 0.9514 | 0.2917 | 641.9 s | 30.10 GB |

No condition passed every preregistered noncompensatory gate. All four missed the required `0.90`
active-intent coverage and `0.50` exact accumulated state-key rate. Qwen3.8 27B 4-bit additionally
missed the mandatory-`NONE` gate on three records. The 8-bit condition repaired that formatting
invariant and modestly improved exact candidate sets, but it missed the same three active records as
the 4-bit Qwen3.8 condition and more than doubled the latter's incremental memory and runtime cost.

The paired contrasts rule out a simple scaling story. Moving within Qwen3.5 from 4B to 27B improved
intent-set exactness by `0.0833` and NONE-only exactness by `0.2083`, but reduced active-intent coverage
by `0.0833` and state recall by `0.1931`. On the 24 active records, the 4B condition alone covered three
cases that the 27B condition missed, while the 27B condition alone covered only one. Moving from
Qwen3.5 27B to Qwen3.8 27B recovered two active cases and much of the state recall, but introduced the
mandatory-`NONE` failures. Moving Qwen3.8 from 4-bit to 8-bit changed no active-intent coverage outcome.

The preregistered small-plus-large union diagnostic was not eligible. It required an independently
qualifying 27B condition, and there was none. No union, cascade, fallback, or retrospective model
selection was executed. The observed error diversity is reported only as paired shadow evidence; it
does not justify combining two unreliable proposers into an authoritative candidate set.

Freeze V90 with the model-free boundary intact. Keep Qwen3.5 4B only as the low-cost historical shadow
baseline when a model comparison is scientifically necessary. Do not adopt a 27B model, 8-bit
quantization, small/large cascade, API fallback, or adapter training for the runtime system. The
authoritative path remains the typed schema renderer, deterministic validator, and exact Bayesian
decision core. Any later LLM branch needs a genuinely different, narrower role whose failure cannot
remove the true intent or corrupt accumulated state; increasing parameter count or precision is now a
closed explanation for this interface on the frozen population.
