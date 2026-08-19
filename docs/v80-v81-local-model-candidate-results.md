# V80–V81 Frozen Local-Model Candidate Results

## Result

Two prospectively locked, one-shot local-model experiments tested whether a frozen
`mlx-community/Qwen3.5-4B-4bit` snapshot could safely sit above the verified V79 decision
mechanism as a structured interpretation proposer. Neither experiment used an API, adapter,
human-language record, real tool, or external side effect. Each used one local model load and 24
deterministic generations with no retry.

V80 asked the model to return a canonical variable-length list of all plausible interpretation
IDs plus `none_of_the_above`. Every response was parseable JSON and every response retained the
escape candidate. The model emitted no probability, confidence, action, or tool fields. However,
only 87.5% of responses satisfied the complete schema, exact candidate sets were correct on 62.5%,
mean gold recall was 86.1%, mean candidate count was 3.67, and none of the four out-of-ontology
requests was rejected exactly. The direct candidate interface therefore failed.

V81 used a materially different interface and 24 fresh requests. The model returned five fixed
Boolean compatibility labels; deterministic code alone composed the V79 candidate IDs. This solved
all eight clear records and all four fully ambiguous records exactly, and bounded the mean candidate
count at 2.46. It still failed the registered population gates: schema validity was 87.5%, exact
label-vector and candidate-set accuracy were each 58.3%, mean label accuracy was 75.8%, and
out-of-ontology label accuracy was 0%. Bare `Alex` was especially unreliable, and three logically
inconsistent label vectors caused the safe parser to discard all candidates.

Independent implementations reparsed every raw response, reconstructed every score and aggregate,
reproduced every gate, and verified the one-load/24-generation/zero-external-access histories. V80
and V81 are frozen negative results and must not be prompt-tuned, rescored, or rerun on their locked
populations.

## Interpretation

The failures do not show that LLMs cannot be used with the decision framework. They show that this
particular frozen local model and zero-shot structured interface do not meet the reliability needed
to define the decision core's hypothesis set. Good performance on explicit clear requests does not
compensate for missing ambiguous recipients or failing to identify unsupported operations.

The protected architectural boundary worked as intended: malformed or inconsistent semantic output
never became a probability, action, tool call, or side effect. The Bayesian planner was not invoked
with fabricated confidence, and `none_of_the_above` remained an explicit hypothesis whenever the
schema was valid.

The next LLM study, if pursued, must be materially different and non-authoritative. Suitable roles
include generating surface wording from an already chosen clarification template, producing
synthetic adversarial language before a new corpus is sealed, or serving as an optional comparator.
An LLM must not re-enter candidate definition, belief assignment, clarification selection, or tool
execution without a new preregistration and a stronger validation mechanism. An API remains optional,
not required; these results used only a pinned local model.
