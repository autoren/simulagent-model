# V138 Thinking-Parser Contract Audit Plan

## Question

Did V137's thinking condition fail semantically, or did its frozen parser contradict the pinned Qwen chat
template's division of the reasoning trace between prompt and generated suffix?

## Frozen evidence

V138 may read only:

- the immutable V137 outcome and aggregate per-fixture metadata that exclude response text and traces;
- the frozen V137 parser source;
- the pinned model manifest and its hashed `chat_template.jinja`;
- synthetic parser test strings written before this audit.

It may not read, reconstruct, or regenerate any V137 response, inspect V134 or external language, load a
model, or change V137's result.

## Contract

For `enable_thinking=true`, the pinned template ends the prompt with `<think>\n`. The generated suffix is
therefore parsed with initial thinking depth one. It must close that depth exactly once, contain no later
thinking tags, and end in the exact one-key JSON answer. For direct inference, the template supplies an
empty closed trace in the prompt and the generated suffix must be exact JSON with no reasoning tags.

The audit must demonstrate on preregistered synthetic strings that the V137 parser rejects the canonical
prompt-opened suffix while the corrected stateful parser accepts it, and that malformed or unclosed forms
remain invalid.

## Decision

Passing establishes a technical measurement defect only. It cannot recover V137's thinking answers because
their text was intentionally not stored. It authorizes at most a separately frozen direct-versus-thinking
comparison on the unused V135 development split with the corrected parser. V137 test reuse, V134 language,
external transfer, APIs, training, induction, authority, action, and execution remain closed.
