# V138 Thinking-Parser Contract Audit Results

## Result

V138 confirms a technical contract mismatch in V137. The pinned Qwen template places the opening
`<think>` tag in the prompt when thinking is enabled. A normal generated suffix therefore closes a trace it
did not open itself. V137 counted tags only inside that suffix and required equal opening and closing counts,
so it rejected the canonical output shape.

All preregistered synthetic contract checks pass. The corrected parser starts at thinking depth one, accepts
one transition to closed depth followed by exact JSON, and still rejects a missing close, extra trace tags
after closure, invalid JSON, unknown choices, and any trace in direct mode. Direct parsing remains unchanged.

The frozen V137 metadata also matches exactly: 93 of 100 thinking outputs were labeled
`unclosed_thinking_trace`, seven were labeled `invalid_final_json`, 93 recorded a thinking tag, and seven
reached the 512-token maximum. No response or trace text is present or was read.

## Interpretation

V137's thinking scores cannot be interpreted semantically and cannot be recovered retrospectively. V138
does not prove that those 93 outputs contained valid JSON after their closing tags; raw text was intentionally
not retained. The seven maximum-length outputs also remain genuine possible budget failures.

The only justified successor is one fresh, separately frozen comparison on the unused V135 development
split using the corrected parser. V137 and V134 remain untouched. External language, APIs, training,
induction, authority, action, and execution remain closed.
