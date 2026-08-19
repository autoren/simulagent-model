# V134 Semantically Non-colliding Source Design Plan

## Question

Can an untouched SGD partition support the same complete 66-cell clarification census while ensuring that
selected novel intent names do not collide with the six declared choices?

## Frozen text-free design

Use the SGD development partition, whose dialogue language has not been used by this branch. Relative to the
train schemas, freeze six declared choices from rental cars, weather, events, and travel. Freeze three novel
domain composites from banks, flights, and media, one unsupported alarm composite, and `A00`.

Select four fixtures for every truth-by-presented-known-candidate cell. The 264 fixtures comprise 240 unique
source identifiers and 24 missing controls. Selection uses only identifiers and structural schema labels.

Read only train/dev schemas to confirm zero selected record collision by normalized intent name and exact
full definition signature. Emit hashes and aggregate collision counts, never raw descriptions, dialogue
language, slot values, or model outputs.

## Boundary

Passing creates a frozen future benchmark asset only. V133 does not authorize extracting the selected dev
utterances or rerunning the local model. No prompt, model, protected data, induction, richer planning, API,
training, authority, or execution is opened by this audit.
