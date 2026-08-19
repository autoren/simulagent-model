# V221 initial attempt failure

The first V221 runtime attempt terminated before catalog construction completed with `KeyError: parserDesign` in the
inherited `semantic_state` function. The V221 config omitted the parser field list even though the catalog design
explicitly required the frozen V220 parser semantics.

The role manifest and design audit already existed. No catalog manifest, observation, residual manifest, summary, or
result was written. Candidate-method evaluation count was zero. Development public/truth inputs had each been loaded
once before the failure; protected JSONL bodies and models were not loaded. The correction is isolated in the
prospectively audited V221r1 repair and does not alter the locked scientific design.

