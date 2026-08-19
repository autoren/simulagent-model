# V221r1 parser-config repair plan

The initial V221 runner stopped during catalog construction because the locked runtime config did not contain the
`parserDesign` object required by the inherited asserted-state function. Development public/truth JSONL files had been
loaded once, but no catalog, observation, residual, summary, or result was written and no candidate method ran.

V221r1 is a narrow implementation repair. It injects an exact copy of the already-frozen V220 `parserDesign` into an
in-memory copy of the V221 config. It does not modify the V221 design lock or change the role manifest, catalog
semantics, method portfolio, scores, budgets, controller, safety gates, residual definition, input hashes, or branch
logic. The repaired evaluation writes to a separate V221r1 output directory.

Protected JSONL files remain hash-only and unloaded. The repair authorizes no network, model, training, registration,
mutation, action, or execution.

