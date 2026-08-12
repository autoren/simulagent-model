# V29 preregistration: posterior-marginal support graphs

V29 is one exposed-development, zero-forward comparison against V28. V28 established that
marginalizing graph uncertainty when selecting a program raises true-program top-1 accuracy
from 8/12 to 11/12 and passes the behavioral support gate, while exact support graphs remain
25/36. V29 tests whether conditioning every support graph on the single program MAP causes
that residual exact-graph failure.

V29 reconstructs exactly V27's locked graph candidates and probabilities. For each support
trace, it computes the exact posterior marginal of every graph by integrating over the shared
program and every other trace's graph. It emits the maximum-posterior graph for that trace,
breaking ties by the existing deterministic graph order. This is component-wise Bayes decoding,
not a credible union; one graph is still emitted per scene.

No model call, new score, fitted parameter, threshold, branch budget, score weight, ontology,
DSL, executor, query prediction, or fresh benchmark is permitted. The same V28 gates apply.

- Passing all gates authorizes a separately preregistered query-graph repair, not a fresh suite.
- Behavioral improvement without an exact-graph pass keeps the work in exposed development.
- No improvement shows that posterior decoding alone cannot repair the residual language score
  errors; LoRA remains unauthorized without a separate representation-level justification.
