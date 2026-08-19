# V85 Local Adversarial Generator Results

V85 is a frozen negative result. The one authorized 24-record local run completed with one model load,
24 generations, and no retries. All responses were parseable JSON, no forbidden authority field was
emitted, every output retained permanently untrusted provenance, and no output was deployable.

The model did not meet the registered usefulness gates. Schema-valid question rate was `0.625`, useful
strict-invalid rate was `0.5`, only nine distinct questions were produced, and per-schema useful rates
ranged from `0.0` for inventory to `1.0` for file lifecycle. Novelty beyond V84's deterministic mutation
set was exactly `0.5`. Three defect categories were observed, but that did not compensate for the failed
noncompensatory gates.

An independent stricter post-outcome diagnostic found one important validator blind spot. For an
operation-only request, the model emitted the exact operation alternative plus the single unrequested
option `Alex Chen`. V84's registered validator forbade only the complete other-slot fragment
`Alex Chen or Alex Kim`, so it classified this semantically wrong question as content-valid. Provenance
still prevented deployment, demonstrating why the provenance gate is load-bearing.

Freeze V85 without prompt edits or rerun. Do not use its outputs to modify the already frozen V84 suite.
The correct successor is a model-free validator correction that rejects every individual option surface
from an unrequested slot, with exhaustive partial-injection mutations, before considering any further
model experiment.
