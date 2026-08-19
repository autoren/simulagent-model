# V85 Local Adversarial Generator Plan

V85 is the first post-V84 local-model test and uses the model in a permanently non-authoritative role.
The model receives only a frozen typed schema, clarification target, and adversarial profile. It never
receives an original instruction, belief, policy value, tool state, or executable authority.

The 24-record census crosses four schemas, three typed targets per schema, and two profiles. The pinned
local MLX model is loaded once and generates exactly one deterministic response per record, without
retry. Raw artifacts and access counters are written durably after every record.

Every generated output has `local_model_adversarial` provenance and is non-deployable regardless of its
content. A positive result requires parseable schema-valid questions, a high rate of strict semantic
defects, diversity beyond V84's sixteen deterministic mutations, and at least two independently detected
defect categories. A negative result is frozen without prompt editing or rerun.

V85 can only establish whether this local model adds useful offline test inputs. It cannot authorize
generated wording, candidate generation, belief assignment, action selection, API access, training, or
tool execution.
