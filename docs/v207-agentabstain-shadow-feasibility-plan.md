# V207 AgentAbstain shadow feasibility plan

## Question

Can the pinned AgentAbstain release support a deterministic, paired, pre-execution act-versus-abstain study in which a local model sees task language but cannot call tools or execute anything?

V207 answers only whether such a population can be selected safely. It does not open task language or evaluate a model.

## Separate claim track

AgentAbstain is not an external implementation of the V205 semantic POMDP. It has paired abstention behavior and irreversible-action cases, but no source-documented calibration likelihood model. A future result would therefore measure behavioral restraint under controlled perturbations, not Bayesian posterior calibration or open-world ontology learning.

## Allowed evidence

The audit pins the dataset repository revision and may read:

- GitHub and Hugging Face tree paths/object metadata;
- the YAML front matter of the dataset card, excluding its prose body;
- the source `tasks.yaml` task-ID set configuration; and
- Python schema/type and evaluator files under the prospectively allowed prefixes.

It persists only paths, hashes, byte counts, and extracted schema identifiers. It may not fetch a dataset task payload or read any instruction, example, dialogue, rationale, expected response, or task description.

## Gates

The nonlanguage evidence must identify at least 40 complete pairs, including at least 20 pre-execution pairs across two scenarios. Pair side, pair identity, pre-execution status, and gold act/abstain decision must be available without an LLM judge or task-text inspection. Schema fields must separate identity, label, scenario, prompt, and rationale. The future subset must be selectable before task text and evaluable in shadow mode with no tool calls or execution.

The source's July 2026 public release predates the August 2026 V195 local-model evaluation, so a future study cannot
claim contamination-free evidence merely from chronology. It must freeze source and model revisions, report the
contamination uncertainty, retain the source's paired perturbation control, and forbid training or fine-tuning on the
benchmark.

## Conditional successor

If every gate passes, V207 authorizes only a separate deterministic extraction preregistration. That later design must freeze exact identifiers, prompt projection, label firewall, costs, parser, and model/runtime condition before task language is opened. A negative result closes the source at metadata level without weakening the gates.
