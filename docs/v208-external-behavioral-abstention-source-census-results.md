# V208 external behavioral-abstention source census result

## Outcome

V208 passed every integrity and access gate but found zero eligible external sources. The six independently pinned candidates were OverSearchQA, AbstentionBench, AbstainEQA, HiL-Bench, ClarifyCodeBench, and Ask or Assume. AgentAbstain was not reopened.

The census read six GitHub repository objects, six recursive repository trees, three declared Hugging Face dataset HEADs, and three card-data metadata objects. It read no README or other blob body, license text, Hugging Face tree or payload, task instruction, example, dialogue, rationale, response, model output, or execution trace.

## Candidate findings

- **OverSearchQA** is the closest text-only source. It declares 1,188 examples balanced 594/594 and an explicit `should_abstain` field across three scenarios. It does not declare matched pairs or an explicit pair-identity field, and GitHub reports its code license as `NOASSERTION`.
- **AbstentionBench** supplies a direct abstention label across six text-only scenarios and a pinned CC-BY-NC-4.0 dataset. It is a heterogeneous collection rather than a matched-pair design, declares no pair identifier, does not document balance for this protocol, and GitHub reports its code license as `NOASSERTION`.
- **AbstainEQA** has the strongest true paired construction: 1,636 unanswerable cases paired with 1,636 originals across five categories, with MIT metadata. The allowed metadata exposes no explicit pair-identity field, and the task depends on visual episode history or embodied exploration rather than a text-only pre-execution shadow decision.
- **HiL-Bench** has 200 public tasks with baseline, full-info, and ask-human modes. Its blockers are intentionally discovered through progressive code/database exploration, so it is a runtime interaction benchmark rather than the required pre-execution text-only decision. It also lacks a direct deterministic act/abstain gold field for this protocol, uses model-mediated help, and exposes no recognized code or dataset license metadata.
- **ClarifyCodeBench** has 419 human-annotated underspecified code tasks, but no matched answerable control population, only one broad scenario for this census, and LLM-judge matching for clarification questions. It therefore cannot independently score the binary paired decision.
- **Ask or Assume** links full and hidden SWE tasks by identifier, but the decision unfolds inside code execution with a GPT-simulated user. It lacks a direct deterministic act/abstain field, has only one broad scenario for this census, and exposes no recognized code or dataset license metadata.

## Interpretation

The external literature now supplies several useful components, but no source combines all of them in a form that supports this project's intended controlled study. Text-only datasets tend to offer independent answerable/unanswerable examples without matched pair identity. Strong paired agent benchmarks tend to require visual, code, database, tool, or simulated-user interaction and often score behavior with an LLM judge.

This is a source-availability negative, not an LLM performance result. It does not imply that the candidates are poor benchmarks for their own purposes. It means they cannot provide the particular independent gold and no-execution firewall required here without relabeling, synthetic pairing, hidden-content inspection, or a changed scientific question.

## Decision

Freeze V208 negative. Do not open candidate task payloads, synthesize pair IDs, infer act/abstain truth from task text, or use an LLM judge as gold. Park the external behavioral-abstention confirmation track. No local-model run is authorized from V208.

The next research goal returns to model-free decision/interface work. The highest-value successor is to connect the positive V205 semantic-POMDP mechanism to a richer, explicitly generated language-observation channel where semantic likelihoods are known by construction, while keeping any later LLM as a non-authoritative observation source and retaining exact defer/clarify/act control.
