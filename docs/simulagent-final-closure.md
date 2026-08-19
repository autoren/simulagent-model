# Simulagent final research closure

**Effective date:** 2026-08-19  
**Status:** closed research program; retained as a read-only evidence and reference-architecture repository

## Decision

Simulagent is closed as an active experimental program.

The project has reached a useful stopping point: it established a coherent architecture for decision-making under
semantic uncertainty, mapped several conditions under which preserving uncertainty changes decisions, and identified
the boundary that its available evidence cannot cross. Continuing by adding more model sizes, prompts, synthetic
utterances, or fictional scenarios would not resolve that boundary.

Closure means:

- no further participant collection, terminal continuation, model run, API call, training run, ontology registration,
  trusted-state mutation, service action, or real-world execution is authorized;
- all frozen historical results and the prospective pilot's locked records remain immutable;
- maintenance, reproduction, and documentation may continue without changing scientific claims; and
- future work must begin as an explicitly authorized successor project, not as an automatic V225-style continuation.

This is a stopping decision, not a claim that unrestricted language understanding has been solved or proved
impossible.

## What the program established

The strongest contribution is architectural and mechanism-level:

1. **Keep uncertainty explicit.** Exact beliefs and version spaces can preserve alternatives that a point estimate or
   nearest-label decision would collapse too early.
2. **Ask only decision-relevant questions.** Information gathering helps when observations can distinguish hypotheses
   that imply different later actions. V70 supplied a positive action-dynamics example; V71 supplied the useful
   boundary case in which sensor-label uncertainty changed interpretation but not the best action.
3. **Separate proposal from authority.** An LLM can propose bounded candidates or clarification language, but a trusted
   controller must validate structure, preserve an outside/unknown state, and decide whether to plan, clarify, or
   defer.
4. **Make failure safe and explicit.** Typed clarification, reversible sandboxing, certificates, robustness checks, and
   terminal deferral compose into a credible non-executing decision architecture.
5. **Distinguish catalog reconstruction from speaker meaning.** Versioned artifacts can support retrospective ontology
   reconstruction. They do not, by themselves, reveal the intended meaning of a new speaker's utterance.

Across the local-model sequence, larger capacity and more reasoning did not supply the missing semantic authority.
Models often obeyed output schemas while still confusing known, novel-valid, insufficient-evidence, and unsupported
states. Bounded low-effort reasoning reduced avoidable truncation, but it did not repair this epistemic limitation.

## Final prospective-pilot disposition

The pilot produced one positive procedural result and two negative results:

- Phase 1 successfully collected and locked 16 participant-authored initial requests with zero assistant generations.
- The frozen Phase 2 local condition completed all 16 records but failed its preregistered usability gates: 13/16
  semantic proposals were structurally valid and one final continuation reached its token limit. The deterministic
  controller nevertheless routed all records safely, with zero retries, APIs, services, actions, or executions.
- An exploratory Phase 3 was opened for the 11 valid clarification outputs. It is now stopped after one locked response.
  That response used the explicit `unable_to_answer` path with reason `do_not_know`.

The Phase 3 failure is a protocol-design finding, not a participant failure. The scenarios were fictional and only
partially specified. Some model-generated questions requested additional world facts—such as resources, access, or
preferences—that neither the card nor a real external situation determined. The participant therefore had no
privileged observation to report. A fictional prompt can define an initial request, but it does not automatically
create an independent answer channel for arbitrary clarification questions.

The remaining 10 clarification responses will not be collected. No terminal plan or defer evaluation will be run.
Phase 2 remains negative, and Phase 3 cannot be used to claim clarification utility, task success, calibration, or
prospective open-world understanding.

## Claim boundary at closure

Simulagent supports the following claims:

- the component mechanisms can be implemented, audited, and composed in controlled settings;
- Bayes-adaptive treatment of uncertainty can matter when uncertainty is identifiable and changes consequential
  actions;
- uncertainty can also have zero decision value when the same action dominates across plausible interpretations;
- LLM proposal quality, output validity, abstention, and truncation are measurable interface properties rather than
  semantic ground truth; and
- an independent observation channel must actually contain information about the latent distinction being inferred.

It does **not** establish:

- unrestricted or general open-world language understanding;
- reliable prospective recognition of novel real-world capabilities;
- calibrated semantic likelihoods for arbitrary natural-language requests;
- successful clarification with real users across a population;
- safe autonomous ontology expansion; or
- safe real-world action or service execution.

## Preserved evidence

The canonical historical synthesis remains:

1. [Cross-track evidence synthesis](cross-track-evidence-synthesis-through-v224.md)
2. [Post-V224 stopping rule](research-stopping-rule-after-v224.md)
3. [Post-V224 roadmap](research-roadmap-after-v224.md)
4. [Model-free reference architecture](model-free-reference-architecture.md)
5. [Reference-architecture integration result](model-free-reference-architecture-integration-results.md)
6. [Dependency-drift provenance addendum](dependency-drift-provenance-addendum-through-v224.md)
7. [Prospective pilot Phase 2 negative result](prospective-language-pilot-v1-phase2-results.md)

The [unresolved-track registry](simulagent-unpursued-tracks.md) records what was not completed and what evidence would
be required to revisit it. The machine-readable closure is in
`outputs/simulagent-final-closure/result.json`.

## Successor-project rule

A future project may reuse the architecture, code, or lessons, but it must freeze its own question, population,
observation channel, baselines, and claims. The proposed low-resource translation-verification direction is such a
separate successor: it may borrow belief tracking, active evidence selection, abstention, and certificates, but its
translation data and human judgments would not be Simulagent evidence.

