# V209 controlled probabilistic language-observation POMDP plan

## Question

Can exact open-world planning use controlled probabilistic utterances to decide when to clarify, act, or safely defer, while retaining an explicit outside-semantics hypothesis and accounting for every delayed consequence?

## Separation from prior evidence

V205 established the corresponding model-free sensor-codebook mechanism. V208 found no fresh external behavioral-abstention source that satisfied the paired deterministic source gate. V209 is therefore neither an external confirmation nor a model evaluation. It preregisters a new, project-authored finite grammar as an exact stochastic observation channel.

No external task text is read. No LLM maps the surface strings. The planner receives only semantic observation IDs sampled from frozen normalized likelihood tables. Text surfaces are deterministic renderings used solely to state and test the eventual interface boundary.

## Frozen process

The hidden task state is `A` or `B`. The semantic regime is `CANONICAL`, `ALTERNATIVE`, or `OUTSIDE_UNKNOWN`. At the root the controller can ask how a known reference is described, ask directly about the target, commit to `A` or `B`, or defer. After a reference answer it can ask the target, commit, or defer. After a target answer it must commit or defer.

The reference and target questions have distinct normalized utterance channels. Target likelihoods after the reference are a frozen one-percent mixture with a history-specific anchor, making the channel genuinely history dependent without changing the intended V205-scale mechanism. Clarification cost is an expected hidden-state/regime-dependent reward with a small history offset after the reference.

Every control commitment receives an automatic correct-or-wrong settlement. A clarification cannot escape the horizon: unfinished clarification receives the safe-deferral terminal value.

## Anti-artifact controls

The same semantic observations have two one-to-one surface families. Replacing every direct surface with its matched paraphrase must change neither value nor policy because surface text is outside the planner.

A frozen permutation swaps the `ALPHA` and `BETA` observation IDs and the corresponding channel axes. After mapping branch keys back, the exact policy and all values must be identical. This tests label-renaming invariance rather than semantic competence.

## Comparators and gates

One locked exact census compares the full policy with a closed-world policy that drops `OUTSIDE_UNKNOWN`, forced commitment, MAP certainty equivalence, persistent posterior sampling, the best observation-blind action program, a myopic immediate-reward policy, and immediate deferral. All comparators are evaluated in the full true mixture.

V209 passes only if all structural, terminal, normalization, anti-artifact, policy-reachability, value, regret, and access gates pass. The exact policy must ask the reference at the root, ask the target after informative `ALPHA` or `BETA` replies, defer after an unresolved reply, and eventually reach both state-specific actions and deferral. No parameter or gate may change after exact values are opened.

## Boundaries and conditional successor

V209 is a mechanism result about a finite synthetic language channel. It cannot establish natural-language understanding, empirical model likelihoods, calibrated confidence, human behavior, or open-world ontology acquisition.

If V209 passes, it authorizes only a separately preregistered fresh controlled-language population and a deterministic observation-projection contract. That successor must freeze population generation, held-out partitions, semantic truth, surface counterfactuals, projection/scoring, and contamination controls before any records are opened. A local model remains a later, separately locked challenger and never becomes the authority or planner.
