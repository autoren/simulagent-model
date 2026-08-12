# V11 frozen-scale polarity capacity preregistration

## Question

V10 showed that frozen Qwen3.5-0.8B layer-6 representations solve the complete
grounding pipeline on new contexts expressed with familiar constructions, but
fail current-state polarity when language families, state lexicons, mechanics,
or operators are omitted from training. The failure persists with oracle spans
and oracle temporal status.

V11 changes only frozen backbone capacity. It asks whether the same polarity
relation becomes linearly accessible in Qwen3.5-4B or Qwen3.5-9B.

## Fixed V10 inputs and evaluation

V11 reuses, byte for byte:

- the V10 corpus, targets, split assignments, and 24 folds;
- all 3,492 base prompt texts and 6,984 hypothesis-conditioned prompt texts;
- mean, evidence-span, and NLI-final pooling definitions;
- the `mean_direct`, `evidence_span_direct`, and primary `nli_final` heads;
- `C=1.0`, class balancing, seed zero, float32 pooled features;
- all four oracle-span/temporal ablations;
- every fold, surface, paired-consistency, symbolic, flip-pair, and complete-
  intervention-group gate; and
- the exact deterministic symbolic evaluator.

No prompt, target, threshold, head, fold, or gate may be changed after either
larger backbone is accessed. Both larger models run regardless of the 4B result,
giving a fixed three-point capacity comparison rather than a conditional search.

## Checkpoints and homologous depth

The checkpoints and immutable Hugging Face revisions are:

- `mlx-community/Qwen3.5-4B-4bit` at
  `0e7ffd5c629ef7719d4cbc04069232580bfa9d9c`;
- `mlx-community/Qwen3.5-9B-4bit` at
  `8b2b98c00a6b4d291155e4890773ca8f769aee53`.

All models use affine 4-bit MLX weights. V10 extracted layer 6 of 24, exactly
25% through the 0.8B text stack. V11 therefore extracts the nearest integer to
25% depth: layer 8 of 32 for both 4B and 9B. This rule is fixed before feature
access. No alternative layer is extracted or selected.

## Execution and firewall

For each model, one frozen pass extracts all three representations together.
One complete 24-fold evaluation then fits the unchanged linear heads. The 4B
and 9B feature artifacts, heads, and results are saved separately and combined
only after both evaluations finish.

V11 permits zero adapter runs, zero final-mechanic evaluations, and no access to
Tone Drift, V3 test records, prior holdouts, untouched V8 mechanics, or V7 model
results. V10's completed result is the only model result used to authorize V11.

## Decision rule

- If 4B and 9B both pass all primary gates, conclude that transferable polarity
  emerges by 4B under this interface. A future 0.8B LoRA compression experiment
  may be proposed but is not authorized.
- If 4B fails and 9B passes, conclude that the tested frozen capability has a
  threshold above 4B and prefer 9B for the grounding component.
- If 4B passes and 9B fails, stop for an inconsistency audit; do not interpret
  the result as monotonic scaling.
- If both fail oracle-polarity or NLI-consistency gates, frozen scale alone is
  insufficient. Test a separately locked nonlinear/token-aware frozen readout
  before LoRA.
- If oracle polarity passes but full-pipeline gates fail, revise only the failed
  evidence-matching or temporal component.

No V11 outcome directly authorizes LoRA or final evaluation.
