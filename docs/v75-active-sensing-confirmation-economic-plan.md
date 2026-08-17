# V75 outcome-untouched active-sensing replication plan

## Purpose

V74 established the mechanism in a source-grounded Tiger development model. V75 asks whether the same qualitative result survives in a different control system before any new policy is optimized. The selected source is the MIT-licensed NOVA repository at commit `fa6f0bf038509cb7bb94fb79e38e691c6e6d83e9`, using its valid four-state `paint_95.pomdp` model. The source supplies a manufacturing state machine, `paint/inspect/ship/reject` actions, a noisy `0.75/0.25` inspection channel, delayed state-dependent control rewards, reset dynamics, and discount `0.95`.

This is an outcome-untouched replication, not a discovery-clean confirmation. V68 parsed a malformed five-state POBAX variant of the same classic paint problem and excluded it for an observation-normalization defect before computing any policy outcome. V75 neither repairs that protected file nor uses it. It selects a separately pinned, normalized four-state source whose policy values, information gains, and regrets have never been evaluated in the project. The limitation is frozen before outcomes.

## Prospective project layer and lower bound

The only added action is a non-harvestable zero-reward reference inspection of a known nonblemished condition. It uses the source's `0.75/0.25` sensor accuracy and an identity target transition. A binary latent codebook either preserves or reverses the source `NBL/BL` labels. Target inspection, state transitions, rewards, and discount remain source-grounded. Source control observations are deterministic and belief-uninformative, so the adapter may collapse them to a separate `none` symbol without changing target-state inference.

The fixed lower-bound policy is registered without optimization: inspect the reference, inspect the target, paint then ship when the labels match, and reject when they differ. With source accuracy `p`, the paired state-and-codebook decision is correct with probability `p^2+(1-p)^2`. The calculation includes the paint failure probability, the asymmetric source rewards after painting, and the full action delay. The comparator is the best observation-independent sequence: zero. Sensing and painting have zero immediate reward; an unconditional reject has zero expectation at the reset prior; shipping is negative before painting and never positive in expectation after unconditioned painting; and either control resets the prior.

The fixed policy must exceed `0.1` raw return and `0.015` of the four-step source reward span, with at least `0.005` additional normalized margin. Failure stops V75 before an adapter or exact Bayes-adaptive, MAP, posterior-sampling, myopic, or regret computation. A pass authorizes a separately frozen evaluator and only one outcome run.

## Rejected source candidate

The fresh MIT-declared `abaisero/gym-pomdps` shopping model had appealing query/navigation economics, but its coordinate observations are deterministic. Under a reversed shared codebook, routine movement labels are impossible under the wrong point model, so MAP and posterior-sampling controls would require off-support policy completion. It is excluded before policy evaluation rather than weakening the common-support standard.

References: [NOVA](https://github.com/kylewray/nova), [pinned paint source](https://github.com/kylewray/nova/blob/fa6f0bf038509cb7bb94fb79e38e691c6e6d83e9/tests/benchmarks/algorithms/domains/paint_95.pomdp), [gym-pomdps](https://github.com/abaisero/gym-pomdps).
