# V76 active-sensing mechanism synthesis

## Bottom line

V71–V75 form a coherent falsification sequence, not five interchangeable
benchmarks. The sequence supports a bounded mechanism claim: in the locked
finite-horizon exact models, preserving the joint posterior over a persistent
unknown observation-label codebook can materially improve posterior-expected
control value over fixed MAP and persistent posterior-sampling controls when
calibration is non-harvestable, the point models retain common support, the
information changes a later state-dependent control decision, and a
prospectively fixed adaptive policy is economically worthwhile after sensing
and delay costs.

V74 supplies positive development evidence and V75 supplies
outcome-untouched external-domain replication. V75 is not discovery-clean
confirmation because V68 had already inspected a malformed variant of the
classic paint family. The next scientific objective is therefore not another
convenient example. It is a source- and repository-disjoint confirmation whose
discovery and role assignment are locked before implementation inspection.

## Evidence ladder

| Stage | Role | Exact result | What it establishes |
|---|---|---|---|
| V71 | Fallback-free negative development boundary | 21/21 records valid; exact, MAP, posterior sampling, open loop, and myopic controls agreed; maximum normalized MAP regret `0` | Sensor-semantic uncertainty and common support do not create decision value when one action remains dominant |
| V72 oracle | Engineered implementation oracle | Exact value `2.6`; MAP and persistent posterior-sampling value `-6`; normalized point-control regret `0.0955556`; zero fallback | The intended calibrate-inspect-contingent-control mechanism is implementable; this fixture is not scientific evidence |
| V72 external | Harvestable-reference negative boundary | Exact, MAP, posterior sampling, and open loop all chose the known-reward route; value `13.786875`; normalized point-control regret `0` | Explicit sensing and delayed loss are insufficient when control can bypass sensing |
| V73 | Prospective structural/economic negative boundary | Fixed adaptive value `-118.9418` versus open loop `-119.4345`; normalized advantage `0.000402659`, below `0.005` | Non-harvestability, information gain, sensor separation, and threshold crossing are insufficient without material economic value |
| V74 | Positive source-grounded development | Exact `5.609355`; MAP and persistent posterior sampling `-44.20125`; normalized regret `0.158746`; exact-over-open-loop normalized advantage `0.0224225` | The complete mechanism passes a one-shot development test after prospective economic screening |
| V75 | Positive outcome-untouched external-domain replication | Exact `0.166398`; MAP and persistent posterior sampling `-0.00428688`; normalized regret `0.0230042`; exact-over-open-loop normalized advantage `0.0224264` | The mechanism transfers to a separately pinned valid external domain without prior policy-outcome access, but not without prior domain-family exposure |

All V71, V72, V74, and V75 point controls stayed on support, and V73 stopped
before any point-control or optimal-planner outcome. Fallback therefore does not
explain either positive result or the preceding negative boundaries.

## What changed across the sequence

The stages isolate a conjunction of requirements.

1. V71 supplied uncertainty and common support, but observations did not alter
   the best control.
2. V72 supplied an active sensor and delayed state-dependent reward, but its
   known-good reference was itself harvestable, so sensing was unnecessary.
3. V73 removed that bypass and demonstrated information and threshold
   crossing, but sensing was too costly relative to control value.
4. V74 prospectively required the fixed adaptive lower bound to clear the
   materiality threshold before implementing an optimal planner; the exact
   posterior policy then separated strongly from both point controls.
5. V75 retained that economic gate and replicated the separation under a new
   pinned source implementation and source-native sensing fidelity.

This does not prove that every listed condition is mathematically necessary in
all POMDPs. It shows that the weaker constructions tested here were empirically
insufficient, while the complete registered construction was sufficient twice.

## Best defensible claim

> In two prospectively screened, source-grounded finite-horizon models with a
> persistent unknown observation-label codebook, a joint-posterior exact
> planner selected materially better policies under the locked joint model than
> fixed MAP and persistent posterior-sampling controls. The separation appeared
> only after the design made calibration non-harvestable, control-relevant, and
> economically valuable relative to open loop. One result is development
> evidence and the other is outcome-untouched external-domain replication; no
> discovery-clean independent confirmation has yet been completed.

“Better” means higher finite-horizon posterior-expected value under the locked
joint model, source dynamics, reward function, prior, and horizon. It is not an
observed-return, asymptotic, safety, or real-world claim.

## Claim boundary

The sequence does not establish that:

- Bayes-adaptive planning universally outperforms point-model planning;
- active sensing, mutual information, non-harvestability, or threshold crossing
  is sufficient in isolation;
- V74 or V75 is an unchanged external benchmark—the latent codebook and
  calibration layer were project-authored;
- V75 is discovery-clean confirmation;
- approximate inference, SMC², longer horizons, continuous spaces, learned
  representations, human interaction, or natural language preserve the effect;
- the locked posterior is calibrated to an unrestricted real environment; or
- the result establishes safety or real-world performance.

No stage in this sequence used human data, performed a model forward pass,
trained an adapter, or ran SMC².

## Frozen decision

V71–V75 are closed. Their sources, partitions, reliabilities, observation
noise, calibration construction, costs, horizons, controls, thresholds, and
gates may not be changed or reused for discovery-clean outcomes. V71's
protected models remain unopened, and no additional V71–V75 policy value may be
computed.

The only authorized empirical successor begins with the separately frozen V76
metadata-only source census. It must exclude every previously exposed
repository and domain family, lock a deterministic repository-disjoint
development/confirmation partition before opening implementation files, and
defer rather than relax its gates if no eligible pair exists.
