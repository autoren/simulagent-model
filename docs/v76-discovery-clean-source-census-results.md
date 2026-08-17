# V76 discovery-clean source census result

## Bottom line

The prospectively locked metadata census found no eligible source pair and
therefore freezes a source-feasibility deferral. Fourteen unique candidate
repositories were recorded after 16 search queries. None passed all eight
provisional metadata gates, so no development source or protected confirmation
source was assigned.

This is the intended behavior of the V76 firewall. The project needs two
repository-disjoint families before implementation inspection; it does not need
to force an attractive-looking active-sensing task through a weakened gate.

## Search integrity

The V76 preregistration was committed and pushed as `293bb769147b5b6c2856667155d71ab7faf89639`
before the first search. The census used public search results, official project
and repository landing descriptions, official documentation surfaced by those
results, and GitHub repository/latest-commit metadata. It did not clone or
download a repository, open a candidate source/model/configuration/test/notebook
or data file, run a simulator or planner, inspect probability or reward arrays,
or compute an action, value, regret, EIG, mutual information, or policy outcome.

One search result unexpectedly printed official `pymdp` cue-chaining tutorial
shapes and configured sizes. That disclosure is recorded rather than hidden.
The candidate was marked development-exposed and ineligible; no repository file
was opened and no computation was performed.

## Candidate census

| Candidate family | Repository | Closest useful structure | Frozen exclusion |
|---|---|---|---|
| Entropy-guided foraging | [IB-POMCP](https://github.com/lsmcolab/ib-pomcp) | Belief-entropy-guided planning | No source-native reference/comparison action; not an apparent exact finite candidate |
| Memory process suite | [POPGym](https://github.com/proroklab/popgym) | Fast partially observable tasks | Landing metadata does not identify the registered sensing/reference/delayed-control conjunction |
| Continuous-control partial observation | [pomdp-baselines](https://github.com/twni2016/pomdp-baselines) | Hidden-state control | PyBullet/MuJoCo families exceed the exact finite envelope and have no reference pathway |
| Sensory scaffolding suite | [Scaffolder](https://github.com/penn-pal-lab/scaffolder) | Privileged sensing during training | Continuous locomotion/manipulation suite; no in-episode reference pathway |
| Active tactile classification | [tactile-mnist](https://github.com/TimSchneider42/tactile-mnist) | Agent-controlled tactile observations | Image-based classification, not finite delayed physical control with reference sensing |
| Active BCI command issuance | [POMDP-BCI](https://github.com/neuroergoISAE/POMDP-BCI) | Repeated observation before command | No repository license; offline calibration and human-data dependence |
| Pig-farm diagnostic treatment | [DecisionProgramming.jl](https://github.com/gamma-opt/DecisionProgramming.jl) | Diagnostic test before treatment | No source-described reference, calibration, or cross-sensor comparison pathway |
| Sequencing Goal-POMDP | [gpt-rewards](https://github.com/bonetblai/gpt-rewards) | Inspection with delayed order penalties | No repository license and no reference pathway |
| Cue chaining | [pymdp](https://github.com/infer-actively/pymdp) | Two cues lead to delayed reward | Closest structural candidate, but the disclosed `35×4×2` joint latent configuration exceeds 64 states; detailed search snippet also made it development-exposed |
| Active-inference maze | [ActiveInference.jl](https://github.com/ComputationalPsychiatry/ActiveInference.jl) | Small partial-state navigation | No state-informative sensing action or reference pathway in landing metadata |
| Confounded off-policy evaluation | [Confounded-POMDP-OPE](https://github.com/callmespring/Confounded-POMDP-OPE) | Partial-observation evaluation | No active-sensing source family |
| Earth-observation tool agent | [Earth-Agent](https://github.com/opendatalab/Earth-Agent) | Multiple sensing/analysis tools | LLM and 104-tool system is not exactly enumerable and violates the no-model-access boundary |
| Tiger/RockSample examples | [POMDPy](https://github.com/pemami4911/POMDPy) | Small discrete listen/check examples | Both domain families were already exposed in V72/V74 |
| Planner framework | [POMDPPlanners](https://github.com/yaacovpariente/POMDPPlanners) | Unified POMDP solvers | No qualifying source domain described by repository metadata |

Searches also returned prior-exposure hits including SARSOP, SBO_AIPPMS,
BetaZero, pomdp-py, and pomdp-solve. They were excluded before candidacy under
the committed registry.

## Decision

The deterministic role partition is empty because the eligible count is zero,
below the registered minimum of two repository-distinct families. V76 stops
before source implementation, structural validation, economic screening, or
any policy evaluation.

Do not broaden the exposure rule, increase the state/horizon envelope, add a
beacon or observation noise, drop the source-native reference requirement, or
reuse Tiger, RockSample, paint, maintenance, or the V71 protected set. A future
attempt requires a new preregistration justified by a materially different
scientific target or by the public release of genuinely new source-native
calibration-and-control benchmarks. Until then, the active-sensing empirical
branch should be reported and deferred.
