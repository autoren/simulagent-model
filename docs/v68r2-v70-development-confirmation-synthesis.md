# V68r2–V70 development-to-confirmation synthesis

## Bottom line

The V68r2–V70 sequence establishes a prospectively separated result for one
project-authored uncertainty construction. V68r2 correctly rejected an
insufficiently decision-relevant command-channel family on four exposed
development models. V69 then qualified a materially revised dominant latent
action-remapping family on those development models without scoring any
confirmatory model. V70 froze that V69 family and evaluation logic before
evaluating a sealed census of 244 records from nine previously untouched,
externally sourced POMDP base models. The V70 evaluation retained every record
and passed all 22 frozen gates.

The resulting claim is narrow but consequential: under the locked finite
latent-remapping family and finite-horizon exact planner, retaining the exact
posterior over possible remappings can change the first action and improve
posterior-expected control value relative to committing to a single MAP model.
The model-level replication criterion held in four of nine confirmatory models,
including two structurally related and two novel models.

## Decision trail

| Stage | Scientific role | Census | Primary evidence | Frozen decision |
|---|---|---:|---|---|
| V68r2 | Negative development screen of the repaired command-channel family | 59 records / 4 exposed models | 0 BA/MAP root-action disagreements; 0 material MAP-regret records; maximum normalized MAP regret `0.0015135820075061503` | Stop the unchanged family before scoring any confirmatory model |
| V69 | Development-only screen of the materially revised dominant remapping family | 59 records / the same 4 exposed models | 8 BA/MAP root-action disagreements; 8 material MAP-regret records; 16 material posterior-sampling records; maximum normalized MAP regret `0.027595609725981726` | Authorize prospective confirmatory preregistration only |
| V70 | Confirmatory evaluation of the unchanged V69 family and frozen model-level hierarchy | 244 records / 9 untouched models | 4 paired-MAP qualifying models; 2/3 related and 2/6 novel; 6 models with material posterior-sampling regret; maximum normalized MAP regret `0.09544903686417067` | Confirm multi-environment replication for the project-authored V69 family |

V68r2 and V69 evaluate different uncertainty families. Their regret values are
therefore not a before/after treatment comparison, a learning curve, or evidence
that performance improved through tuning. Their legitimate role is procedural:
the first family failed on development data and stopped; the revised family
passed on development data and alone advanced to a separately locked
confirmation.

## What replicated

V70 applied one fixed family, prior, planner, point-control semantics,
materiality thresholds, and reporting hierarchy across the nine sealed models.
Four models met the paired criterion of an exact Bayes-adaptive/MAP first-action
disagreement and material MAP regret on the same record:

| Evidence class | Model | Stratum | Qualifying records | Interpretation |
|---|---|---|---:|---|
| Fallback-free | `4x3.POMDP` | Structurally related | 6 | The MAP control never entered its totalization rule on any record |
| Complete locked procedure | `fully_observable_tmaze2.POMDP` | Structurally related | 1 | The qualifying effect overlaps the frozen MAP totalization diagnostic |
| Complete locked procedure | `hallway.POMDP` | Novel | 6 | Broad procedure-level replication, but the qualifying effects overlap totalization |
| Fallback-free | `network.POMDP` | Novel | 5 | The strongest clean novel-model evidence; all five disagreements were material |

The two fallback-free models show that the central phenomenon cannot be reduced
to the handling of histories that are impossible under a selected point model.
They do not, by themselves, meet the preregistered requirement of three
qualifying models. The full four-model confirmation therefore supports the
complete locked comparison, while the stronger fallback-independent
interpretation is limited to `4x3.POMDP` and `network.POMDP`.

Across the complete census, exact Bayes-adaptive and MAP first actions differed
on 60 records. Eighteen records combined a first-action disagreement with MAP
regret of at least `0.005`; 44 records had material MAP regret and 52 had
material posterior-sampling regret. These counts are descriptive. The frozen
scientific decision was model-level, not a pooled-record significance claim.

## Secondary and negative evidence

The terminating and nonterminating cheese models form a Tier B stress pair.
Each had nine material MAP-regret records, but neither had an exact BA/MAP first-
action disagreement under the frozen tie semantics, and every material effect
overlapped point-control totalization. They do not qualify the primary claim.
Their similarity supports only a limited robustness observation about terminal
handling.

The two cheese models are Tier B only and must remain outside every headline
count of models that replicated the paired decision-selection effect.

`fully_observable_tmaze5.POMDP` and `heavenhell.POMDP` showed no qualifying MAP
effect. `shuttle.POMDP` had one first-action disagreement, but its normalized
MAP regret was below the materiality threshold. This heterogeneity is part of
the result: posterior retention was consequential in some locked environments,
not universally advantageous on every model or record.

## Numerical and procedural integrity

- The V70 evaluation retained all 244 records and passed all 22 frozen gates.
- The maximum primary-versus-convergence normalized value error was
  `1.2454736823139355e-15`.
- Source validation, finite-metric, belief-normalization, census-completion, and
  convergence-action rates were all `1.0`.
- No development model was rescored during V70, and no confirmatory record or
  model was selected, rejected, replaced, or rescored after locking.
- V68r2 and V69 each scored zero confirmatory models.
- No stage in this sequence ran SMC², accessed human records, performed a model
  forward pass, or trained an adapter.
- The independent V70 outcome audit reproduced every aggregate and gate from
  the sealed rows and verified the complete hash and one-shot chain.

The roughly 5.5-hour V70 runtime demonstrates a concrete computational
bottleneck for this implementation and census, especially on `hallway.POMDP`.
It is not a scaling law and should not be presented as evidence about the
asymptotic cost of exact Bayes-adaptive planning.

## Best defensible claim

> Across nine sealed, externally sourced POMDP base environments, a
> prospectively frozen Bayes-adaptive planner using the exact posterior over a
> project-authored latent action-remapping family made materially better
> finite-horizon decisions than locked point-model controls in four
> environments, including two structurally novel models. Two qualifying models
> demonstrated this effect without invoking the point-control fallback rule.

Here, “better” means higher finite-horizon posterior-expected value under the
locked family, prior, source model, and reward function. It does not mean higher
observed return in an unrestricted real environment.

## Claim boundary

This sequence does not establish that:

- Bayes-adaptive planning always outperforms MAP or posterior sampling;
- the finite remapping family spans all plausible environment dynamics;
- the external benchmark authors supplied or endorsed the uncertainty family;
- approximate inference, including SMC², preserves the V70 decision gains;
- the result extends to long horizons, large or continuous state spaces, or
  unrestricted real-world control;
- fallback-involved effects isolate point-estimate collapse independently of
  the complete locked point-policy procedure; or
- the nine V70 models may now be reused as development data without forfeiting
  their confirmatory status.

## Frozen provenance and next authorization

The three source outcomes are bound by:

- `configs/v68r2-development-outcome-lock.json`
- `configs/v69-development-outcome-lock.json`
- `configs/v70-confirmatory-outcome-lock.json`

This synthesis is a report over those completed outcomes. It authorizes no
rerun, rescore, threshold revision, model replacement, or retrospective tuning.
The V69 family and all V70 models are closed for development.

The next defensible program is boundary testing under a new preregistration:

1. define a materially different uncertainty family before viewing outcomes;
2. select fresh development and protected confirmatory base models;
3. keep fallback-free and complete-procedure evidence as separate registered
   estimands;
4. treat approximate inference, longer horizons, and real interaction as
   separate validation stages rather than consequences of V70; and
5. preserve V68r2, V69, and V70 as an immutable negative-development,
   positive-development, and positive-confirmation sequence.
