# V32 factorized polarity and scope

## Verdict

Scientific factorization gate: `fail`. Intermediate-supervision gate: `fail`. Absolute selected system: `none`.

## Full sealed evaluation

| System | Predicate | Relation order | Truth | Exact fact | Exact scene | Absolute pass |
|---|---:|---:|---:|---:|---:|:---:|
| monolithic | 0.252 | 0.145 | 0.384 | 0.091 | 0.033 | no |
| auxiliaryDirect | 0.289 | 0.207 | 0.521 | 0.141 | 0.028 | no |
| factorizedCompiled | 0.289 | 0.207 | 0.486 | 0.141 | 0.028 | no |

## Composition holdout

Factorized sign accuracy: 0.603.
Factorized operation accuracy: 0.385.
Factorized compiled truth accuracy: 0.408.
Factorized minus auxiliary-direct exact-fact delta: +0.040 (family-bootstrap 95% interval [+0.001, +0.080]).
Auxiliary-direct minus monolithic exact-fact delta: +0.041 (family-bootstrap 95% interval [+0.010, +0.075]).

## Decision integrity

One V28 replay authorized: `false`.
Seed selection: `none`.
Checkpoint or hyperparameter selection: `none`.
Post-result integrity audit: `pass`.

## Interpretation

V32 is a valid negative result for the registered one-epoch interface, not a decisive rejection
of semantic factorization. Neither learned artifact fit its own training population adequately.
A bounded post-result diagnostic using only the already-saved fit features found:

| Training artifact (three-seed mean) | Predicate | Relation order | Truth | Exact fact | Sign | Operation |
|---|---:|---:|---:|---:|---:|---:|
| monolithic | 0.268 | 0.157 | 0.536 | 0.139 | 0.714† | 0.143† |
| joint auxiliary | 0.325 | 0.217 | 0.735 | 0.232 | 0.682 | 0.696 |

† The monolithic sign and operation heads were intentionally unsupervised and are not scientific
measurements; their values reflect their fixed random predictions and class frequencies.

On the composition holdout, joint-auxiliary exact atom recovery was only 0.304–0.335. Predicted
sign with the oracle operation compiled to the correct truth on 0.460–0.679 of records; oracle sign
with the predicted operation reached 0.553–0.614; and both predicted components were correct on
only 0.200–0.263. Thus weak component and atom learning, not the deterministic compiler, set the
dominant ceiling.

The two positive paired effects remain descriptive evidence worth preserving. Auxiliary
supervision improved composition exact fact over monolithic by +0.041, and deterministic
compilation improved it over the identical auxiliary-direct artifact by +0.040. Both bootstrap
intervals excluded zero, but both effects missed their preregistered material-effect thresholds
and the absolute component-accuracy requirements by large margins.

## Next research stage

Do not reuse the opened V32 evaluation strata for model or hyperparameter selection. Use only the
exposed V32 fit/calibration representations for a development-stage adequacy study:

1. establish learning curves and require high fit plus unseen-family calibration accuracy before
   another sealed evaluation is allowed;
2. test independently optimized atom, lexical-sign, and outer-operation modules to locate
   representation, optimization, and multitask-interference limits;
3. preserve the supervision-matched auxiliary-direct versus factorized-compiled decoding control;
4. after freezing a development-qualified interface, generate a fresh sign-by-scope suite with
   new surface families and constructions for the next one-shot test.

V28 remains closed because no V32 system passed the absolute language-interface gates.
