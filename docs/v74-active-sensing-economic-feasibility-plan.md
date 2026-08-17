# V74 prospective economic value-of-information plan

## Purpose

V73 established that sensor separation and control-threshold crossing do not by themselves create a material planning problem. V74 therefore screens the economics before an adapter or optimal evaluator exists. The fresh development source is the MIT-licensed `h2r/pomdp-py` Tiger problem at commit `bd0e4392247aebfe9a95b449275237dcc25e7737`. Its source provides two hidden states, explicit listening and door-opening actions, symmetric reset dynamics, noisy state-dependent observations, a `+10` safe-open reward, a `-100` wrong-open reward, and a `-1` listening cost.

The source exposes observation noise as a constructor parameter. V74 prospectively fixes it at `0.01` (`p=0.99`) rather than using the `0.15` default. This is a deliberate high-fidelity development configuration chosen before any adapter, policy optimization, regret, or planner outcome. The discount `0.95` is taken from the repository's Tiger value-function and conversion examples.

## Fixed project-authored layer

The future adapter may add one non-harvestable `calibrate-beacon` action. The beacon has known `tiger-left` condition, uses the configured Tiger observation channel, costs `-0.5`, applies the source listening transition to the target, and supplies no control reward. A latent binary codebook either preserves or reverses the two source observation labels. The target listen action retains the source `-1` cost. The three-action horizon and all parameters are fixed here.

The lower-bound policy is not optimized: calibrate once, listen to the target once, then use the parity of the two labels to open the inferred safe door. With independent accuracy `p`, the joint codebook-and-state decision is correct with probability `p^2+(1-p)^2`. Its discounted value includes both sensing costs and the full delay before opening.

## Comparator and decision rule

The prospective best open-loop comparator is three beacon actions. Under the uniform state prior and symmetric transition/reset rules, any unconditioned door opening has expected reward `-45`; target listening costs `-1`; and the beacon costs `-0.5` without changing the marginal target belief. Thus no open-loop action sequence can improve on repeated beacon use. This algebraic argument avoids implementing or searching an adapter.

The fixed policy must be strictly positive, beat the open-loop comparator by at least `5.0` raw return, and exceed `0.015` of the frozen three-step return scale with at least `0.005` additional normalized margin. Failure stops V74 before adapter code or any exact Bayes-adaptive, MAP, posterior-sampling, myopic, or protected-source evaluation. A pass authorizes only preregistration and implementation-stage validation.

This is source-grounded development evidence, not an unchanged external Tiger result. The beacon, latent codebook, reduced beacon cost, and nondefault observation noise remain explicit project-authored choices.

References: [pomdp-py](https://github.com/h2r/pomdp-py), [Tiger source](https://github.com/h2r/pomdp-py/blob/master/pomdp_py/problems/tiger/tiger_problem.py).
