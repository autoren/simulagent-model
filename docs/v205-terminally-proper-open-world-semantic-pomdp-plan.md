# V205 terminally proper open-world semantic POMDP plan

## Question

Can an exact planner benefit from retaining an explicit outside-semantics hypothesis and safely defer on histories where semantic evidence remains unusable, once every control commitment is guaranteed to receive its delayed state-dependent consequence?

## Why this is separate from V204

V204 allowed a zero-immediate-reward repair on the last finite-horizon step without forcing the later settlement. V205 does not change or rerun V204. It preregisters a different fixed-stage process with a structural terminal contract:

1. before calibration, the agent may calibrate a known reference, inspect the unknown target directly, repair, or defer;
2. after calibration, it may inspect the target, repair, or defer;
3. after inspection, it must repair or defer;
4. every repair transitions through an automatic, unavoidable correct-or-wrong settlement outside the controllable decision count; and
5. every unfinished sensing history receives the safe-deferral terminal value.

There is therefore no controllable action that can postpone settlement past the episode boundary.

## Frozen uncertainty

The target condition is `A` or `B`. Sensor semantics are `CANONICAL`, `REVERSED`, or `OUTSIDE_UNKNOWN`. Calibration observes a known `A` reference; inspection observes the target. All hypotheses assign positive probability to every observation. The outside hypothesis emits the same distribution for either condition, so its observations cannot identify the correct repair.

The priors are explicit project-authored mechanism weights, not LLM ranks, probabilities, or confidence.

## Comparators

The single exact evaluation compares the full open-world policy against a closed-world policy that deletes the outside hypothesis, forced commitment without deferral, MAP certainty-equivalence, persistent posterior sampling, the best observation-blind action program, an immediate-reward myopic policy, and immediate deferral. Every policy is evaluated under the same full true mixture.

## Noncompensatory gates

V205 passes only if all structural, policy, regret, terminal-accounting, normalization, and access gates pass. In particular, exact planning must calibrate at the root, inspect after red and blue calibration evidence, defer after green, reach both repairs on other histories, and differ from the closed-world policy after green. Mandatory settlement and safe terminal deferral must each hold on every applicable path, with zero unsettled repairs or horizon escapes.

No parameter, stage, reward, likelihood, comparator, threshold, or decision rule may change after exact values or actions are opened.

## Boundaries

V205 reads no language and uses no LLM, API, training, protected evidence, ontology registration, trusted-state mutation, service, side effect, action, or execution. A positive oracle authorizes only a separate metadata/source feasibility design for an external analogue; it does not authorize candidate evaluation.
