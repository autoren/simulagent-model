# Research roadmap after V177

## Confirmed result

V175 developed an exact planner whose objective matches the trusted routed system. V177 strongly confirmed that
unchanged mechanism on a separately frozen four-constraint population with zero exact target-context overlap with the
development population. Across both populations, the planner safely completes trusted alias and composition cases,
defers provisional cases, and dominates the recommendation-oriented V167 planner plus immediate, greedy, random, and
fixed-query controls.

The supported claim is now:

> Within the fixed finite ontology and deterministic observation model, exact certification-aware adaptive planning
> lowers routed loss while preserving a unanimous deterministic authority gate and reversible sandbox boundary.

This is not evidence for unseen concepts, open-world language, noisy observations, ontology misspecification, real
services, or deployment.

## Research tracks

### Track A — open-world language/model proposals: dormant

Keep local and API models out of the active robustness sequence. The confirmed result is an exact decision mechanism,
not a repair for the language branch's boundary-recognition and calibration failures. Reintroduce language only later
as an explicitly untrusted observation/proposal channel, with a preregistered utterance-level identifiability protocol.

### Track B — reversible trusted sandbox: confirmed support mechanism

Retain the V171 transaction, recovery, provenance, restart, and independent-verification contract unchanged. Repeat
its validation only if routing or persistence semantics change.

### Track C — clean exact certification-aware planning: strongly confirmed

Close V175/V177. Do not rerun, tune, expand thresholds, or select states. Preserve these results as the clean,
well-specified reference condition.

### Track E — bounded observation-corruption robustness: next active track

The next scientific question is whether the safety architecture survives imperfect inspection results without
introducing probability-threshold commit authority.

Use a conservative set-membership model:

- initial frozen constraints remain trusted;
- among subsequent inspection outcomes, at most one may be corrupted;
- after any history, retain every candidate whose predicted outcomes disagree with the observed history in no more
  than the corruption budget;
- route only if this robust version space is unanimously alias or unanimously composition;
- defer mixed or unanimously provisional robust version spaces;
- never route from posterior probability alone.

#### Recommended sequence

1. **V178 robust-certificate feasibility census.** On a declared development population, enumerate every frozen target,
   every inspection subset, and every admissible zero-or-one-corruption pattern. Compute exact minimal worst-case robust
   certificates and target-blind achievable trusted completion by horizon. Score no routed risk, policy, or sandbox.
2. **If single-pass certification is infeasible, test measurement redundancy structurally.** Preregister repeated
   inspections or another fixed error-correcting observation design before opening its feasibility results. Do not
   weaken unanimity or add an empirical confidence threshold.
3. **If feasible, develop a robust certification-aware planner.** Freeze the horizon, redundancy, corruption model,
   cost, and controls from the structural census; optimize actual routed loss; keep the V171 gate and sandbox authority
   unchanged.
4. **Fresh confirmation only after positive development.** Construct another exact-context-disjoint population before
   scoring the unchanged robust policy.

The corruption condition should enumerate adversarial admissible flips rather than report only average random-noise
performance. This cleanly separates worst-case safety from expected efficiency.

## Standing boundaries

- Planner, model, oracle, posterior, and hidden target never authorize commit.
- `provisional_primitive` never enters the trusted sandbox.
- Population construction, feasibility, planner development, and confirmation remain separately locked.
- Safe negative and mixed results are retained without in-place repair.
- Registration, real services, trusted real-state mutation, side effects, and execution remain zero.
