# V70 Confirmatory Multi-Environment Results

## Bottom line

V70 passed all 22 frozen confirmatory gates on the complete sealed census of
244 records from nine untouched models. The frozen decision is:

`confirm_multi_environment_replication_for_project_authored_V69_family`

The result confirms the V69 development finding at the preregistered
model-level hierarchy: the distribution-aware Bayes-adaptive mixture can make
materially better finite-horizon decisions than frozen point controls under the
project-authored dominant latent-remapping family. This is a claim about the
pinned source suite and the locked V69 family, not a universal claim about all
POMDPs, posterior approximations, or externally supplied uncertainty families.

## Primary result

- All 244 sealed records and all nine valid models were retained.
- Four models qualified the paired MAP replication gate; three were required.
- The qualifying set contains two of three structurally related models and two
  of six novel models, passing both frozen stratum subgates.
- Six models exhibited material posterior-sampling regret; two were required.
- The maximum normalized MAP regret was `0.09544903686417067`, above the frozen
  `0.01` gate.
- Across records, there were 60 exact BA/MAP first-action disagreements. Eighteen
  records met the stricter same-record criterion of both a disagreement and MAP
  regret of at least `0.005`.
- There were 44 records with material MAP regret and 52 with material
  posterior-sampling regret. These record counts are descriptive; they do not
  replace the frozen model-level decision hierarchy.

| Model | Stratum | Records | BA/MAP disagreements | Qualifying MAP records | Material MAP | Material PS | MAP fallback affected |
|---|---:|---:|---:|---:|---:|---:|---:|
| `4x3.POMDP` | related | 25 | 8 | 6 | 6 | 12 | 0 |
| `fully_observable_tmaze2.POMDP` | related | 15 | 1 | 1 | 1 | 2 | 2 |
| `fully_observable_tmaze5.POMDP` | related | 15 | 0 | 0 | 0 | 0 | 2 |
| `cheese.95.POMDP` | novel | 28 | 0 | 0 | 9 | 9 | 15 |
| `cheese.95_nonterminating.POMDP` | novel | 28 | 0 | 0 | 9 | 9 | 16 |
| `hallway.POMDP` | novel | 104 | 45 | 6 | 14 | 14 | 86 |
| `heavenhell.POMDP` | novel | 11 | 0 | 0 | 0 | 0 | 2 |
| `network.POMDP` | novel | 8 | 5 | 5 | 5 | 6 | 0 |
| `shuttle.POMDP` | novel | 10 | 1 | 0 | 0 | 0 | 2 |

## Numerical and protocol checks

- Source validation, belief normalization, finite metrics, census completion,
  and primary-action convergence rates were all `1.0`.
- The maximum primary-versus-convergence normalized value error was
  `1.2454736823139355e-15`, well below the frozen `1e-8` limit.
- No records or models were selected, rejected, replaced, or rescored after the
  locks.
- Development models were not rescored. There were no SMC2 runs, human records,
  model forward passes, or adapter-training runs.
- The evaluator completed in `20014.290989167057` seconds. The unusually long
  runtime was dominated by the 104-record `hallway.POMDP` block; it did not
  alter the census or reporting rules.
- The independent outcome audit reproduced every aggregate and gate from the
  sealed rows, matched all 244 record identities to the census, verified the
  one-shot and hash chain, and confirmed that fallback diagnostics do not enter
  the primary gates.

The durable outcome is bound by
`configs/v70-confirmatory-outcome-lock.json`. Its result hash is
`0e92a0c66019848d1b8e01b9f89ac2a52036ccbec4f2e1db4fbae161776bf115`.

## Fallback and Tier B interpretation

The point controls use the frozen totalization rule after an exact-zero
point-model observation. This rule is part of the preregistered control, while
its diagnostics are explicitly secondary and non-decisional.

Two qualifying models give fallback-free evidence: `4x3.POMDP` and
`network.POMDP` have zero affected MAP records and still qualify. The qualifying
records in `fully_observable_tmaze2.POMDP` and `hallway.POMDP` overlap the
totalization diagnostic, so those effects should be described as performance
of the complete locked point-control procedures, not as a fallback-independent
theorem about point estimates.

The terminating and nonterminating cheese models are the separate Tier B stress
pair. Each has nine material MAP-regret records but no exact BA/MAP root-action
disagreement under the frozen tie semantics, and every material effect overlaps
the fallback diagnostic. They therefore do not qualify the primary paired gate
and do not support an external uncertainty-family claim. Their similarity is a
useful robustness observation about termination handling only.

## Scientific direction after V70

V70 closes the planned development-to-confirmation sequence for the V69 family.
The next defensible stage is synthesis and boundary testing, not retrospective
tuning on these nine models:

1. Consolidate V68r2, V69, and V70 into a single methods-and-results narrative
   with the development/confirmation firewall made explicit.
2. Separate fallback-free evidence from effects involving the frozen
   totalization rule in every headline table and claim.
3. If broader generalization is pursued, preregister a new uncertainty family
   and fresh models before computing outcomes. V70 models and gates must remain
   frozen and must not become a new development set.
4. Treat approximate inference, SMC2, human data, and model/adapter experiments
   as separate future programs requiring their own feasibility and validation
   stages.
