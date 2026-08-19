# V119 Asymmetric Adaptive Evidence Results

## Outcome

V119 is a negative but sharply informative simulator result:

> `adaptive_causal_feasibility_fails_keep_language_and_model_closed`

The frozen language-free tree routed a root candidate-confirmation observation to two identity-specific
mechanisms after `CONFIRM`, or to two support-status mechanisms after `REJECT`/`UNSURE`. Every path retained
all 17 hypotheses, cost 0.30, and permitted only exact known, unsupported, or abstain shadow decisions. No
language, model, API, protected access, manual record inspection, or execution occurred.

The asymmetric architecture repaired every V117 safety and accuracy failure. It missed only the frozen
all-prior mean-regret gate, by small margins in three correlated conditions. Because the gate is
noncompensatory, V119 is formally negative and cannot authorize a mechanism realization or language run.

## Primary 95% reliability result

For the correlation-aware planner:

| Prior | Correlation | Mean regret | Known exact | Unsupported correct | False known |
| --- | ---: | ---: | ---: | ---: | ---: |
| uniform | 0.00 | 0.7574 | 87.43% | 92.51% | 0.005% |
| uniform | 0.25 | 0.7865 | 85.30% | 88.05% | 0.157% |
| uniform | 0.50 | 0.7775 | 87.54% | 90.37% | 0.313% |
| moderate | 0.00 | 0.7409 | 91.80% | 92.51% | 0.123% |
| moderate | 0.25 | 0.7720 | 91.86% | 88.05% | 0.561% |
| moderate | 0.50 | 0.7692 | 89.73% | 90.37% | 0.373% |
| strong | 0.00 | 0.7409 | 91.80% | 92.51% | 0.123% |
| strong | 0.25 | 0.7720 | 91.86% | 88.05% | 0.561% |
| strong | 0.50 | 0.7866 | 91.92% | 90.37% | 0.999% |

Every known-accuracy result exceeded 80%, every unsupported result exceeded 80%, and every false-known
probability was below 1%, far inside the 10% ceiling. Mean regret nevertheless exceeded the frozen 0.7760
baseline under uniform correlation 0.25 (by 0.01046), uniform correlation 0.50 (by 0.00141), and strong
correlation 0.50 (by 0.01053). The moderate prior passed throughout the required correlation range.

## Controls and interpretation

At actual correlation 0.50, the independence-assumed planner stayed below the preregistered 1.125
misspecification ceiling and below 1% false-known probability under every prior. At a perfect channel, every
prior reached exactly 0.690625 regret, 96.875% known accuracy, 100% unsupported correctness, and zero
false-known probability. All implementation, retention, safety, and control gates passed.

Compared with V117, the structural gain is large. V117 had 0% uniform-prior known accuracy and about 1.21
regret through correlation 0.50. V119 raised uniform known accuracy above 85% and reduced regret to
0.7775--0.7865. The remaining failure is not basic identifiability; it is that the residual decision loss
plus the frozen 0.30 clarification cost does not robustly dominate the already competent 0.7760 historical
policy in every condition.

Freeze V119 without changing cost, reliability, priors, correlation, or gates. The appropriate next step is
an aggregate, model-free regret decomposition of the three failing conditions and a stop/go assessment of
whether any non-retuned architecture can improve value through selective querying rather than paying 0.30
on every record. This cannot mine individual records or authorize language/model use.
