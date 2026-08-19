# V117 Causal Clarification Simulator Plan

V116's benefit required two 95%-reliable independent typed answers and disappeared when they were fully
correlated. V117 replaces the abstract duplicate answers with two distinct causal mechanisms. One observes
whether the current candidate is the exact known intent. The other observes catalog status: declared,
undeclared but within a visible scenario, outside every visible scenario, or insufficient.

The simulator preserves each mechanism's marginal correctness while sweeping a shared-failure correlation
from zero to one. A correlation-aware Bayes policy is compared with a misspecified policy that always assumes
independence. Both retain all 17 hypotheses and may only abstain or emit known/unsupported shadow decisions.

The primary gate requires the aware policy at 95% marginal reliability to beat the frozen V115 baseline for
all three priors and every correlation through 0.50. The misspecified policy must remain no worse than ask-
always at correlation 0.50. Correlation 1.00 is a mandatory stress report, not silently treated as independent.

This is a no-language, no-model, aggregate-only simulator. A pass could justify only a separately locked
fresh unprotected multi-turn language protocol. It cannot authorize that run immediately, open protected
data, begin induction or richer planning, call an API, train, grant authority, or execute.
