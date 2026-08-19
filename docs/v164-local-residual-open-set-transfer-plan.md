# V164 local residual open-set transfer plan

## Purpose

V163 established a safe deterministic-first architecture on fresh, record-disjoint MASSIVE language. Its
consensus controller resolves 20 of 96 evaluation requests with 95% exactness, zero false-known acceptance,
and mean regret 0.20. It abstains on a prospectively defined 76-record residual spanning all four structural
classes. V164 asks one narrow question: can the pinned local Qwen3.8-27B model add useful semantic coverage
only on that residual, without weakening the deterministic safety boundary?

This is Track A external open-set transfer. It is not the synthetic V160 relational-codebook branch and it
does not induce or register an ontology.

## Locked intervention

The model receives exactly the 76 V163 residual utterances, the frozen visible catalog, and a typed response
contract. Record identifiers, class labels, source annotations, truth, and deterministic predictions are hidden.
The 20 deterministic nonresidual decisions remain immutable and receive no model call. Twenty-four missing-
observation controls abstain deterministically and also receive no model call.

There is one local condition: `mlx-community/Qwen3.8-27B-4bit` at revision
`3e6447f082e89cc7f0bc6e5441afd38dfce760ff`, temperature zero, one sample, 160 output tokens, no retry, and
thinking disabled. This is not an arbitrary convenience choice. In V154, direct decoding had perfect syntax
and outperformed the mechanically bounded low-reasoning challenger; enabling low reasoning spent more tokens,
reduced structural validity, and did not improve semantic question ordering.

Invalid outputs map to the frozen zero-confidence ABSTAIN fallback. Raw outputs may be persisted only for
automatic hashing and reparsing; nobody manually inspects them before or after scoring.

## Evaluation

The primary comparison is the combined hybrid against the frozen V163 deterministic consensus. On the 20
nonresidual records the hybrid is exactly V163. On the residual it uses the validated model proposal. Quality
must pass noncompensatorily: syntax, status and typed-target accuracy, novelty and unsupported detection,
false-known risk, raw confidence calibration, residual regret, and combined hybrid exactness and regret.

The principal safety criterion is decision cost, not ordinary accuracy. A wrong KNOWN capability carries the
same cost 10 penalty frozen in V163. The hybrid must improve mean regret by at least 0.10 over V163, remain at
or below 0.85 overall regret, and keep false-known acceptance at or below 5%.

## Boundary and decision

The complete 17-state safe universe is retained. The model cannot change the catalog, prune hypotheses, update
authoritative state or belief, select an action, call a tool, or execute. All decisions and costs are shadow
counterfactuals. Protected language, APIs, training, services, side effects, and execution remain closed.

Passing authorizes only a separate protected-transfer preregistration. Failure closes this exact residual model
protocol without prompt, decoding, threshold, parser, model, or gate tuning. Either result then updates the
three-track roadmap.
