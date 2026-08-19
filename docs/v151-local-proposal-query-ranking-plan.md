# V151 local proposal and query-ranking plan

V151 is the first local-model study under the typed witness firewall. It does not ask the model to accept a known capability or produce trusted evidence. On each of 96 V149 development request fixtures, the pinned local model emits one bounded JSON proposal containing an evidence-status judgment, one to three candidate state IDs, a complete ranking of all six registered clarification questions, and a diagnostic confidence.

The proposal never prunes the complete authoritative state set. A deterministic controller uses the question ranking only to order closed questions. Every preceding irrelevant question is explicitly passed through the typed witness firewall with no trusted selection and must remain non-executable `A00` with the full hypothesis universe retained. The controller continues until the registered discriminating question produces an immutable answer event. The model's state candidates are measured for recall but cannot determine the final state.

Four request stages per group create 96 generations. The ambiguous request is evaluated on both latent sides, producing 120 sequential episodes. The run uses one pinned Qwen3.8-27B 4-bit load, direct inference without thinking, one sample per prompt, no retry, no raw response persistence, zero closed-answer or evaluation-split generations, no API, no training, and zero execution.

Noncompensatory gates cover structural validity, compatible-state recall, exact set recovery, abstention status, decidable top-one accuracy, query rank, false-known singleton proposals, candidate attraction, sequential evidence cost, final safety after trusted answers, and complete hypothesis retention. Confidence calibration and risk-coverage are diagnostic only; no threshold is fitted.

Passing authorizes only a separate V149 evaluation preregistration. Failure closes the branch without prompt changes, retries, reruns, threshold fitting, model changes, or evaluation access.
