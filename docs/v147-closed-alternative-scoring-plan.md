# V147 closed-alternative scoring plan

V147 tests whether the pinned local model can rank exact registered certificates without generating a thinking trace or certificate syntax. It uses only the 144 fresh V146 development fixtures. The V146 test partition will receive zero scores, but it is retired as future evidence because two rows were displayed during pre-preregistration implementation inspection. A passing result can therefore authorize only a newly authored blind successor population, never use of V146 test.

Each fixture deterministically permutes the 14 certificate codes onto opaque aliases `C00` through `C13`. Every alias is exactly three tokens under the pinned tokenizer. The complete alias-to-certificate table is visible in the prompt. The scorer computes the summed next-token log likelihood of every alias and selects the unique maximum. Because all alternatives have equal token length, selection requires no length normalization. A tie, non-finite value, missing alternative, tokenizer-boundary failure, or invalid vector fails closed to `A00`.

The model is loaded once. There are 144 prompt scoring operations and exactly 2,016 candidate-sequence scores, but zero generated tokens, retries, API calls, training, or execution. Only registered codes, numeric score diagnostics, hashes, and timings may be persisted; there is no raw generated text or trace.

The preregistered semantic and sequential gates match the V144 evidence contract. Candidate-normalized softmax values, calibration error, Brier score, and risk-coverage summaries are diagnostic only and are not described as calibrated probabilities.

Passing authorizes only preregistration of a newly authored blind successor population. Failure closes the branch without rescoring, alias changes, prompt tuning, or rerunning. Neither outcome authorizes any V146-test use.
