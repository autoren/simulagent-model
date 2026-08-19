# V187r1 clean-planner outcome verification repair plan

V187r1 is a verification-only repair. The frozen V187 runner wrote `raw_question_count: 0` in `problem-summary.json` because that presentation artifact read the unaugmented locked config after reconstruction. The independently evaluated `policy-summary.json` correctly contains the reconstructed count of 164. The original verifier expected 164 in both locations and therefore rejected the otherwise exact outcome.

V187r1 may change only the expected value for that one presentation field to the value actually specified by the frozen runner. It must reproduce every policy, target path, record result, scientific gate, decision, and hash without rerunning or modifying the V187 evaluation. The original failed audit remains immutable evidence.

Passing freezes the V187 negative result. It cannot authorize the correlated-error or model successor because V187's scientific development gates failed.
