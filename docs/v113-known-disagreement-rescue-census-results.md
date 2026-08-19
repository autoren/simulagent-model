# V113 Historical Known-Disagreement Rescue Census Result

## Outcome

V113 evaluated 239 preregistered simple rescue rules over the exact preserved V112 transfer fixtures,
treating that population as historical policy-design evidence only. Sixteen of 192 records were eligible:
the local model had proposed a typed known intent, but deterministic nearest-intent retrieval selected a
different exact intent. No new model inference, protected-test access, or manual language or response
inspection occurred, and no individual feature or prediction was persisted.

Eighty candidates passed every one of the seventeen inherited V112 quality gates. The frozen selection rule
was:

```text
rescue the LLM's known intent when
    proposed-intent retrieval score >= 0.60
and nearest-minus-proposed score gap <= 0.15
```

The rule rescued four disagreements. On this historical population, known exact-intent accuracy increased
from 78.13% to 82.29%, top-confidence-80% error fell from 22.08% to 19.48%, mean regret improved from
`0.8047` to `0.7839`, and exact decision accuracy increased from 62.5% to 64.58%. False-known acceptance
remained 6.25%, confidence ECE remained low at 0.0287, and unsupported recall and precision remained
93.75% and 95.74%.

Because the rescue applies only to direct `KNOWN` disagreements, the abstention novelty signal was exactly
unchanged: 70.21% precision, 68.75% recall, 9.72% non-novel false-positive rate, and 0.0494 ECE. Safe
hypothesis retention remained 100% and execution count remained zero.

## Boundary and decision

Freeze the selected rescue rule for a separately locked evaluation on a new disjoint population. V113 is
not fresh transfer evidence because rules were selected using V112 labels. Do not use the existence of 80
historically feasible candidates as evidence that the repair generalizes, and do not retune the selected
thresholds.

Protected-test access, schema induction, sequential planning, additional model inference, APIs, adapter
training, capability creation, action authority, real service calls, and execution remain unauthorized.
