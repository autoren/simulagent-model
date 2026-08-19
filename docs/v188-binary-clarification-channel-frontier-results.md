# V188: Binary clarification-channel frontier

## Bottom line

V188 explains V187 structurally. The frozen 14-contract prior contains `3.634254` bits of uncertainty. An unrestricted optimal binary code needs `3.70` questions on average, while the semantic partitions available in V186 need `3.941667` questions on average and up to seven. Under V187's four-question horizon, exact adaptive querying has value only while a typed question costs at most `0.0875` on the frozen grid; at `0.09` it switches to generic clarification.

All frontier gates passed. The result authorizes design—but not execution—of a separately locked multiway typed-channel feasibility study.

## Information controls

- Shannon entropy: `3.6342540514` bits.
- Optimal unrestricted Huffman expected depth: `37/10 = 3.70` questions.
- Huffman depth range: 3–6.
- Frozen-codebook exact expected depth: `473/120 = 3.9416666667` questions.
- Frozen-codebook depth range: 2–7.
- Frozen-codebook exact leaves: 14 of 14.
- Target retention and exactness: 1.0.

The frozen semantic codebook therefore pays an average geometry overhead of about `0.241667` questions over the optimal unrestricted binary code. More importantly for interaction design, several low-prior contracts require six or seven sequential questions.

At the V187 cost of 0.10, administering the complete restricted exact tree would cost `0.394167` on average—barely below the 0.40 generic answer—but would exceed V187's four-question horizon for some targets and impose as many as seven interactions. This is a theoretical depth control, not a revision of V187.

## Four-question cost frontier

The prospective grid contained all 81 costs `i/400`, for `i=0..80`. Exact adaptive and best fixed open-loop policies were rebuilt at every cell.

- Positive-value exact-adaptive cells: 36.
- Cells with at least `0.005` adaptive advantage over open loop: 36.
- Largest positive/adaptive-advantage cost: `0.0875`.
- At `0.0875`: exact cost `0.395`, typed-only completion `0.80`, open-loop/generic cost `0.40`.
- At `0.09`: exact and open loop both choose generic immediately at cost `0.40`.
- The exact V187 control at `0.10` was reproduced: cost `0.40`, zero questions, zero typed-only completion.

The observed grid breakpoints were:

1. `0.0000`: zero-cost policy ordering;
2. `0.0025`: stable positive-cost adaptive tree;
3. `0.0725`: best open loop gives up and becomes always-generic while adaptive querying remains useful;
4. `0.0900`: exact adaptive also gives up and becomes always-generic.

Between `0.0725` and `0.0875`, the most relevant value comes specifically from history-dependent selection: the fixed open-loop policy is already generic, while exact adaptive querying retains a small positive benefit.

## Interpretation

V187 was not a mysterious planner failure and should not be repaired by retuning. Its 0.10 question cost lies above the four-question binary break-even interval, which the locked grid brackets between `0.0875` and `0.09`.

The full-depth control adds an important nuance: with unlimited patience, the same codebook is just barely economical at 0.10 in expectation, but only by tolerating paths of up to seven questions. This is not a good operational substitute for generic clarification. It exposes the actual design problem as a bandwidth/burden trade-off:

> The semantics are sufficient, but one binary answer at a time is too low-bandwidth for the allowed interaction budget.

The target-informed V187 oracle remains much cheaper at 0.10 because it knows which direct confirmation to ask. A future shadow proposal could try to approximate that choice, but V185 did not validate such a signal and no model condition is authorized here.

## Decision

Freeze V188 as a positive structural frontier. The four preregistered successor conditions all hold:

- V187 remains at the generic boundary;
- lower binary costs have positive value;
- the target-informed oracle gap is positive; and
- the four-question binary break-even is below the V187 question cost.

Proceed only to design a finite multiway typed-channel feasibility protocol. Its answer categories must be derived from the same allowed semantic attributes, and its costs must be fixed through an explicit information/cognitive-burden rule before scoring. A categorical answer must not be priced as though it were one binary bit.

No utterance language, protected language, model/API, training, registration, trusted-state mutation, service call, side effect, action, or execution was used or authorized.
