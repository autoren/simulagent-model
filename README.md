# Simulagent QLoRA

This project turns Simulagent's deterministic world engine into a counterfactual transition
dataset and a reproducible Qwen3.5 LoRA/QLoRA experiment.

The research target is intentionally narrower than a complete agent policy:

```text
observation history + candidate action -> identifiable? + complete possible-transition set
```

Action choice, transition prediction, and belief revision remain separately measurable. The
simulator—not a frontier model—supplies every transition label.

## What is implemented

- Bounded breadth-first discovery of reachable simulator states.
- Counterfactual execution of every available action from every discovered state.
- Canonical state, inventory, flag, affordance, and environment deltas.
- Agent-view epistemic targets that preserve ambiguity instead of forcing one latent-world label.
- A privileged-state track with all transition-relevant scenario rules.
- Prompt- and context-disjoint train/validation/test splits.
- MLX chat-format JSONL output for Qwen3.5 QLoRA.
- Dataset integrity checks and deterministic split tests.
- Baseline/adapter inference and exact plus field-level evaluation scripts.
- Dataset audits for observational ambiguity, prompt overlap, imbalance, and split integrity.
- No-change, action-majority, exact-prompt lookup, and nearest-neighbour baselines.
- Smoke configurations for Qwen3.5-4B and 9B, plus a 9B v2 agent configuration.
- A compact outcome-count calibration track, strict evaluator, balanced variant, and 0.8B config.
- A V3 context-group stratifier over ambiguity, outcome counts, actions, scenario families, and
  supported mechanic tags, plus full-validation multi-seed checkpoint selection.

## Prerequisites

- Apple silicon Mac for MLX training.
- Node.js 20 or newer.
- Python 3.10 or newer.
- The sibling project at `../simulagent`.

## Bootstrap

```bash
./scripts/bootstrap.sh
```

This installs Node and Python dependencies, builds a pilot dataset, type-checks the code, and
runs the tests.

## Generate data

```bash
npm run dataset:pilot
npm run dataset:validate -- data/pilot
npm run dataset:full
npm run dataset:v2
npm run dataset:validate:v2
npm run dataset:v3
npm run dataset:validate:v3
```

Each generated dataset contains:

```text
data/<name>/
├── manifest.json
├── records/
│   ├── train.jsonl
│   ├── valid.jsonl
│   └── test.jsonl
└── mlx/
    ├── agent/{train,valid,test}.jsonl
    ├── outcome-count/{train,valid,test}.jsonl
    ├── outcome-count-balanced/{train,valid,test}.jsonl
    └── privileged/{train,valid,test}.jsonl
```

The v1 `records` files preserve metadata and oracle traces for evaluation. Dataset v2 is written
under `data/v2`, with separate `records/agent` and `records/privileged` splits plus matching MLX
chat files. Agent v2 deduplicates observationally equivalent prompts and labels each with the
complete empirical possible-outcome set.

Dataset v3 is the calibration revision of the agent track. It retains whole observation-context
groups while aligning class and mechanic distributions. Its MLX and record artifacts live under
`data/v3`; the privileged track remains in v2 because V3 currently tests agent-view calibration.

## Audit and deterministic baselines

```bash
npm run audit
npm run baselines
npm run audit:v2
npm run baselines:v2:agent
npm run baselines:v2:outcome-count
npm run baselines:v2:privileged
npm run audit:v3
npm run baselines:v3:outcome-count
```

The legacy v1 full corpus has no scenario-group leakage, but every test agent prompt appears
verbatim in training and 77.3% of records belong to prompt groups with conflicting targets. An
exact-prompt lookup reaches 84.6% test exact match, compared with 75.6% for nearest neighbour,
58.1% for action majority, and 45.2% for no change. These results mean the current split is useful
for pipeline development but not for a held-out-generalization claim. See `docs/dataset-audit.md`
and `docs/baseline-results.md`.

Dataset v2 closes that audit gate: no exact prompt crosses a split, no privileged full input has
contradictory targets, and all 612 ambiguous agent prompts carry explicit possible-outcome sets.
Its current hash split is not ambiguity-stratified, however: ambiguity ranges from 26.1% in test
to 64.1% in validation. Treat calibration results as diagnostic until that split is revised.
See `docs/v2-audit.md`, `docs/v2-agent-baselines.md`,
`docs/v2-outcome-count-baselines.md`, and `docs/v2-privileged-baselines.md`.

V3 reduces the ambiguity-rate spread to under one percentage point while preserving zero prompt
and context overlap. The largest observed common-mechanic share gap is 5.54%. See
`docs/v3-audit.md` and `docs/v3-outcome-count-baselines.md`.

## Training

First verify Qwen3.5 backward-pass and adapter compatibility:

```bash
./scripts/train-smoke.sh
```

The matched 9B v1 compatibility check is:

```bash
./scripts/train-smoke-9b.sh
```

The v2 agent-task compatibility check is:

```bash
./scripts/train-v2-agent-smoke-9b.sh
```

After the smoke run succeeds, generate v2 and train the 9B agent adapter:

```bash
source .venv/bin/activate
mlx_lm.lora --config configs/qwen35-9b-v2-agent.yaml
```

The 100-step configuration trains 5.410M LoRA weights across eight transformer layers. On the
64 GB M4 Pro development machine it reached a measured peak of 54.4 GB on an unusually long
multi-outcome batch. Do not run other memory-heavy applications during training.

The preferred fast calibration experiment uses Qwen3.5-0.8B and a single-digit outcome-count
target:

```bash
./scripts/train-v2-outcome-count-0.8b.sh
```

This trains 3.608M LoRA weights with about 6.3 GB measured peak unified memory. The current
checkpoint results are diagnostic rather than a production model: generated behavior oscillates
between majority modes because the context-disjoint splits are not class-stratified.

The preregistered V3 calibration run trains the same 0.8B adapter under three seeds and evaluates
every saved checkpoint on the complete fixed validation set using constrained next-token scores
over digits 1 through 5:

```bash
./scripts/run-v3-count-multiseed.sh
```

It never opens the test split. The gate requires every selected seed to exceed 50% balanced
identifiability, at least two seeds and the mean to reach 55%, no more than a ten-point seed
range, and nonconstant predictions for every selected checkpoint. Oracle-count transition
generation proceeds only after that gate passes.

The completed run failed that gate: all three validation-selected checkpoints predicted count 1
for every prompt and reached 50% balanced identifiability. A balanced-prior token Naive Bayes
baseline using only visible input reached 73.08%, locating the remaining problem in the current
five-way digit SFT objective rather than the V3 split or the absence of input signal. See
`docs/v3-results.md`, `docs/v3-calibration-results.md`, and `docs/v3-logit-diagnostics.md`.

The legacy v1 exact-transition and privileged configs remain available for compatibility work:

```bash
mlx_lm.lora --config configs/qwen35-4b-transition.yaml
mlx_lm.lora --config configs/qwen35-4b-privileged.yaml
```

The v2 long-form configs use a 1,536-token cap. Qwen3.5 token audits found maxima of 1,289 tokens
for agent epistemic examples and 1,222 for privileged examples. Outcome-count examples are at
most 613 tokens under their 1,024-token cap, so none of the three tracks is truncated.

## Baseline and adapter evaluation

Generate a small untuned baseline:

```bash
source .venv/bin/activate
python python/predict_mlx.py \
  --records data/pilot/records/test.jsonl \
  --output outputs/qwen35-4b-baseline.jsonl \
  --limit 20
```

Evaluate it:

```bash
python python/evaluate_predictions.py \
  --gold data/pilot/records/test.jsonl \
  --predictions outputs/qwen35-4b-baseline.jsonl
```

For a trained adapter, add `--adapter-path adapters/qwen35-4b-transition` to the prediction
command.

Evaluate the v2 epistemic adapter over its entire prompt-disjoint test set:

```bash
python python/predict_epistemic_mlx.py \
  --adapter-path adapters/qwen35-9b-v2-agent \
  --output outputs/qwen35-9b-v2-agent-test.jsonl
python python/evaluate_epistemic.py \
  --predictions outputs/qwen35-9b-v2-agent-test.jsonl \
  --output outputs/qwen35-9b-v2-agent-test-metrics.json
```

Generate and evaluate the compact count track with a trained adapter:

```bash
python python/predict_outcome_count_mlx.py \
  --model mlx-community/Qwen3.5-0.8B-4bit \
  --adapter-path adapters/qwen35-0.8b-v2-outcome-count-digit \
  --output outputs/qwen35-0.8b-v2-outcome-count-test.jsonl
python python/evaluate_outcome_count.py \
  --predictions outputs/qwen35-0.8b-v2-outcome-count-test.jsonl
```

The inference script explicitly disables Qwen3.5 thinking mode. The MLX chat template used for
training supplies an empty thinking block before each target; matching that format at inference
keeps the token budget focused on the JSON transition rather than a visible reasoning trace.

## Important limitation

Dataset v2/v3 possible-outcome sets are empirical equivalence classes over the currently generated
worlds. They are complete for this corpus, not a proof over every world the simulator could ever
express. A stronger generalization claim still needs broader procedural mechanics, entity
renamings, and held-out mechanic combinations.

See `docs/research-plan.md` for the experiment design and claim boundaries.
The exact environment and measurements from the completed compatibility run are recorded in
`docs/smoke-results.md`.
