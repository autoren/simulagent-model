# Simulagent QLoRA

> **Current research status (2026-08-19):** The sequence through V224 is consolidated and experimental/model
> escalation is frozen. The supported result is a model-free mechanism architecture, not externally validated
> open-world language understanding. Start with
> [`docs/README.md`](docs/README.md),
> [`docs/cross-track-evidence-synthesis-through-v224.md`](docs/cross-track-evidence-synthesis-through-v224.md), and
> [`docs/research-stopping-rule-after-v224.md`](docs/research-stopping-rule-after-v224.md). Older roadmap files are
> historical snapshots and do not authorize work.

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
- A V4 binary identifiability track with a context-disjoint calibration fold, calibrated A/B
  logit scoring, token shortcut ablations, score-resolution diagnostics, and preregistered gates.
- A V5 frozen-representation track with quartile-layer last/mean pooling, true-float32 regularized
  linear heads, three optimization seeds, context-group bootstrap intervals, and a no-history
  shortcut diagnostic.
- A locked V5 challenge track over new short-start relock and held-out power-trip worlds, with
  entity-renaming/paraphrase triplets, evidence contrasts, hash-locked one-shot evaluation, and
  preregistered robustness gates.
- A V6 shortcut-resistant corpus with paired surface supervision, evidence interventions,
  context-disjoint development, a newly implemented mirror-rejection mechanic holdout, and a
  hash-locked one-shot transfer decision before LoRA.

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
npm run dataset:v4
npm run dataset:validate:v4
npm run dataset:v6
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

Dataset v4 reads only V3 training and validation records. It carves a calibration fold from V3
training contexts, preserves V3 validation for one frozen evaluation per seed, and never reads V3
test. Its binary MLX targets use `A` for identifiable and `B` for ambiguous.

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
npm run baselines:v4:binary
npm run diagnostics:v4:binary
npm run diagnostics:v4:fp32
npm run probe:v5:0.8b
npm run challenge:v5:run
npm run holdout:v6:run
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

V4's train/calibration/validation ambiguity rates differ by at most 1.13 points, its maximum
mechanic-share gap is 3.78 points, and no prompt or context crosses a split. The primary calibrated
full-input token baseline reaches 58.24% validation balanced accuracy. Removing history and
memories reaches 76.92% as a diagnostic ablation, motivating stronger shortcut and invariance
tests. See `docs/v4-experiment-plan.md` and `docs/v4-binary-baselines.md`.

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

The preregistered V4 run applies the smallest controlled change: balanced binary `A`/`B` labels,
checkpoint and threshold selection on a separate calibration fold, and one validation pass per
frozen seed:

```bash
./scripts/run-v4-binary-multiseed.sh
```

V4 also failed. Selected-seed validation balanced accuracy was 49.15%, 48.96%, and 55.49%, for a
51.20% mean; both the engineering stability and scientific token-baseline gates failed. The
selected vocabulary-logit margins contained only two or three unique values. Direct float32
projection recovered 153–154 distinct scores but only 51.16% mean balanced accuracy and 0.540 AUC,
so precision was not the main failure. See `docs/v4-results.md`,
`docs/v4-binary-results.md`, and `docs/v4-score-diagnostics.md`.

V5 bypasses the vocabulary head with a regularized discriminative probe over frozen hidden
states. On Qwen3.5-0.8B, all three true-float32 fits select layer 6 mean pooling. Full input reaches
96.15% validation balanced accuracy and 0.9998 AUC; the fixed no-history diagnostic reaches 91.64%
and 0.9803. This decisively establishes accessible representational signal, so 4B/9B extraction is
deferred. It does not establish semantic reasoning: no-history errors are concentrated in three
of 19 contexts, including two wholly failed contexts. That result triggered the locked,
shortcut-resistant challenge described below. See `docs/v5-frozen-probe-results.md`.

The locked challenge shows that the apparent V5 success did not transfer. Without retraining or
recalibration, the canonical probe reaches 49.58% balanced accuracy and 0.457 AUC across 120 new
base cases from 63 contexts. Power-trip is a constant-identifiable 50.00%; short-start relock is
52.28%. Entity-renamed and paraphrased inputs remain near chance. Six of eight preregistered gates
fail, although two narrow relock evidence contrasts retain correct score direction. This locates
the 96.15% development result in dataset/template correlations rather than a demonstrated general
epistemic boundary. See `docs/v5-challenge-plan.md` and `docs/v5-challenge-results.md`.

V6 retrains only the fixed 0.8B layer-6 mean linear head on 143 base records from short-start
relock and power-trip, with canonical/entity-renamed/paraphrased triplets and evidence
interventions. A newly implemented mirror-rejection mechanic is reserved for one locked
evaluation. Development calibration reaches 90.32% balanced accuracy, but the untouched mechanic
reaches only 63.16% (95% context-group interval 51.55%–82.44%) and fails six of ten gates.
Entity-renamed transfer reaches 69.74%, paraphrased transfer 57.89%, and evidence-directional
accuracy is 0% across 12 label-changing comparisons. The failure is systematic: announced
development records are exclusively ambiguous, whereas the corresponding holdout evidence is
identifiable, so the probe learns an inverted evidence-wording shortcut. LoRA remains a no-go.
See `docs/v6-experiment-plan.md` and `docs/v6-results.md`.

V7 removes that observable wording shortcut before model access. Its final corpus contains 408
training, 172 calibration, and 70 untouched base records, always as complete
canonical/entity-renamed/paraphrased triplets. Conditional label gaps are exactly zero, and both
metadata-only and evidence-text-only audits score exactly 50% balanced accuracy. The fixed frozen
probe passes its development gate at 76.16%, after which a genuinely new tone-drift simulator
mechanic is opened once under a hash lock. Canonical rank ordering transfers (0.807 AUC), but the
locked threshold predicts 69 of 70 cases as identifiable: balanced accuracy is 48.57%, the grouped
bootstrap lower bound is 42.86%, and both evidence-direction metrics are 0%. Eight preregistered
holdout gates fail. The resulting decision is **no-go for LoRA**; no adapter training follows this
run. See `docs/v7-experiment-plan.md` and `docs/v7-results.md`.

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
