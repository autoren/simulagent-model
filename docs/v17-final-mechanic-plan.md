# V17 preregistration: one-shot final-mechanic evaluation

## Authorization and claim boundary

V16 is the sole authorization for V17. V17 performs one blind evaluation of the unchanged V15 frozen 4B neuro-symbolic architecture on one newly constructed simulator-derived mechanic. It trains no adapter, changes no representation, prompt, layer, regularization value, threshold, symbolic rule, or success gate, and permits no second final-mechanic score.

A pass supports the narrow claim that the frozen architecture transfers to one unseen action mechanic when state concepts, temporal language, polarity operators, and lexicons are supported. It does not establish transfer to arbitrary new ontologies or authorize LoRA. A failure rejects that final-generalization claim; the V17 corpus then becomes permanently diagnostic and cannot be used for another final score.

## Final mechanic

The final mechanic is `beacon_console_diagnostic`. The candidate action inspects the beacon console's diagnostic readout in the simulator. Its determinants are the already supported concepts `generator_stable`, `mirror_seated`, and `fork_calibrated`, but its candidate action and transition table are absent from development. The simulator is run over all eight Boolean assignments. The action's actual outcome and action-surface delta must yield exactly eight distinct transition codes, making this a new multiway transition mechanic rather than a renamed development record.

To isolate mechanic composition from vocabulary novelty, the frozen `beacon_calibration` phrase registry supplies the three state lexicons. The final action is rendered independently in canonical, entity-renamed, and paraphrased form. Every source scaffold is expanded across V14's unchanged nine current-state surface families: three affirmative-gold, three negated-opposite, and three contrastive forms.

The constructor may use only replica 0 of the frozen V8 `beacon_calibration` scaffolds to preserve the previously audited assignment, intervention, evidence-balancing, and lexicon topology. It must replace the action, simulator-derived transition table, identifiers, ambiguity labels, possible transitions, and intervention kind. No V17 record exists or is read before the construction lock.

## Corpus topology and data seal

The locked topology is 144 source scaffolds and 1,296 final records: 24 contexts, nine template families, three state lexicons, and 216 complete six-record intervention groups. Every template-by-lexicon cell contains 48 records and both identifiability labels. All unresolved/resolved pairs must be label-changing because the eight-way table is injective.

Construction is a one-shot phase. The compiler refuses to overwrite an existing V17 directory. It must pass deterministic structural and symbolic validation, record its own construction-lock hash, and write artifact hashes. A separate seal then pins the completed manifest and record artifact without fitting a model or running inference. Feature extraction and evaluation require that seal.

## Frozen deployment fit

The backbone remains `mlx-community/Qwen3.5-4B-4bit` at revision `0e7ffd5c629ef7719d4cbc04069232580bfa9d9c`, layer 8, float32 evidence-span means and hypothesis-token means, with a 512-token maximum. The V15 base and NLI prompts are reused byte-for-byte for development training. Final prompts use the identical V15 prompt builders and span rules.

Exactly three balanced logistic probes are fit once, with `C=1.0`, seed 0, `lbfgs`, and the V15 standardization:

1. match on every unique V14 development base prompt;
2. temporal status on every matched unique V14 development base prompt; and
3. signed active-versus-inactive polarity on every current matched unique V14 development prompt.

All V14 train and evaluation records are now development data and are eligible for this final deployment fit. Prompt deduplication preserves one weight per exact prompt. V17 labels are never used in a fit, model choice, threshold, or retry. The three fitted heads and their hashes are retained.

## One-shot scoring and gates

The exact V15 selector, temporal classifier, signed polarity comparator, allowed-value derivation, and symbolic evaluator produce the final predictions. Nine template families are treated as the final transfer folds; their three state-lexicon cells are the surface cells. This gives nine fold-level and 27 template-by-lexicon cell evaluations under the unchanged V15 gate function and thresholds.

Every gate must pass:

- fold/cell span accuracy: 0.65 / 0.60;
- fold/cell predicted-span temporal accuracy: 0.70 / 0.65;
- fold/cell oracle-span/oracle-temporal polarity accuracy: 0.70 / 0.65;
- fold/cell hypothesis-pair consistency: 0.70 / 0.65;
- fold/cell fully predicted allowed-values accuracy: 0.65 / 0.60;
- fold/cell fully predicted symbolic balanced accuracy: 0.65 / 0.60;
- every template's complete label-flip-pair accuracy: 0.60; and
- every template's complete six-record intervention-group accuracy: 0.50.

The complete overall result and all component ablations are reported regardless of outcome. Mean performance cannot compensate for a failed template or lexicon cell.

## Firewall and run limits

- one final corpus construction and one immutable data seal;
- one frozen feature extraction over the final prompts;
- three development-only linear fits and one final evaluation;
- zero adapter runs, alternate models, alternate layers, alternate features, prompt changes, threshold changes, hyperparameter searches, or final retries;
- zero access to Tone Drift, V3 test records, prior holdouts, untouched V8 mechanics, or V7 model outputs;
- no V17 target may influence training; and
- any post-result experiment must treat V17 as exposed diagnostic data, never as a final holdout.
