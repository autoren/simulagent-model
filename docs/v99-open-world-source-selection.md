# V99 Open-World Source Selection

## Decision

Use two complementary, independently authored, CC BY 4.0 sources under separate evaluation roles:

1. **MASSIVE 1.1, en-US** is the primary controlled open-set transfer source. It provides human-created
   English virtual-assistant utterances with 18 scenarios/domains, 60 intents, 55 slot types, fixed
   train/dev/test partitions, raw utterances, slot-annotated utterances, and machine-readable scenario
   and intent labels.
2. **PRESTO v1, en-US human-context records** is an auxiliary paired evidence-sufficiency source. Each
   selected human-authored utterance will be evaluated twice: once with its original human context and
   once with that context withheld. Eligibility must be mechanically proven by a target argument that is
   absent from the current utterance but present in the human previous turns or seeded state. This makes
   the ablated condition genuinely insufficient for the complete typed interpretation while holding the
   utterance and source style fixed.

Do not combine source identities as if they were exchangeable class labels. MASSIVE answers the core
known/unfamiliar/novel/unsupported question. PRESTO separately answers whether abstention responds to
missing evidence rather than dataset style.

## Primary evidence

MASSIVE's official dataset card and paper describe more than one million labeled utterances across 52
languages; the en-US source derives from human-created SLURP utterances. Its structured fields include
`scenario`, `intent`, `utt`, and slot-annotated `annot_utt`. The published ontology has 18 domains, 60
intents, and 55 slots. The dataset license is CC BY 4.0.

- Official dataset: https://huggingface.co/datasets/AmazonScience/massive
- Official paper: https://aclanthology.org/2023.acl-long.235/
- Publisher PDF: https://assets.amazon.science/98/31/d83b6aaf4fa486f160bee73e001b/massive-a-1m-example-multilingual-natural-language-understanding-dataset-with-51-typologically-diverse-languages.pdf
- Immutable-stage artifact: `amazon-massive-dataset-1.1.tar.gz`, 40,251,390 bytes, S3 ETag
  `51e0da2a3ff7a016f109e1d1b4306e93-3`, last modified 2022-11-07.

PRESTO's official repository and EMNLP paper describe more than 550,000 human/assistant conversations.
Each record supplies a current input, target semantic parse, locale, human-versus-synthetic context tag,
previous turns, and structured contacts/lists/notes. Its license is CC BY 4.0.

- Official repository: https://github.com/google-research-datasets/presto
- Official paper: https://aclanthology.org/2023.emnlp-main.667/
- Immutable-stage artifact: `presto_v1.zip`, 415,990,813 bytes, MD5/ETag
  `5fb5bd7e437a07fbae4991b5b4a573f4`, GCS generation `1678604196509246`.

## Rejected alternatives

- **SGD** is closed by V97/V98: its development families are exhausted and its test split adds only
  `Payment` and `Trains`, fewer than the four fresh families required.
- **CLINC150** has excellent crowdsourced English in-scope and explicit out-of-scope labels (150 intents,
  ten domains, and 1,000 OOS test examples), but no slot schema. Using it only for unsupported examples
  would make dataset identity a shortcut and would not test typed interpretation.
- **PRESTO as the sole core source** is not selected because its semantic parses do not expose a simple
  authoritative domain catalog equivalent to MASSIVE's scenario/intent/slot ontology. Its contextual
  structure is instead uniquely suited to the paired insufficiency question.

## Locked construction direction

For MASSIVE, build an ontology from structural scenario, intent, and slot annotations before emitting
language. Hash-select one fully withheld scenario as unsupported, three catalog scenarios, and one hidden
intent from each of two catalog scenarios as novel-valid. Retain at least three declared intents.
Known familiar/unfamiliar separation uses current-utterance overlap with normalized intent identifiers,
and source partitions remain example-disjoint.

For PRESTO, admit only `en-US`, `metadata.context == "human"` examples with nonempty human context and a
machine-verifiable target argument absent from the current input but present in previous turns or seeded
contacts/lists/notes. The paired full-context and ablated-context records must share the same utterance,
target, and identifier. Synthetic-context examples are prohibited from evaluation.

Neither source may be downloaded until its own archive lock, parser, text-free inventory, gates, tests,
and verifier are frozen. This selection authorizes no population, language extraction, model inference,
API use, training, planning, execution, or learned likelihood.
