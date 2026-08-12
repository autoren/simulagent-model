# V21r2 amendment: prompt-inference budget only

The original V21 design lock capped new prompt inferences at 2,000. After V20 evaluation—but before
any V21 seed draw, record construction, model inference, or final-data access—a deterministic
prompt-inventory preflight found that the locked language topology produces exactly 5,136 unique
prompts: 1,712 base prompts and 3,424 NLI prompts.

The count was reproduced with four explicitly ineligible test seeds. It is invariant because the
surface and evidence-order policies depend on construction family and semantic grounding rather
than the selected mechanic. The original cap was therefore incompatible with the already-locked
requirement to retain all nine V14 surface families and both paired ontology views.

V21r2 changes exactly one config value:

- `limits.maximumNewModelForwardPasses`: 2,000 → 5,200.

The generator, family predicates, episode quotas, language renderer, delayed seed rule, systems,
metrics, thresholds, decision hierarchy, and no-retry policy remain byte-identical. The cap counts
unique prompt examples, not batched device calls. No prompt is omitted or grouped merely to satisfy
the resource gate.

The V20 result was known when this amendment was written and is disclosed in the amendment lock.
It cannot affect this change: V20 uses saved V19 features, while the amendment only reconciles a
deterministic V21 prompt count with the extraction budget. The V21 suite still does not exist.
