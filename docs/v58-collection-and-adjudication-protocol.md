# V58 blinded human collection and adjudication protocol

This protocol operationalizes V58 without collecting any human-authored text. It preserves the V58 claim boundary: the concepts and executable ASTs come from the frozen V40 ontology, while only the surface wording is independently human-authored. V57's pass does not make V58 optional and does not authorize a joint new-wording/new-concept claim.

## Roles and privacy

The collection coordinator assigns pseudonymous writer IDs and retains the identity mapping outside this repository. Two pilot writers are disjoint from at least ten evaluation writers. Writers may not validate their own text. Candidate developers may not write, validate, or adjudicate evaluation text. Each submission is reviewed independently by two validators; a third human adjudicates disagreements. Validators and the adjudicator are blind to writer identity and candidate output, and validators are blind to one another's records.

No names, email addresses, demographic attributes, or other personally identifying information belong in the dataset. Every submission includes a human-authorship, no-generative-assistance, right-to-contribute, research-consent, and CC-BY-4.0 attestation. The attestation is necessary provenance rather than proof that authorship can be inferred from text.

## Construction holdout and quotas

The construction-family split is fixed without looking at human text. Families are ranked by SHA-256 of the V58 design-lock payload, the literal string `|pilot-family|`, and the family name. The first five are pilot-exposed: distractor scope, argument reversal, contrastive focus, inverse relation, and direct relation. The remaining five—denial, lexical negation, plain assertion, unresolved status, and double denial—are evaluation-only and receive zero pilot text.

Each pilot writer supplies 60 accepted primary utterances, 12 for each pilot-exposed family. Pilot text may support development but never evaluation. Each evaluation writer supplies 60 accepted primary utterances, exactly six per each of the ten families, for at least 600 balanced primary items. Each evaluation writer also supplies ten accepted abstention items, one per family, for at least 100 items in a separate safety stratum. Rejected or withdrawn items never count. If an evaluation writer withdraws, all of that writer's items are removed and a new disjoint writer replaces the entire quota.

## Prompt and submission boundary

Prompt meanings are sampled from the sealed V40 population. Author packets may show a role-labeled target meaning, entity legend, known-ontology glossary, and required construction family. They never show the original V40 evidence sentence, oracle metadata, or a reference surface realization. Writers produce one natural utterance, preserve sign, outer operation, and relation role order, avoid meta-commentary or AST field names, and do not use a language model or other generative writing tool.

The primary prompts require a unique expression of their target AST. The safety prompts alternate between genuine ambiguity and unsupported or unknown reference, balanced across authors and families. Safety writers must not announce that the sentence is ambiguous; these records are accepted only when human validation confirms that no unique supported AST exists.

## Validation and adjudication

The two validators see the submitted text, entity and ontology legend, prompted target or safety condition, and assigned construction family. They independently label one of: equivalent and unique, not equivalent, ambiguous, unsupported, or malformed/meta. They also record an inferred AST when one exists, whether the construction family was realized, and relation order when applicable.

Raw validator agreement is exact pre-adjudication verdict agreement over every double-coded submission and must be at least 0.90 before a population may be sealed. Any verdict, AST, construction-family, or relation-order disagreement requires third-person adjudication. A primary item is accepted only if the final verdict is uniquely equivalent, the final AST exactly equals the prompt target, and the assigned construction is realized. A safety item is accepted only if its final verdict is ambiguous or unsupported and no unique supported AST exists. Unadjudicated disagreements are never accepted.

## Freeze and release order

The protocol is frozen first, then text-free pilot and evaluation packets are generated and hashed. Only pilot packets may initially be released. After pilot collection and validation, the candidate parser, all baselines, prompts, and scoring implementation must be frozen. Evaluation packet hashes are rechecked before evaluation packets are released. Evaluation text is then collected and validated by the collection team without candidate access. Only after the author/family census, provenance, agreement, adjudication, and leakage audits pass may primary and abstention populations be sealed and a one-shot evaluator authorized.

At the current stage, the only permitted next mutation is generation and audit of blinded text-free author packets. Releasing packets, collecting text, writing the candidate, or evaluating anything remains unauthorized.
