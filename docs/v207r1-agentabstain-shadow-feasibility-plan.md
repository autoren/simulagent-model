# V207r1 AgentAbstain transport-repair plan

## Scope

V207r1 repairs only the failed Hugging Face tree request in V207. V207 requested 10,000 entries in one page; the endpoint accepts at most 1,000 and returned HTTP 400 before a scientific result existed. The locked V207 source revisions, metadata patterns, pair and pre-execution definitions, qualification thresholds, contamination treatment, future selection rule, and no-language/no-model/no-execution boundary remain unchanged.

## Repair

The dataset-tree census starts with a 1,000-entry page and follows only explicit `rel="next"` links. Every next URL must use HTTPS, remain on `huggingface.co`, and preserve the pinned dataset-revision path. Repeated URLs are rejected, the number of physical pages is bounded at 20, and the census qualifies only if it reaches a terminal page without a next link. The result records page counts and hashes, never response bodies or cursor URLs.

The original V207 requirement of one dataset-tree census is preserved as one logical census. Physical page reads are separately and honestly counted; they must fall within the preregistered pagination bounds.

## Prior exposure

The V207 run and transport diagnostics read only code-tree metadata, eight allowed code schema files, the dataset HEAD, and dataset-tree object metadata. No dataset payload, task instruction, example, dialogue, rationale, dataset-card body, model, tool, service, or action was accessed. The exact cumulative counts are frozen in the V207 transport-failure record and the V207r1 design.

## Decision

Passing V207r1 authorizes only a separately locked deterministic text-extraction design. It does not authorize opening task language or running a model. A scientific gate failure freezes a negative result without weakening any gate; another transport failure remains a technical non-result.
