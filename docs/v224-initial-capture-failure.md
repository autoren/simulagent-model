# V224 initial capture failure

The first capture attempt retrieved and stored the pinned Mondo scope-policy document, then stopped on the first
GraphQL record page before GitHub returned any record metadata.

The `gh api graphql` command uses a form field named `query` for the GraphQL document. The frozen V224 document also
declares a variable named `query`. The capture implementation attempted to pass both as repeated `-f query=...`
arguments, so the CLI could not encode the document and variable separately and exited with `CalledProcessError`.

Failure boundary:

- successful record GraphQL response count: 0;
- formal record metadata read count: 0;
- issue or pull title/body read count: 0;
- comment or review-text read count: 0;
- completed search slices: 0;
- scope-policy document retrieval count: 1;
- scope-policy snapshot SHA-256: `4773e510e784cdf7063743f010f5c3a0f8b21d2e55d73f0d2e5f38893bfa234e`;
- census, release, deep-audit, summary, and result artifacts written: 0; and
- model, protected-record, training, registration, action, side-effect, and execution counts: 0.

No scientific query field, source window, exclusion, disposition mapping, sample rule, threshold, or decision rule was
evaluated or changed.

