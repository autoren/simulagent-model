# Cross-track evidence audit through V224/V224r2

## Purpose

This is a documentation and reproducibility audit, not V225 and not a new language or model experiment. It reconstructs
the frozen evidence chain from the outcome locks currently present in the repository and tests what can and cannot be
composed into one uncertainty-aware decision architecture.

## Frozen scope

- enumerate every `configs/v*-outcome-lock.json` from V33 through V224;
- validate each lock's canonical payload hash;
- link each outcome to exact referenced and same-version design locks;
- verify hashes only for non-sensitive referenced dependencies;
- record, but do not rehash, protected/raw/holdout/record-language/model-weight bodies;
- preserve implementation repairs as repairs rather than count them as independent scientific replication;
- classify the 17 preregistered experiment families in
  `configs/cross-track-evidence-audit-through-v224.json`;
- reconstruct the frozen critical chain using exact outcome-lock hashes;
- identify composable mechanisms, external validation, interface diagnostics, retrospective evidence, and source
  failures separately; and
- authorize at most one successor only if every frozen next-experiment condition passes.

## Prohibited actions

The audit may not open protected record bodies or request language, rehash declared protected data, run a local/API
model, train or tune, register concepts, mutate trusted state, call a service, act, or execute.

## Outputs

- `outputs/cross-track-evidence-audit-through-v224/experiment-ledger.json`
- `outputs/cross-track-evidence-audit-through-v224/family-ledger.json`
- `outputs/cross-track-evidence-audit-through-v224/reproducibility-audit.json`
- `outputs/cross-track-evidence-audit-through-v224/critical-chain.json`
- `outputs/cross-track-evidence-audit-through-v224/claim-and-risk-matrix.json`
- `docs/cross-track-evidence-synthesis-through-v224.md`
- `docs/research-stopping-rule-after-v224.md`
- `configs/cross-track-evidence-audit-through-v224-outcome-lock.json`

An unfrozen result document is never promoted to frozen evidence. Missing outcome versions and dependency drift are
reported as gaps; historical files are not silently rewritten to make the audit pass.
