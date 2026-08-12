# V21 preregistration: sealed population multi-mechanic final

## Confirmatory claim

V21 tests whether the frozen V19 hard pipeline transfers across a declared population of unseen
one-step mechanics under a four-variable Boolean ontology and supported language. It does not test
ontology discovery, persistent next-state mutation, sequential planning, causal discovery, or a
general world model.

The hard system is primary confirmatory. The V20 probabilistic interface is a separately
preregistered challenger only if it preserves supported-language development before V21 execution
is locked. Challenger success cannot retroactively rescue a failed hard-system claim.

## Population and estimand

The suite contains exactly 40 mechanics, stratified across four construction families:

| Family | Mechanics | Bits | Relevant-projection behavior |
|---|---:|---:|---|
| `primitive_one_bit` | 8 | 1 | injective |
| `composed_one_bit` | 12 | 1 | non-injective |
| `factorized_two_bit` | 12 | 2 | injective |
| `nested_two_bit` | 8 | 2 | non-injective |

Injectivity is defined after projecting the 16 assignments onto semantically relevant determinants:
distinct relevant assignments must produce distinct visible codes. This makes one-variable one-bit
and two-variable two-bit bijections injective while deeper many-to-one programs are non-injective.

The construction families are independently specified strata inside one implementation, not
independent benchmark authors. Results support inference only to the fixed declared mixture. V21
reports the stratified overall score, family-macro score, every family separately, and an exact
binomial interval for complete mechanics as a descriptive exchangeability calculation—not as a
claim about arbitrary mechanics.

Every target behavior is absent from V18. No V17 records or scores are read. Candidate programs are
selected by hashing a delayed seed with their complete behavioral signatures; no model result,
language score, or target difficulty measure participates in selection.

## Delayed construction firewall

The config, generator, construction-family predicates, language renderer, validation tests, metrics,
gates, and seed policy are locked before V20 is evaluated. Final records do not yet exist.

After V20 is evaluated, the eligible systems and all extraction/evaluation code are frozen in a
separate execution lock. Only then may the constructor draw one 256-bit seed from the operating
system CSPRNG. It writes the seed and its hash into the immutable manifest and refuses to run if the
output directory exists. Test seeds are marked ineligible and can never become final seeds.

There is one construction, one feature extraction, one evaluation, one independent replay, and no
retry. Structural failure terminates the final attempt; it does not authorize another seed.

## Language intervention

Each latent item has exactly paired views.

- `supported` uses the four V15-registered canonical state concepts and is primary.
- `novel_ontology` uses a fresh ontology assigned by construction family and is non-gating.

Across items, the supported view deterministically covers all nine V14 surface families: three
affirmative-gold, three negated-opposite, and three contrastive-both surfaces. Unresolved queries
cover unknown, stale-only, and conflicting-current evidence. Surface, unresolved mode, and evidence
order are functions of the semantic grounding and family, not of model output or mechanic identity.
This both preserves exact prompt deduplication and prevents a mechanic-specific language shortcut.

The two views preserve the program, support assignments, observed transition codes, query allowed
states, and target answers exactly. Only the concept ontology and its phrases change.

## Mechanics and executor boundary

Programs remain inside the frozen V18 Boolean grammar and produce one or two visible outcome bits.
An action may describe a visible post-action reading, but V21 does not update persistent state across
steps. Truly sequential mutation requires a different executor and belongs to the later relational
benchmark.

Each support set is greedily behavior-identifying under oracle grounding. Queries include every
unseen complete assignment and every single-variable unresolved assignment. The agent sees support
observations and visible transition codes but never receives the target program, behavioral
signature, relevant variables, or action-dependency table.

## Metrics and gates

The mechanic is the primary unit. Query-pooled results are diagnostics. V21 reports mechanic-macro
transition-set exact match, complete mechanics, family-macro and per-family results, worst mechanic,
target retention, empty version/posterior rate, identifiability, invariant/sensitive unknowns, and
the exact interval for the complete-mechanic proportion.

The primary hard supported system passes only if:

- mechanic-macro exact match is at least 0.95;
- at least 38/40 mechanics are completely correct;
- family complete counts are at least 7/8, 11/12, 11/12, and 7/8 respectively;
- target behavior remains after all supports in at least 0.95 of mechanics;
- empty version spaces occur in at most 0.05; and
- the oracle-grounded ceiling is exactly 1.0.

Under an exchangeable 40-trial interpretation, 38/40 complete mechanics has a two-sided 95% exact
lower bound above 0.80. Because V21 is stratified and generator-bound, the raw and per-family counts
remain the authoritative description.

If eligible, the probabilistic supported challenger has a separate claim with the same macro,
complete, and retention floors, empty-posterior rate at most 0.05, and mean excess outcomes at most
0.10 per query. Novel-ontology outcomes never affect either supported-language gate.

## Decisions

- Hard pass: the current architecture transfers within the declared population; proceed to a new
  relational/sequential benchmark.
- Hard fail and probabilistic pass: reject the hard-interface claim and retain structured modularity
  with propagated uncertainty as the next hypothesis.
- Both supported systems fail: reject population robustness and diagnose family, surface, grounding,
  and DSL compatibility without adapting on V21.
- Supported pass with novel failure: retain the supported claim and pursue definitions, retrieval,
  or ontology anchoring on new development ontologies before weight adaptation.

No V21 result authorizes a retry or LoRA on V21.
