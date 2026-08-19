# V216 bounded external-artifact population results

## Bottom line

V216 selected the frozen negative branch `NEGATIVE_EXTERNAL_PAYLOAD_OR_POPULATION_FEASIBILITY` because one
noncompensatory parser gate failed: each Uberon release was required to contain at least 20,000 parsed `[Term]`
stanzas, but `uberon-basic.obo` contained 15,719 terms in `v2025-05-28` and 15,763 in `v2025-08-15`.

That gate is not being changed after seeing the payload. V216 therefore does not authorize V217 deterministic methods,
protected evaluation, or any model run.

## Retrieval and provenance

All three lock-authorized payloads were retrieved exactly once and preserved raw:

| Payload | Bytes | SHA-256 |
|---|---:|---|
| Uberon basic `v2025-05-28` | 11,890,018 | `67fa31d176195f27cc92454a58e2a5210e21f4e820a719b4f762911d1e379432` |
| Uberon basic `v2025-08-15` | 11,940,815 | `9cb9db511e9d1d1d411902084eda676bdb6750f3e3009999cab30fe13836a452` |
| W3C OWL 2 archived `all.rdf` | 3,093,215 | `5383f1ddf4cf2f03703a2f886f41d4e5bc375633a1cfa94a03254fd89330f8bb` |

The total was 26,924,048 bytes, below the frozen 28 MB ceiling. Exact payload accounting, raw-hash coverage, expected
byte counts, and the one published digest check were all `1.0`. No unlisted request was made.

## Parser and population findings

The W3C file parsed successfully as RDF/XML and exposed 504 distinct RDF subjects. This validates the bounded XML
parsing path only; it is not OWL-reasoner evidence.

The two OBO releases yielded the following raw change census:

- 44 added terms;
- 158 terms with changed asserted logical fields; and
- one term with a changed definition.

After the frozen requirements for a current name, current definition, current asserted logical axiom, and an eligible
change, 197 records in 197 observation groups remained. The split assigned 148 groups to development and 49 to the
protected partition. The records covered three primary change types: 44 added, 152 logical-axiom changes, and one
definition change.

All substantive population diagnostics passed:

- current-definition coverage: `1.0`;
- current-asserted-axiom coverage: `1.0`;
- exact version-space reconstruction: `1.0`;
- boundary-witness coverage: `1.0`;
- cross-split group overlap: `0`;
- duplicate case IDs: `0`; and
- public source-identifier leakage: `0`.

No identical public observation mapped to multiple asserted-axiom classes, so this release pair produced zero
empirically ambiguous groups under the frozen observation representation.

## Interpretation

This is a formal negative caused by a preregistered source-size proxy, not by payload corruption or an inadequate
derived population. The observed 197-record population substantially exceeded the direct record/group/split gates.
Nevertheless, treating those passing downstream metrics as permission to ignore the failed 20,000-term gate would be
post-hoc threshold relaxation.

The result exposes two useful design lessons:

1. Total ontology term count is a poor proxy for the number of eligible paired reconstruction events. Future source
   selection should preregister observable release metadata or pilot-independent event evidence rather than an
   arbitrary global term threshold.
2. This particular release pair is rich in asserted logical changes but contains almost no definition change and no
   public-observation ambiguity. Even absent the size failure, it would be better suited to version-diff controls than
   to a broad open-world language claim.

## Claim and access boundary

V216 reconstructed normalized asserted OBO fields; it did not compute inferred OWL equivalence or recover complete
curator rationale. The protected partition was structurally verified but not used for downstream method evaluation or
manually inspected. No V213 protected data, model, API model, training, registration, trusted mutation, service action,
or execution was used.

Because the original outcome verifier incorrectly required a positive scientific audit even though the design also
specified freezing negative outcomes, a separate verification-only repair is required. That repair may validate and
freeze this exact negative result; it may not rerun retrieval, rebuild records, change a gate, or authorize V217.

