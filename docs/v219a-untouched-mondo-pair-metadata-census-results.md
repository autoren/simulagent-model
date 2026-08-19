# V219A untouched Mondo pair metadata census results

## Outcome

V219A is a positive metadata-only result. The prospectively frozen census evaluated the five releases in the existing,
content-hashed V217A Mondo GitHub API snapshot. It made no network request, read no new payload body or ontology record,
and did not inspect any V218 development or protected record.

The frozen outcome branch is `UNTOUCHED_MONDO_PAIR_PAYLOAD_DESIGN_ELIGIBLE`.

## Exact selection

Both fully untouched adjacent pairs passed every preregistered metadata gate:

| Older release | Newer release | Required asset bytes | Official body categories | Eligible |
| --- | --- | ---: | ---: | --- |
| `v2026-05-05` | `v2026-06-02` | 96,095,320 | 4/4 | yes |
| `v2026-04-07` | `v2026-05-05` | 95,309,262 | 4/4 | yes |

The deterministic newest-pair-first rule selected `v2026-05-05__to__v2026-06-02`. Adjacency was computed in the
original full release order before excluding the V218-opened releases `v2026-07-06` and `v2026-08-04`.

The selected newer release's exact API body hash is
`76b1133b7196e1547ca91470bcbb8e59422e8fe2fe815e7ebab250473d72050a`. The body contains all four frozen categories:
addition, synonym/label change, text-definition change, and obsoletion/replacement. A repository README was not used.

## Selected asset contract

The selected pair has exactly the eight required metadata-attested roles:

| Role | Release | Asset | Bytes | Published SHA-256 |
| --- | --- | --- | ---: | --- |
| older source | `v2026-05-05` | `mondo-base.obo` | 47,474,428 | `57e4563920b922dc952ed954d36368319592f3a32946504728665c248000a3ac` |
| newer target | `v2026-06-02` | `mondo-base.obo` | 48,298,257 | `363a2dd9e653e12db0182332a974a6e3c3621a4556ab32c70eaa592cd37a376d` |
| changed-term control | `v2026-06-02` | `mondo_release_diff_changed_terms.tsv` | 5,891 | `63aa226f56bd5ab6aa94c65a02c3c249b9cc5bfe7155efa9bcb65e8d1568a553` |
| new-term control | `v2026-06-02` | `mondo_release_diff_new_terms.tsv` | 125,701 | `1fc7c53bed11c3a96732dba0ff6fd5655c5a1d8138b4e8ae0f874fb67ce677c3` |
| older candidate-status control | `v2026-05-05` | `mondo_obsoletioncandidates.tsv` | 94,222 | `c4c7d52941d4478cd9392ef82530394e4e481c0a86b5b5ad54474a2de1835c67` |
| newer candidate-status control | `v2026-06-02` | `mondo_obsoletioncandidates.tsv` | 95,091 | `392880f93470a3f361efa408a9fc1202a1f294309a1dcada64075ad474039a74` |
| older source-version control | `v2026-05-05` | `source-versions.tsv` | 865 | `35e627f3985481a198e373ccb4c747198597f71308a66691fd1eb21031f72c45` |
| newer source-version control | `v2026-06-02` | `source-versions.tsv` | 865 | `c5094c1125b4f9a578e29e8fc82550fadcb797bd15c295545fd553f65546b840` |

## Gate results

Snapshot-hash accuracy, original-adjacency accuracy, pair-assessment coverage, asset-role coverage, published-digest
coverage, release-body-hash coverage, release-body-category coverage, and selection-priority correctness were all
`1.0`. Exactly two untouched adjacent pairs were assessed and exactly one was selected. All metrics were finite.

Every forbidden access counter remained zero: network, payload body, ontology records, V218 development/protected,
V216/V213 protected, deterministic method evaluation, model use, training, registration, mutation, service action,
external effect, and execution.

## Interpretation and authorization

This result repairs only the experimental design path that V218 exposed: the published release-summary control must be
the official API release body, not a README asset. It does not repair or reinterpret V218, and it does not establish
payload integrity, parser correctness, population feasibility, semantic equivalence, or new-speaker intent.

V219A authorizes one separately frozen payload-design protocol for the selected untouched pair. It does not authorize
retrieval until that design passes a prospective audit, and it does not authorize protected evaluation or a model.

## Frozen artifacts

- Design lock: `configs/v219a-untouched-mondo-pair-metadata-census-lock.json`
- Evidence: `outputs/v219a-untouched-mondo-pair-metadata-census/census/evidence.json`
- Summary: `outputs/v219a-untouched-mondo-pair-metadata-census/census/summary.json`
- Result: `outputs/v219a-untouched-mondo-pair-metadata-census/census/result.json`

