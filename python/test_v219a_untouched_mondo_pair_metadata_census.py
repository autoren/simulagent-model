from __future__ import annotations

from copy import deepcopy

from v219a_untouched_mondo_pair_metadata_census import (
    assess_pair,
    build_census,
    enumerate_untouched_adjacent_pairs,
    release_body_categories,
)


BODY = """
<summary>New terms:</summary>
<summary>Terms renamed</summary>
<summary>Text definitions added</summary>
<summary>Terms obsoleted with replacement</summary>
"""


def _config() -> dict:
    roles = []
    for side, prefix in (("OLDER", "old"), ("NEWER", "new")):
        for suffix in ("source", "candidate", "provenance"):
            roles.append({"role": f"{prefix}_{suffix}", "releaseSide": side, "assetName": f"{suffix}.tsv", "format": "TSV"})
    roles.extend(
        [
            {"role": "changed", "releaseSide": "NEWER", "assetName": "changed.tsv", "format": "TSV"},
            {"role": "added", "releaseSide": "NEWER", "assetName": "added.tsv", "format": "TSV"},
        ]
    )
    return {
        "experiment": "test",
        "evidenceSource": {"sourceId": "snapshot", "path": "snapshot.json", "sha256": "a" * 64},
        "pairEnumeration": {
            "excludedReleaseTags": ["v5", "v4"],
            "expectedUntouchedAdjacentPairCount": 2,
            "selectFirstEligiblePair": True,
        },
        "requiredAssetRoles": roles,
        "assetRequirements": {
            "requiredAssetRoleCount": 8,
            "maximumSingleAssetBytes": 100,
            "maximumPairPayloadBytes": 800,
        },
        "releaseBodyControl": {
            "categoryPatterns": {
                "ADDITION": "(?i)<summary>(new terms|classes added):",
                "SYNONYM_OR_LABEL": "(?i)<summary>terms renamed",
                "TEXT_DEFINITION": "(?i)<summary>text definitions? (added|changed)",
                "OBSOLETION_OR_REPLACEMENT": "(?i)<summary>terms obsoleted with replacement",
            }
        },
    }


def _release(tag: str, day: int, config: dict) -> dict:
    names = sorted({role["assetName"] for role in config["requiredAssetRoles"]})
    return {
        "tag_name": tag,
        "published_at": f"2026-01-{day:02d}T00:00:00Z",
        "body": BODY,
        "assets": [
            {
                "name": name,
                "browser_download_url": f"https://example.org/{tag}/{name}",
                "size": 10,
                "digest": "sha256:" + (f"{index + 1:064x}"),
            }
            for index, name in enumerate(names)
        ],
    }


def test_original_adjacency_is_computed_before_exclusion_and_newest_pair_wins() -> None:
    config = _config()
    releases = [_release(f"v{day}", day, config) for day in range(1, 6)]
    pairs, order = enumerate_untouched_adjacent_pairs(releases, config)
    assert order == ["v5", "v4", "v3", "v2", "v1"]
    assert [pair["pair_id"] for pair in pairs] == ["v2__to__v3", "v1__to__v2"]
    evidence, metrics = build_census(releases, config, snapshot_hash_accurate=True)
    assert metrics["eligible_pair_count"] == 2
    assert evidence["selected_pair_ids"] == ["v2__to__v3"]


def test_release_body_category_parser_requires_all_four_categories() -> None:
    config = _config()
    assert all(release_body_categories(BODY, config).values())
    missing = release_body_categories(BODY.replace("Terms renamed", "Other changes"), config)
    assert not missing["SYNONYM_OR_LABEL"]


def test_missing_digest_or_body_category_makes_pair_ineligible() -> None:
    config = _config()
    releases = [_release(f"v{day}", day, config) for day in range(1, 4)]
    pairs, _ = enumerate_untouched_adjacent_pairs(releases, {**config, "pairEnumeration": {**config["pairEnumeration"], "excludedReleaseTags": []}})
    damaged = deepcopy(pairs[0])
    damaged["newer_release"]["assets"][0]["digest"] = None
    assert not assess_pair(damaged, config)["eligible"]
    damaged = deepcopy(pairs[0])
    damaged["newer_release"]["body"] = BODY.replace("Terms renamed", "Other changes")
    assert not assess_pair(damaged, config)["eligible"]
