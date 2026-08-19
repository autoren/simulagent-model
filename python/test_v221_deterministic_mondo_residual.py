from __future__ import annotations

from v221_deterministic_mondo_residual import (
    build_catalog,
    controller_decision,
    derive_role_manifest,
    generate_candidates,
    normalize_surface,
)


def _config() -> dict:
    return {
        "experiment": "v221_test",
        "roleSplit": {"assignment": "test"},
        "catalogDesign": {
            "sourcePopulationExperiment": "v220_prospective_fresh_mondo_artifact_population",
            "stateClassDefinition": "test",
        },
        "parserDesign": {
            "logicalFields": ["is_a", "relationship", "intersection_of", "equivalent_to", "disjoint_from"],
            "mappingFields": ["xref"],
        },
    }


def test_role_split_is_hash_derived_disjoint_and_complete() -> None:
    groups = [f"G_{index:04d}" for index in range(10)]
    manifest = derive_role_manifest(groups, _config())
    assert len(manifest["evaluation_group_ids"]) == 2
    assert len(manifest["calibration_group_ids"]) == 8
    assert manifest["group_overlap_count"] == 0
    assert manifest["source_group_accounting_exact"]


def test_surface_normalization_is_unicode_casefolded_and_punctuation_stable() -> None:
    assert normalize_surface("  Café—Syndrome! ") == "café syndrome"


def test_family_expansion_is_atomic_and_final_controller_fails_closed_on_overflow() -> None:
    config = _config()
    older = {
        "MONDO:1": {"id": ["MONDO:1"], "name": ["Alpha condition"]},
    }
    newer = {
        "MONDO:1": {"id": ["MONDO:1"], "name": ["Alpha condition"], "is_a": ["MONDO:2"]},
    }
    catalog = build_catalog(older, newer, config)
    public = {"surface_text": "Alpha condition", "evidence_mode": "VERSION_UNSPECIFIED"}
    generated = generate_candidates(public, catalog, "M3_FINAL_FAIL_CLOSED", 1)
    assert generated["overflow"]
    assert generated["candidate_class_ids"] == []
    decision, fail_closed = controller_decision(public, generated, catalog, "M3_FINAL_FAIL_CLOSED")
    assert decision == "PRESERVE_VERSION_SPACE_OR_CLARIFY"
    assert fail_closed

