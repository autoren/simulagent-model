#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any

from retrieve_v220_fresh_mondo_artifacts import dependency_hashes_exact
from v10_protocol import file_sha256
from v218_mondo_artifact_population import event_types, load_obo, parse_tsv
from v220_fresh_mondo_artifact_population import build_population_records
from v22r2_grounding import PROJECT_ROOT


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def write_jsonl(path: Path, values: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(value, sort_keys=True) + "\n" for value in values))


def official_release_body_control(config: dict[str, Any]) -> dict[str, Any]:
    control = config["releaseSummaryControl"]
    snapshot_path = PROJECT_ROOT / control["snapshotPath"]
    if file_sha256(snapshot_path) != control["snapshotSha256"]:
        raise RuntimeError("release metadata snapshot hash mismatch")
    releases = json.loads(snapshot_path.read_text())
    matches = [release for release in releases if release.get("tag_name") == control["newerReleaseTag"]]
    if len(matches) != 1 or not isinstance(matches[0].get("body"), str):
        raise RuntimeError("exact newer release body is unavailable")
    body = matches[0]["body"]
    observed_hash = hashlib.sha256(body.encode("utf-8")).hexdigest()
    categories = sorted(
        category for category, pattern in control["categoryPatterns"].items()
        if re.search(pattern, body)
    )
    return {
        "parse_success": bool(body.strip()),
        "body_sha256": observed_hash,
        "body_hash_matches_frozen_control": observed_hash == control["bodySha256"],
        "categories": categories,
        "repository_readme_used": False,
        "network_request_count": 0,
    }


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: build_v220_fresh_mondo_artifact_population.py LOCK")
    lock = json.loads(Path(sys.argv[1]).resolve().read_text())
    if not dependency_hashes_exact(lock):
        raise RuntimeError("V220 design lock or dependency hash mismatch")
    if not lock["authorization"]["build_one_role_separated_population"]:
        raise RuntimeError("V220 population build is not authorized")
    config = lock["config_payload"]
    artifacts = {key: PROJECT_ROOT / value for key, value in config["artifacts"].items()}
    output_keys = ["parserControl", "developmentPublic", "developmentTruth", "protectedPublic", "protectedTruth", "split", "populationManifest"]
    if any(artifacts[key].exists() for key in output_keys):
        raise RuntimeError("V220 population output already exists")
    retrieval = json.loads(artifacts["retrievalManifest"].read_text())
    attempt_by_id = {row["payload_id"]: row for row in retrieval["attempts"]}
    payload_by_id = {payload["payloadId"]: payload for payload in config["payloads"]}
    for payload_id, payload in payload_by_id.items():
        row = attempt_by_id[payload_id]
        raw = PROJECT_ROOT / payload["rawPath"]
        if not row["success"] or not raw.is_file() or file_sha256(raw) != row["sha256"] == payload["declaredSha256"]:
            raise RuntimeError(f"unfrozen or invalid raw payload: {payload_id}")
    population = config["populationDesign"]
    older_terms = load_obo(PROJECT_ROOT / payload_by_id[population["olderPayloadId"]]["rawPath"])
    newer_terms = load_obo(PROJECT_ROOT / payload_by_id[population["newerPayloadId"]]["rawPath"])
    changed_control = parse_tsv(PROJECT_ROOT / payload_by_id[population["changedControlPayloadId"]]["rawPath"])
    new_control = parse_tsv(PROJECT_ROOT / payload_by_id[population["newControlPayloadId"]]["rawPath"])
    older_candidate_control = parse_tsv(PROJECT_ROOT / payload_by_id[population["olderCandidatePayloadId"]]["rawPath"])
    newer_candidate_control = parse_tsv(PROJECT_ROOT / payload_by_id[population["newerCandidatePayloadId"]]["rawPath"])
    source_controls = [
        parse_tsv(PROJECT_ROOT / payload["rawPath"])
        for payload in config["payloads"]
        if payload["role"] in {"OLDER_SOURCE_PROVENANCE_CONTROL", "NEWER_SOURCE_PROVENANCE_CONTROL"}
    ]
    release_control = official_release_body_control(config)
    older_candidates = set(older_candidate_control["mondo_ids"])
    newer_candidates = set(newer_candidate_control["mondo_ids"])
    all_ids = set(older_terms) | set(newer_terms) | older_candidates | newer_candidates
    changed_ids = {
        term_id for term_id in all_ids
        if event_types(older_terms.get(term_id), newer_terms.get(term_id), term_id in older_candidates, term_id in newer_candidates, config)
    }
    added_ids = set(newer_terms) - set(older_terms)
    new_control_ids = set(new_control["mondo_ids"])
    changed_control_ids = set(changed_control["mondo_ids"])
    required_categories = set(config["controlAgreement"]["releaseSummaryRequiredCategories"])
    tabular_validations = [
        changed_control["parse_success"], new_control["parse_success"],
        older_candidate_control["parse_success"] and older_candidate_control["mondo_id_unique"],
        newer_candidate_control["parse_success"] and newer_candidate_control["mondo_id_unique"],
        *(control["parse_success"] and control["row_count"] > 0 for control in source_controls),
    ]
    parser_control = {
        "schema_version": "220-fresh-mondo-parser-and-published-control",
        "older_term_count": len(older_terms), "newer_term_count": len(newer_terms),
        "unique_term_id_rate": 1.0, "remote_import_resolution_count": 0,
        "tabular_control_parse_rate": sum(tabular_validations) / len(tabular_validations),
        "new_term_control_agreement": 1.0 if new_control_ids == added_ids else 0.0,
        "parsed_added_id_count": len(added_ids), "new_control_id_count": len(new_control_ids),
        "changed_term_control_precision": len(changed_control_ids & changed_ids) / len(changed_control_ids) if changed_control_ids else 0.0,
        "parsed_changed_id_count": len(changed_ids), "changed_control_id_count": len(changed_control_ids),
        "release_summary_category_coverage": len(required_categories & set(release_control["categories"])) / len(required_categories),
        "release_summary_categories": release_control["categories"],
        "release_summary_body_sha256": release_control["body_sha256"],
        "release_summary_body_hash_matches_frozen_control": release_control["body_hash_matches_frozen_control"],
        "release_summary_repository_readme_used": release_control["repository_readme_used"],
        "release_summary_network_request_count": release_control["network_request_count"],
        "older_obsoletion_candidate_count": len(older_candidates),
        "newer_obsoletion_candidate_count": len(newer_candidates),
        "source_version_control_row_counts": [control["row_count"] for control in source_controls],
        "asserted_state_is_inferred_owl_equivalence": False,
    }
    write_json(artifacts["parserControl"], parser_control)
    public_records, truth_records, build_manifest = build_population_records(
        older_terms, newer_terms, older_candidates, newer_candidates, config
    )
    truth_by_id = {record["case_id"]: record for record in truth_records}
    development_public = [record for record in public_records if record["split"] == "DEVELOPMENT"]
    protected_public = [record for record in public_records if record["split"] == "PROTECTED"]
    development_truth = [truth_by_id[record["case_id"]] for record in development_public]
    protected_truth = [truth_by_id[record["case_id"]] for record in protected_public]
    write_jsonl(artifacts["developmentPublic"], development_public)
    write_jsonl(artifacts["developmentTruth"], development_truth)
    write_jsonl(artifacts["protectedPublic"], protected_public)
    write_jsonl(artifacts["protectedTruth"], protected_truth)
    split = {
        "schema_version": "220-fresh-mondo-concept-family-split",
        "assignment": config["splitDesign"]["assignment"],
        "development_group_ids": build_manifest["development_group_ids"],
        "protected_group_ids": build_manifest["protected_group_ids"],
        "cross_split_group_overlap_count": len(set(build_manifest["development_group_ids"]) & set(build_manifest["protected_group_ids"])),
        "protected_downstream_method_evaluation_count": 0,
        "protected_manual_semantic_inspection_count": 0,
        "protected_file_load_for_scoring_count": 0,
    }
    write_json(artifacts["split"], split)
    population_manifest = {
        **build_manifest,
        "experiment": config["experiment"],
        "retrieval_manifest_sha256": file_sha256(artifacts["retrievalManifest"]),
        "parser_control_sha256": file_sha256(artifacts["parserControl"]),
        "raw_payload_sha256": {payload_id: attempt_by_id[payload_id]["sha256"] for payload_id in sorted(payload_by_id)},
        "public_truth_written_separately": True,
        "protected_partition_not_used_for_method_evaluation": True,
        "protected_partition_not_manually_semantically_inspected": True,
        "protected_files_not_loaded_for_scoring": True,
        "protected_public_sha256": file_sha256(artifacts["protectedPublic"]),
        "protected_truth_sha256": file_sha256(artifacts["protectedTruth"]),
    }
    write_json(artifacts["populationManifest"], population_manifest)
    print(json.dumps({"parser_control": parser_control, "population_manifest": population_manifest}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
