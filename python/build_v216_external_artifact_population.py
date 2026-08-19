#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

from retrieve_v216_bounded_external_artifacts import dependency_hashes_exact
from v10_protocol import file_sha256
from v216_bounded_external_artifact_population import build_population_records, load_obo, w3c_rdfxml_control
from v22r2_grounding import PROJECT_ROOT


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def write_jsonl(path: Path, values: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(value, sort_keys=True) + "\n" for value in values))


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: build_v216_external_artifact_population.py LOCK")
    lock_path = Path(sys.argv[1]).resolve()
    lock = json.loads(lock_path.read_text())
    if not dependency_hashes_exact(lock):
        raise RuntimeError("V216 design lock or dependency hash mismatch")
    if not lock["authorization"]["build_one_role_separated_population"]:
        raise RuntimeError("V216 population build is not authorized")
    config = lock["config_payload"]
    artifacts = {key: PROJECT_ROOT / value for key, value in config["artifacts"].items()}
    output_keys = ["parserControl", "developmentPublic", "developmentTruth", "protectedPublic", "protectedTruth", "split", "populationManifest"]
    if any(artifacts[key].exists() for key in output_keys):
        raise RuntimeError("V216 population output already exists")
    manifest = json.loads(artifacts["retrievalManifest"].read_text())
    attempt_by_id = {row["payload_id"]: row for row in manifest["attempts"]}
    payload_by_id = {payload["payloadId"]: payload for payload in config["payloads"]}
    for payload_id, payload in payload_by_id.items():
        row = attempt_by_id[payload_id]
        raw = PROJECT_ROOT / payload["rawPath"]
        if not row["success"] or not raw.is_file() or file_sha256(raw) != row["sha256"]:
            raise RuntimeError(f"unfrozen or invalid raw payload: {payload_id}")
    older_payload = payload_by_id[config["populationDesign"]["olderPayloadId"]]
    newer_payload = payload_by_id[config["populationDesign"]["newerPayloadId"]]
    control_payload = next(payload for payload in config["payloads"] if payload["role"] == "RDF_XML_PARSER_VALIDATION_CONTROL")
    older_terms = load_obo(PROJECT_ROOT / older_payload["rawPath"], config["parserDesign"]["oboEncoding"])
    newer_terms = load_obo(PROJECT_ROOT / newer_payload["rawPath"], config["parserDesign"]["oboEncoding"])
    w3c = w3c_rdfxml_control(PROJECT_ROOT / control_payload["rawPath"])
    parser_control = {
        "schema_version": "216-parser-control",
        "older_term_count": len(older_terms),
        "newer_term_count": len(newer_terms),
        "w3c": w3c,
        "asserted_axiom_signature_is_inferred_semantic_equivalence": False,
        "w3c_control_is_reasoner_evidence": False,
    }
    write_json(artifacts["parserControl"], parser_control)
    public_records, truth_records, build_manifest = build_population_records(older_terms, newer_terms, config)
    development_public = [record for record in public_records if record["split"] == "DEVELOPMENT"]
    protected_public = [record for record in public_records if record["split"] == "PROTECTED"]
    truth_by_id = {record["case_id"]: record for record in truth_records}
    development_truth = [truth_by_id[record["case_id"]] for record in development_public]
    protected_truth = [truth_by_id[record["case_id"]] for record in protected_public]
    write_jsonl(artifacts["developmentPublic"], development_public)
    write_jsonl(artifacts["developmentTruth"], development_truth)
    write_jsonl(artifacts["protectedPublic"], protected_public)
    write_jsonl(artifacts["protectedTruth"], protected_truth)
    split = {
        "schema_version": "216-group-disjoint-split",
        "assignment": config["splitDesign"]["assignment"],
        "development_group_ids": build_manifest["development_group_ids"],
        "protected_group_ids": build_manifest["protected_group_ids"],
        "cross_split_group_overlap_count": len(set(build_manifest["development_group_ids"]) & set(build_manifest["protected_group_ids"])),
        "protected_downstream_method_evaluation_count": 0,
        "protected_manual_semantic_inspection_count": 0,
    }
    write_json(artifacts["split"], split)
    population_manifest = {
        **build_manifest,
        "experiment": config["experiment"],
        "schema_version": "216-external-artifact-population-manifest",
        "retrieval_manifest_sha256": file_sha256(artifacts["retrievalManifest"]),
        "parser_control_sha256": file_sha256(artifacts["parserControl"]),
        "raw_payload_sha256": {payload_id: attempt_by_id[payload_id]["sha256"] for payload_id in sorted(payload_by_id)},
        "public_truth_written_separately": True,
        "protected_partition_not_used_for_method_evaluation": True,
    }
    write_json(artifacts["populationManifest"], population_manifest)
    print(json.dumps({"parser_control": parser_control, "population_manifest": population_manifest}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

