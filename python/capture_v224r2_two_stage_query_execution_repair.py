#!/usr/bin/env python3
from __future__ import annotations

from collections import Counter
from http.client import RemoteDisconnected
from urllib.error import HTTPError, URLError
import json
import subprocess
import time
from typing import Any
from urllib.request import Request, urlopen

import capture_v224_mondo_record_disposition_metadata as base
from run_v223_archived_semantic_adjudication_metadata_census import dependency_hashes_exact
from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from v224_graphql_queries import forbidden_selected_fields
from v224_mondo_record_disposition_metadata_census import SUBSTANTIVE, parse_time, selection_key
from v224r1_graphql_transport_repair import graphql_request_payload
from v224r2_graphql_queries import DEEP_NODE_QUERY, RELEASE_QUERY, THIN_RECORD_QUERY
from v224r2_two_stage_query_execution_repair import preliminary_disposition_thin


def graphql_with_retry(query: str, variables: dict[str, str | None]) -> tuple[dict[str, Any], str, int]:
    token = subprocess.run(["gh", "auth", "token"], check=True, capture_output=True, text=True).stdout.strip()
    raw_request = graphql_request_payload(query, variables)
    for attempt in range(3):
        request = Request(
            "https://api.github.com/graphql",
            data=raw_request,
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
                "User-Agent": "simulagent-v224r2-metadata-census/1.0",
            },
            method="POST",
        )
        try:
            with urlopen(request, timeout=90) as response:
                raw = response.read()
            payload = json.loads(raw)
            if payload.get("errors"):
                raise RuntimeError(payload["errors"])
            return payload, base.sha_bytes(raw), len(raw)
        except HTTPError as exc:
            if exc.code not in {502, 503, 504} or attempt == 2:
                raise
        except (URLError, RemoteDisconnected):
            if attempt == 2:
                raise
        time.sleep(0.5 * (attempt + 1))
    raise AssertionError("unreachable")


def deep_node(node_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
    payload, digest, byte_count = graphql_with_retry(DEEP_NODE_QUERY, {"id": node_id})
    return payload["data"]["node"], {"node_id": node_id, "sha256": digest, "byte_count": byte_count}


def main() -> None:
    repair_lock_path = PROJECT_ROOT / "configs/v224r2-two-stage-query-execution-repair-lock.json"
    repair_lock = json.loads(repair_lock_path.read_text())
    if not dependency_hashes_exact(repair_lock):
        raise RuntimeError("V224r2 repair lock or dependency hash mismatch")
    if not repair_lock["authorization"]["run_one_completed_two_stage_V224_metadata_census"]:
        raise RuntimeError("V224r2 census is not authorized")
    parent = json.loads((PROJECT_ROOT / repair_lock["parent_V224_design_lock"]).read_text())
    config = parent["config_payload"]
    artifacts = {key: PROJECT_ROOT / value for key, value in config["artifacts"].items()}
    failure = repair_lock["config_payload"]["failureBoundary"]
    if file_sha256(artifacts["scopePolicySnapshot"]) != failure["scopePolicySnapshotSha256"]:
        raise RuntimeError("scope policy changed")
    if any(path.exists() for key, path in artifacts.items() if key != "scopePolicySnapshot"):
        raise RuntimeError("unexpected V224 artifact exists before V224r2")
    base.gh_graphql = graphql_with_retry
    base.RECORD_QUERY = THIN_RECORD_QUERY
    base.RELEASE_QUERY = RELEASE_QUERY
    scope_text = artifacts["scopePolicySnapshot"].read_text().casefold()
    scope_supported = "non-human" in scope_text and ("out of scope" in scope_text or "outside scope" in scope_text)

    records, slices = base.capture_record_frame(config)
    rows_by_outcome: dict[str, list[int]] = {name: [] for name in (*SUBSTANTIVE, "AMBIGUOUS_EXCLUDE")}
    reasons = Counter()
    for record in records:
        outcome, record_reasons = preliminary_disposition_thin(record, config)
        rows_by_outcome[outcome].append(record["number"])
        reasons.update(record_reasons)
    selected = {
        name: sorted(rows_by_outcome[name], key=lambda number: selection_key(number, config))[
            : config["samplingContract"]["maximumDeepAuditRecordsPerStratum"]
        ]
        for name in SUBSTANTIVE
    }
    preliminary_pass = all(
        len(rows_by_outcome[name]) >= config["samplingContract"]["minimumPreliminaryRecordsPerStratum"]
        for name in SUBSTANTIVE
    )
    preliminary = {
        "counts": {name: len(rows_by_outcome[name]) for name in (*SUBSTANTIVE, "AMBIGUOUS_EXCLUDE")},
        "exclusion_reason_counts": dict(sorted(reasons.items())),
        "selected_issue_numbers": selected,
        "preliminary_gate_passed": preliminary_pass,
    }

    release_metadata: dict[str, Any] = {"skipped": True, "reason": "PRELIMINARY_GATE_FAILED"}
    release_index: list[dict[str, Any]] = []
    release_pages: list[dict[str, Any]] = []
    deep_rows: list[dict[str, Any]] = []
    node_queries: list[dict[str, Any]] = []
    by_number = {record["number"]: record for record in records}
    if preliminary_pass:
        releases, release_pages = base.capture_releases(config)
        release_index = base.release_id_index(releases, config)
        release_metadata = {"skipped": False, "releases": releases, "page_hashes": release_pages}
        for disposition in SUBSTANTIVE:
            for number in selected[disposition]:
                thin_record = by_number[number]
                row: dict[str, Any] = {"issue_number": number, "disposition": disposition}
                if disposition == "ACCEPTED_NEW":
                    full, query_meta = deep_node(thin_record["id"])
                    node_queries.append(query_meta)
                    row.update(base.deep_accept(full, release_index, config))
                elif disposition == "EXISTING_OR_DUPLICATE":
                    cutoff = parse_time(config["source"]["eventCutoff"])
                    duplicate = base.latest_event(thin_record, "MarkedAsDuplicateEvent", None, cutoff)
                    canonical_ref = (duplicate or {}).get("canonical") or {}
                    canonical_number = canonical_ref.get("number")
                    if canonical_number in config["priorExposure"]["excludedIssueNumbers"]:
                        row.update({"retained": False, "human_independence": False, "provenance_complete": False,
                                    "exclusion_reasons": ["CANONICAL_PRIOR_EXPOSURE"]})
                    else:
                        canonical, query_meta = deep_node(canonical_ref["id"])
                        node_queries.append(query_meta)
                        canonical_outcome, _ = base.preliminary_disposition(canonical, config)
                        accepted = base.deep_accept(canonical, release_index, config) if canonical_outcome == "ACCEPTED_NEW" else {
                            "retained": False, "human_independence": False, "provenance_complete": False,
                            "exclusion_reasons": ["CANONICAL_NOT_ACCEPTED_NEW"]
                        }
                        release_before = accepted.get("first_release_published_at") and (
                            parse_time(accepted["first_release_published_at"]) < parse_time(thin_record["createdAt"])
                        )
                        if accepted.get("retained") and release_before:
                            row.update(accepted)
                            row["canonical_issue_number"] = canonical_number
                        else:
                            row.update({"retained": False, "human_independence": False, "provenance_complete": False,
                                        "exclusion_reasons": ["CANONICAL_TERM_NOT_RELEASED_BEFORE_DUPLICATE_REQUEST"]})
                elif disposition == "INSUFFICIENT_OR_CLARIFY":
                    row.update({"retained": True, "human_independence": True, "provenance_complete": True,
                                "exclusion_reasons": []})
                else:
                    row.update({"retained": scope_supported, "human_independence": scope_supported,
                                "provenance_complete": scope_supported,
                                "exclusion_reasons": [] if scope_supported else ["PINNED_SCOPE_POLICY_MAPPING_FAILED"]})
                deep_rows.append(row)

    query_manifest = {
        "schema_version": "224r2-mondo-record-disposition-query-manifest",
        "execution_repair": "THIN_ENUMERATION_THEN_HASH_SELECTED_DEEP_QUERY",
        "prior_failed_capture_attempt_count": 2,
        "prior_task_language_persistence_or_research_exposure_count": 0,
        "search_slices": slices,
        "node_queries": node_queries,
        "release_query_pages": release_pages,
        "scope_policy_sha256": file_sha256(artifacts["scopePolicySnapshot"]),
        "scope_policy_byte_count": artifacts["scopePolicySnapshot"].stat().st_size,
        "scope_policy_supports_non_human_animal_out_of_scope": scope_supported,
        "forbidden_selected_field_count": sum(
            len(forbidden_selected_fields(query)) for query in (THIN_RECORD_QUERY, DEEP_NODE_QUERY, RELEASE_QUERY)
        ),
        "task_language_persistence_count": 0,
        "raw_pull_patch_persistence_count": 0,
        "raw_release_tsv_persistence_count": 0,
        "record_metadata_count": len(records),
    }
    base.write_json(artifacts["queryManifest"], query_manifest)
    base.write_jsonl(artifacts["recordMetadata"], records)
    base.write_json(artifacts["preliminaryCensus"], preliminary)
    base.write_json(artifacts["releaseMetadata"], release_metadata)
    base.write_json(artifacts["releaseIdIndex"], release_index)
    base.write_jsonl(artifacts["deepAudit"], deep_rows)
    print(json.dumps({
        "record_count": len(records), "preliminary_counts": preliminary["counts"],
        "preliminary_gate_passed": preliminary_pass, "deep_audit_row_count": len(deep_rows),
        "search_page_count": sum(row["page_count"] for row in slices),
        "task_language_persistence_count": 0,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

