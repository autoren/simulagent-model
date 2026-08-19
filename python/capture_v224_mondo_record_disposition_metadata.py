#!/usr/bin/env python3
from __future__ import annotations

import calendar
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any
from urllib.request import Request, urlopen

from run_v223_archived_semantic_adjudication_metadata_census import dependency_hashes_exact
from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from v224_graphql_queries import NODE_QUERY, RECORD_QUERY, RELEASE_QUERY, forbidden_selected_fields
from v224_mondo_record_disposition_metadata_census import (
    SUBSTANTIVE,
    human_actor,
    latest_event,
    mondo_ids_from_patch,
    parse_time,
    preliminary_disposition,
    selection_key,
)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def gh_graphql(query: str, variables: dict[str, str | None]) -> tuple[dict[str, Any], str, int]:
    cmd = ["gh", "api", "graphql", "-f", f"query={query}"]
    for key, value in variables.items():
        if value is not None:
            cmd.extend(["-f", f"{key}={value}"])
    raw = subprocess.run(cmd, check=True, capture_output=True).stdout
    payload = json.loads(raw)
    if payload.get("errors"):
        raise RuntimeError(payload["errors"])
    return payload, sha_bytes(raw), len(raw)


def gh_json(endpoint: str) -> tuple[Any, str, int]:
    raw = subprocess.run(["gh", "api", endpoint], check=True, capture_output=True).stdout
    return json.loads(raw), sha_bytes(raw), len(raw)


def fetch_bytes(url: str, user_agent: str) -> bytes:
    request = Request(url, headers={"User-Agent": user_agent})
    with urlopen(request, timeout=90) as response:
        return response.read()


def month_slices(start_iso: str, end_iso: str) -> list[tuple[str, str]]:
    start = parse_time(start_iso)
    end = parse_time(end_iso)
    assert start and end
    result = []
    year, month = start.year, start.month
    while (year, month) <= (end.year, end.month):
        last = calendar.monthrange(year, month)[1]
        result.append((f"{year:04d}-{month:02d}-01", f"{year:04d}-{month:02d}-{last:02d}"))
        month = 1 if month == 12 else month + 1
        year = year + 1 if month == 1 else year
    return result


def capture_record_frame(config: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    slices: list[dict[str, Any]] = []
    excluded = set(config["priorExposure"]["excludedIssueNumbers"])
    for first, last in month_slices(config["source"]["creationWindowStart"], config["source"]["creationWindowEnd"]):
        search_query = (
            f'repo:{config["source"]["repository"]} is:issue '
            f'created:{first}..{last} updated:<2026-01-01'
        )
        after: str | None = None
        page_hashes: list[dict[str, Any]] = []
        retrieved = 0
        issue_count: int | None = None
        for page_index in range(config["source"]["maximumPagesPerSlice"]):
            payload, digest, byte_count = gh_graphql(RECORD_QUERY, {"query": search_query, "after": after})
            search = payload["data"]["search"]
            issue_count = search["issueCount"] if issue_count is None else issue_count
            if issue_count > config["source"]["maximumSearchResultsPerSlice"]:
                raise RuntimeError(f"search slice exceeds frozen maximum: {first}")
            nodes = [node for node in search["nodes"] if node and node.get("number") not in excluded]
            records.extend(nodes)
            retrieved += len(search["nodes"])
            page_hashes.append({"page": page_index + 1, "sha256": digest, "byte_count": byte_count})
            if not search["pageInfo"]["hasNextPage"]:
                after = None
                break
            after = search["pageInfo"]["endCursor"]
        if after is not None:
            raise RuntimeError(f"search slice pagination exceeded frozen maximum: {first}")
        if retrieved != issue_count:
            raise RuntimeError(f"search slice count mismatch: {first}: {retrieved} != {issue_count}")
        slices.append({
            "start": first,
            "end": last,
            "query": search_query,
            "issue_count_including_exclusions": issue_count,
            "retained_after_prior_exclusions": sum(
                first <= record["createdAt"][:10] <= last for record in records
            ),
            "page_count": len(page_hashes),
            "page_hashes": page_hashes,
        })
    numbers = [record["number"] for record in records]
    if len(numbers) != len(set(numbers)):
        raise RuntimeError("duplicate issue numbers across frozen search slices")
    return sorted(records, key=lambda row: row["number"]), slices


def capture_releases(config: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    releases: list[dict[str, Any]] = []
    pages: list[dict[str, Any]] = []
    after: str | None = None
    while True:
        payload, digest, byte_count = gh_graphql(RELEASE_QUERY, {"after": after})
        connection = payload["data"]["repository"]["releases"]
        pages.append({"sha256": digest, "byte_count": byte_count})
        releases.extend(connection["nodes"])
        if not connection["pageInfo"]["hasNextPage"]:
            break
        after = connection["pageInfo"]["endCursor"]
    start = parse_time(config["source"]["releaseWindowStart"])
    end = parse_time(config["source"]["releaseWindowEnd"])
    assert start and end
    kept = []
    for release in releases:
        published = parse_time(release.get("publishedAt"))
        if published and start <= published <= end and not release["isDraft"] and not release["isPrerelease"]:
            if release["releaseAssets"]["pageInfo"]["hasNextPage"]:
                continue
            kept.append(release)
    return sorted(kept, key=lambda row: (row["publishedAt"], row["tagName"])), pages


def release_id_index(releases: list[dict[str, Any]], config: dict[str, Any]) -> list[dict[str, Any]]:
    result = []
    required = config["source"]["requiredReleaseAssetName"]
    for release in releases:
        matches = [asset for asset in release["releaseAssets"]["nodes"] if asset["name"] == required]
        if len(matches) != 1:
            continue
        asset = matches[0]
        raw = fetch_bytes(asset["downloadUrl"], "simulagent-v224-metadata-census/1.0")
        lines = raw.decode("utf-8").splitlines()
        if not lines or lines[0].split("\t")[0].strip().casefold() != "mondo_id":
            continue
        ids = sorted(
            {
                line.split("\t", 1)[0].strip()
                for line in lines[1:]
                if line.split("\t", 1)[0].strip().startswith("MONDO:")
            }
        )
        result.append({
            "tag_name": release["tagName"],
            "published_at": release["publishedAt"],
            "asset_name": required,
            "download_url": asset["downloadUrl"],
            "raw_sha256": sha_bytes(raw),
            "raw_byte_count": len(raw),
            "mondo_ids": ids,
        })
    return result


def pull_added_ids(pull: dict[str, Any], config: dict[str, Any]) -> tuple[list[str], dict[str, Any]]:
    endpoint = f'repos/{config["source"]["repository"]}/pulls/{pull["number"]}/files?per_page=100'
    files, digest, byte_count = gh_json(endpoint)
    ids = sorted({mondo_id for file in files for mondo_id in mondo_ids_from_patch(file.get("patch", ""))})
    sanitized = [
        {
            "path": file.get("filename"),
            "status": file.get("status"),
            "additions": file.get("additions"),
            "deletions": file.get("deletions"),
            "added_mondo_ids": mondo_ids_from_patch(file.get("patch", "")),
        }
        for file in files
    ]
    return ids, {
        "endpoint": endpoint,
        "raw_response_sha256": digest,
        "raw_response_byte_count": byte_count,
        "sanitized_files": sanitized,
        "raw_patch_persisted": False,
    }


def first_release_for_id(mondo_id: str, merged_at: str, index: list[dict[str, Any]]) -> dict[str, Any] | None:
    merged = parse_time(merged_at)
    assert merged
    eligible = [row for row in index if parse_time(row["published_at"]) >= merged and mondo_id in row["mondo_ids"]]
    return min(eligible, key=lambda row: row["published_at"], default=None)


def deep_accept(
    record: dict[str, Any], index: list[dict[str, Any]], config: dict[str, Any]
) -> dict[str, Any]:
    requester = (record.get("author") or {}).get("login")
    cutoff = parse_time(config["source"]["eventCutoff"])
    assert cutoff
    candidates = []
    for pull in record["closedByPullRequestsReferences"]["nodes"]:
        merged_at = parse_time(pull.get("mergedAt"))
        if not pull.get("merged") or not merged_at or merged_at > cutoff:
            continue
        if not human_actor(pull.get("mergedBy"), requester, config):
            continue
        if not any(file["path"] == "src/ontology/mondo-edit.obo" for file in pull["files"]["nodes"]):
            continue
        ids, capture = pull_added_ids(pull, config)
        if len(ids) != 1:
            continue
        release = first_release_for_id(ids[0], pull["mergedAt"], index)
        if release:
            candidates.append((pull, ids[0], release, capture))
    if len(candidates) != 1:
        return {
            "retained": False,
            "human_independence": False,
            "provenance_complete": False,
            "exclusion_reasons": ["ACCEPTED_CHAIN_NOT_EXACTLY_ONE"],
        }
    pull, mondo_id, release, capture = candidates[0]
    return {
        "retained": True,
        "human_independence": True,
        "provenance_complete": True,
        "exclusion_reasons": [],
        "pull_number": pull["number"],
        "merge_commit_oid": pull["mergeCommit"]["oid"] if pull.get("mergeCommit") else None,
        "merged_at": pull["mergedAt"],
        "merged_by": pull["mergedBy"]["login"],
        "mondo_id": mondo_id,
        "first_release_tag": release["tag_name"],
        "first_release_published_at": release["published_at"],
        "pull_file_capture": capture,
    }


def main() -> None:
    lock_path = PROJECT_ROOT / "configs/v224-mondo-record-disposition-metadata-census-lock.json"
    lock = json.loads(lock_path.read_text())
    if not dependency_hashes_exact(lock):
        raise RuntimeError("V224 design lock or dependency hash mismatch")
    if not lock["authorization"]["run_one_metadata_only_record_disposition_census"]:
        raise RuntimeError("V224 metadata census is not authorized")
    config = lock["config_payload"]
    artifacts = {key: PROJECT_ROOT / value for key, value in config["artifacts"].items()}
    if any(path.exists() for path in artifacts.values()):
        raise RuntimeError("V224 formal artifacts already exist")

    scope_raw = fetch_bytes(config["source"]["pinnedScopePolicyUrl"], "simulagent-v224-metadata-census/1.0")
    artifacts["scopePolicySnapshot"].parent.mkdir(parents=True, exist_ok=True)
    artifacts["scopePolicySnapshot"].write_bytes(scope_raw)
    scope_text = scope_raw.decode("utf-8").casefold()
    scope_supported = "non-human" in scope_text and ("out of scope" in scope_text or "outside scope" in scope_text)

    records, slices = capture_record_frame(config)
    rows_by_outcome: dict[str, list[int]] = {name: [] for name in (*SUBSTANTIVE, "AMBIGUOUS_EXCLUDE")}
    reasons = Counter()
    for record in records:
        outcome, record_reasons = preliminary_disposition(record, config)
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
        releases, release_pages = capture_releases(config)
        release_index = release_id_index(releases, config)
        release_metadata = {"skipped": False, "releases": releases, "page_hashes": release_pages}
        for disposition in SUBSTANTIVE:
            for number in selected[disposition]:
                record = by_number[number]
                base = {"issue_number": number, "disposition": disposition}
                if disposition == "ACCEPTED_NEW":
                    base.update(deep_accept(record, release_index, config))
                elif disposition == "EXISTING_OR_DUPLICATE":
                    cutoff = parse_time(config["source"]["eventCutoff"])
                    duplicate_event = latest_event(record, "MarkedAsDuplicateEvent", None, cutoff)
                    canonical_ref = (duplicate_event or {}).get("canonical") or {}
                    canonical_number = canonical_ref.get("number")
                    if canonical_number in config["priorExposure"]["excludedIssueNumbers"]:
                        base.update({"retained": False, "human_independence": False, "provenance_complete": False,
                                     "exclusion_reasons": ["CANONICAL_PRIOR_EXPOSURE"]})
                    else:
                        payload, digest, byte_count = gh_graphql(NODE_QUERY, {"id": canonical_ref.get("id")})
                        canonical = payload["data"]["node"]
                        node_queries.append({"node_id": canonical_ref.get("id"), "sha256": digest, "byte_count": byte_count})
                        canonical_outcome, _ = preliminary_disposition(canonical, config)
                        accepted = deep_accept(canonical, release_index, config) if canonical_outcome == "ACCEPTED_NEW" else {
                            "retained": False, "human_independence": False, "provenance_complete": False,
                            "exclusion_reasons": ["CANONICAL_NOT_ACCEPTED_NEW"]
                        }
                        release_before = accepted.get("first_release_published_at") and (
                            parse_time(accepted["first_release_published_at"]) < parse_time(record["createdAt"])
                        )
                        if accepted.get("retained") and release_before:
                            base.update(accepted)
                            base["canonical_issue_number"] = canonical_number
                        else:
                            base.update({"retained": False, "human_independence": False, "provenance_complete": False,
                                         "exclusion_reasons": ["CANONICAL_TERM_NOT_RELEASED_BEFORE_DUPLICATE_REQUEST"]})
                elif disposition == "INSUFFICIENT_OR_CLARIFY":
                    base.update({"retained": True, "human_independence": True, "provenance_complete": True,
                                 "exclusion_reasons": []})
                else:
                    base.update({"retained": scope_supported, "human_independence": scope_supported,
                                 "provenance_complete": scope_supported,
                                 "exclusion_reasons": [] if scope_supported else ["PINNED_SCOPE_POLICY_MAPPING_FAILED"]})
                deep_rows.append(base)

    query_manifest = {
        "schema_version": "224-mondo-record-disposition-query-manifest",
        "search_slices": slices,
        "node_queries": node_queries,
        "release_query_pages": release_pages,
        "scope_policy_sha256": file_sha256(artifacts["scopePolicySnapshot"]),
        "scope_policy_byte_count": artifacts["scopePolicySnapshot"].stat().st_size,
        "scope_policy_supports_non_human_animal_out_of_scope": scope_supported,
        "forbidden_selected_field_count": sum(
            len(forbidden_selected_fields(query)) for query in (RECORD_QUERY, NODE_QUERY, RELEASE_QUERY)
        ),
        "task_language_persistence_count": 0,
        "raw_pull_patch_persistence_count": 0,
        "raw_release_tsv_persistence_count": 0,
        "record_metadata_count": len(records),
    }
    write_json(artifacts["queryManifest"], query_manifest)
    write_jsonl(artifacts["recordMetadata"], records)
    write_json(artifacts["preliminaryCensus"], preliminary)
    write_json(artifacts["releaseMetadata"], release_metadata)
    write_json(artifacts["releaseIdIndex"], release_index)
    write_jsonl(artifacts["deepAudit"], deep_rows)
    print(json.dumps({
        "record_count": len(records),
        "preliminary_counts": preliminary["counts"],
        "preliminary_gate_passed": preliminary_pass,
        "deep_audit_row_count": len(deep_rows),
        "task_language_persistence_count": 0,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

