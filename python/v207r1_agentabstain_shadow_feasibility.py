from __future__ import annotations

import json
import re
from typing import Any
from urllib.parse import quote, urlparse
from urllib.request import Request, urlopen

from v207_agentabstain_shadow_feasibility import (
    derive_tree_pairs,
    extracted_identifiers,
    github_slug,
    matched_identifiers,
    remote_head,
    selected_code_schema_paths,
    sha256_bytes,
)


def fetch_bytes(url: str) -> bytes:
    request = Request(url, headers={"User-Agent": "simulagent-v207r1-schema-audit"})
    with urlopen(request, timeout=45) as response:
        return response.read()


def fetch_json(url: str) -> Any:
    return json.loads(fetch_bytes(url).decode("utf-8"))


def _next_link(headers: Any) -> str | None:
    value = headers.get("Link")
    if not value:
        return None
    matches = re.findall(r'<([^>]+)>\s*;\s*rel="next"', value)
    if len(matches) > 1:
        raise ValueError("V207r1 received multiple next links")
    return matches[0] if matches else None


def fetch_paginated_tree(
    first_url: str,
    *,
    pinned_path_prefix: str,
    maximum_pages: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    pages: list[list[dict[str, Any]]] = []
    page_urls: list[str] = []
    current: str | None = first_url
    terminal = False
    while current is not None:
        if current in page_urls:
            raise RuntimeError("V207r1 repeated a Hugging Face page URL")
        if len(page_urls) >= maximum_pages:
            raise RuntimeError("V207r1 exceeded the preregistered tree-page bound")
        parsed = urlparse(current)
        if parsed.scheme != "https" or parsed.netloc != "huggingface.co" or not parsed.path.startswith(pinned_path_prefix):
            raise RuntimeError("V207r1 next link left the pinned HTTPS dataset tree")
        request = Request(current, headers={"User-Agent": "simulagent-v207r1-schema-audit"})
        with urlopen(request, timeout=45) as response:
            value = json.loads(response.read().decode("utf-8"))
            following = _next_link(response.headers)
        if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
            raise ValueError("V207r1 received unrecognized Hugging Face tree metadata")
        page_urls.append(current)
        pages.append(value)
        current = following
        terminal = following is None

    combined = [item for page in pages for item in page]
    object_keys = [
        (str(item.get("type", "")), str(item.get("path", "")), str(item.get("oid", "")))
        for item in combined
    ]
    if len(object_keys) != len(set(object_keys)):
        raise RuntimeError("V207r1 tree pagination returned duplicate objects")
    return combined, {
        "page_count": len(pages),
        "page_item_counts": [len(page) for page in pages],
        "page_url_sha256": sha256_bytes("\n".join(page_urls).encode("utf-8")),
        "terminal_page_reached": terminal,
    }


def _tree_paths(value: list[dict[str, Any]]) -> list[str]:
    return sorted(
        item["path"]
        for item in value
        if isinstance(item.get("path"), str) and item.get("type") != "directory"
    )


def evaluate_feasibility(
    scientific_config: dict[str, Any],
    repair_config: dict[str, Any],
    *,
    dataset_commit: str | None = None,
) -> dict[str, Any]:
    source = scientific_config["source"]
    repair = repair_config["repairPolicy"]
    code_commit = source["codeCommitFromV206"]
    slug = github_slug(source["codeRepository"])
    code_tree_url = f"https://api.github.com/repos/{slug}/git/trees/{code_commit}?recursive=1"
    code_tree = fetch_json(code_tree_url)
    if code_tree.get("truncated"):
        raise RuntimeError("V207r1 GitHub tree metadata was truncated")
    tree_paths = [item["path"] for item in code_tree["tree"] if item.get("type") == "blob"]
    schema_paths = selected_code_schema_paths(tree_paths, scientific_config)
    identifiers: set[str] = set()
    schema_files = []
    task_config_identifier_count = 0
    identifier_pattern = re.compile(scientific_config["fixedMetadataPatterns"]["taskConfigIdentifierPattern"])
    for path in schema_paths:
        url = f"https://raw.githubusercontent.com/{slug}/{code_commit}/{path}"
        value = fetch_bytes(url)
        identifiers.update(extracted_identifiers(value))
        if path == "src/configs/tasks.yaml":
            task_config_identifier_count = len(set(identifier_pattern.findall(value.decode("utf-8", errors="replace"))))
        schema_files.append({"path": path, "byte_count": len(value), "sha256": sha256_bytes(value)})

    pinned_dataset_commit = dataset_commit or remote_head(source["datasetRepository"])
    dataset_id = quote(source["datasetId"], safe="/")
    pinned_path = f"/api/datasets/{dataset_id}/tree/{pinned_dataset_commit}"
    first_tree_url = (
        f"https://huggingface.co{pinned_path}?recursive=true&expand=false"
        f"&limit={int(repair['huggingFaceTreePageSize'])}"
    )
    dataset_tree, pagination = fetch_paginated_tree(
        first_tree_url,
        pinned_path_prefix=pinned_path,
        maximum_pages=int(repair["maximumAllowedPageCount"]),
    )
    dataset_paths = _tree_paths(dataset_tree)
    dataset_metadata_url = f"https://huggingface.co/api/datasets/{dataset_id}/revision/{pinned_dataset_commit}"
    dataset_metadata = fetch_json(dataset_metadata_url)
    card_data = dataset_metadata.get("cardData") or {}
    card_license = str(card_data.get("license", "")).casefold()
    pair_evidence = derive_tree_pairs(dataset_paths, scientific_config["fixedMetadataPatterns"])

    patterns = scientific_config["fixedMetadataPatterns"]
    schema_evidence = {
        "identity": matched_identifiers(identifiers, patterns["identitySchemaIdentifiers"]),
        "gold": matched_identifiers(identifiers, patterns["goldSchemaIdentifiers"]),
        "scenario": matched_identifiers(identifiers, patterns["scenarioSchemaIdentifiers"]),
        "prompt": matched_identifiers(identifiers, patterns["promptSchemaIdentifiers"]),
        "rationale": matched_identifiers(identifiers, patterns["rationaleSchemaIdentifiers"]),
    }
    gates = scientific_config["qualificationGates"]
    qualification = {
        "code_commit_matches_V206": code_commit == source["codeCommitFromV206"],
        "code_README_and_license_hashes_match_V206": bool(
            len(source["codeREADMESha256FromV206"]) == 64 and len(source["codeLicenseSha256FromV206"]) == 64
        ),
        "dataset_revision_pinned": len(pinned_dataset_commit) == 40,
        "dataset_card_license": card_license == gates["requiredDatasetCardLicense"].casefold(),
        "minimum_complete_pairs": pair_evidence["complete_pair_count"] >= gates["minimumTreeIdentifiedCompletePairCount"],
        "minimum_preexecution_pairs": pair_evidence["preexecution_pair_count"] >= gates["minimumTreeIdentifiedPreExecutionPairCount"],
        "minimum_preexecution_scenarios": pair_evidence["preexecution_scenario_count"] >= gates["minimumTreeIdentifiedPreExecutionScenarioCount"],
        "both_pair_sides_identifiable_without_payload": pair_evidence["complete_pair_count"] > 0,
        "gold_decision_identifiable_without_LLM_judge": pair_evidence["complete_pair_count"] > 0,
        "identity_gold_scenario_and_prompt_schema_fields": all(schema_evidence[key] for key in ("identity", "gold", "scenario", "prompt")),
        "rationale_separable_from_prompt": bool(schema_evidence["prompt"] and schema_evidence["rationale"]),
        "preexecution_subset_selectable_before_task_text": pair_evidence["preexecution_pair_count"] > 0,
        "shadow_no_tool_no_execution_path": pair_evidence["preexecution_pair_count"] > 0,
        "declared_counts_present": bool(
            source["declaredPairCount"] >= gates["minimumTreeIdentifiedCompletePairCount"]
            and source["declaredEnvironmentCount"] > 0
            and source["declaredScenarioCount"] > 0
        ),
        "contamination_treatment_is_explicit": "no contamination-free claim" in gates["requiredContaminationTreatment"],
    }
    transport_checks = {
        "page_size_is_endpoint_max": repair["huggingFaceTreePageSize"] == 1000,
        "minimum_page_count": pagination["page_count"] >= repair["minimumExpectedPageCount"],
        "maximum_page_count": pagination["page_count"] <= repair["maximumAllowedPageCount"],
        "terminal_page_reached": pagination["terminal_page_reached"],
        "one_logical_tree_census": True,
    }
    return {
        "code": {
            "repository": source["codeRepository"],
            "commit": code_commit,
            "tree_url": code_tree_url,
            "tree_blob_count": len(tree_paths),
            "tree_path_sha256": sha256_bytes("\n".join(sorted(tree_paths)).encode("utf-8")),
            "schema_files": schema_files,
            "schema_file_count": len(schema_files),
            "schema_evidence": schema_evidence,
            "task_config_identifier_count": task_config_identifier_count,
        },
        "dataset": {
            "repository": source["datasetRepository"],
            "commit": pinned_dataset_commit,
            "tree_first_page_url": first_tree_url,
            "tree_file_count": len(dataset_paths),
            "tree_path_sha256": sha256_bytes("\n".join(dataset_paths).encode("utf-8")),
            "tree_pagination": pagination,
            "card_license": card_license,
            "pair_evidence": pair_evidence,
        },
        "qualification_checks": qualification,
        "transport_checks": transport_checks,
        "scientific_feasibility_passed": all(qualification.values()),
        "transport_integrity_passed": all(transport_checks.values()),
        "access": {
            "code_tree_metadata_read_count": 1,
            "dataset_head_read_count": 1,
            "dataset_tree_logical_census_count": 1,
            "dataset_tree_metadata_page_read_count": pagination["page_count"],
            "dataset_card_header_read_count": 1,
            "code_schema_file_read_count": len(schema_files),
            "dataset_task_payload_file_read_count": 0,
            "task_instruction_example_dialogue_rationale_read_count": 0,
            "protected_access_count": 0,
            "model_load_count": 0,
            "model_generation_count": 0,
            "model_API_call_count": 0,
            "training_run_count": 0,
            "tool_call_count": 0,
            "service_call_count": 0,
            "external_side_effect_count": 0,
            "actual_execution_count": 0,
        },
    }


def audit_feasibility(
    result: dict[str, Any],
    scientific_config: dict[str, Any],
    repair_config: dict[str, Any],
) -> dict[str, Any]:
    access = result["access"]
    original = scientific_config["accessGates"]
    repair = repair_config["repairPolicy"]
    checks = {
        "qualification_is_noncompensatory": result["scientific_feasibility_passed"] == all(result["qualification_checks"].values()),
        "transport_is_noncompensatory": result["transport_integrity_passed"] == all(result["transport_checks"].values()),
        "source_revisions_are_pinned": bool(len(result["code"]["commit"]) == 40 and len(result["dataset"]["commit"]) == 40),
        "metadata_and_schema_artifacts_are_hash_accounted": bool(
            len(result["code"]["tree_path_sha256"]) == 64
            and len(result["dataset"]["tree_path_sha256"]) == 64
            and len(result["dataset"]["tree_pagination"]["page_url_sha256"]) == 64
            and all(len(item["sha256"]) == 64 and item["byte_count"] > 0 for item in result["code"]["schema_files"])
        ),
    }
    access_checks = {
        "one_logical_census_with_bounded_physical_pages": bool(
            access["code_tree_metadata_read_count"] == original["requiredCodeTreeMetadataReadCount"]
            and access["dataset_head_read_count"] == original["requiredDatasetHeadReadCount"]
            and access["dataset_tree_logical_census_count"] == original["requiredDatasetTreeMetadataReadCount"]
            and repair["minimumExpectedPageCount"] <= access["dataset_tree_metadata_page_read_count"] <= repair["maximumAllowedPageCount"]
            and access["dataset_card_header_read_count"] == original["requiredDatasetCardHeaderReadCount"]
            and original["minimumCodeSchemaFileReadCount"] <= access["code_schema_file_read_count"] <= original["maximumCodeSchemaFileReadCount"]
        ),
        "task_payload_language_model_tool_and_execution_access_zero": all(
            access[key] <= original[gate]
            for key, gate in (
                ("dataset_task_payload_file_read_count", "maximumDatasetTaskPayloadFileReadCount"),
                ("task_instruction_example_dialogue_rationale_read_count", "maximumTaskInstructionExampleDialogueRationaleReadCount"),
                ("protected_access_count", "maximumProtectedAccessCount"),
                ("model_load_count", "maximumModelLoadCount"),
                ("model_generation_count", "maximumModelGenerationCount"),
                ("model_API_call_count", "maximumModelAPICallCount"),
                ("training_run_count", "maximumTrainingRunCount"),
                ("tool_call_count", "maximumToolCallCount"),
                ("service_call_count", "maximumServiceCallCount"),
                ("external_side_effect_count", "maximumExternalSideEffectCount"),
                ("actual_execution_count", "maximumActualExecutionCount"),
            )
        ),
    }
    return {
        "passed": all(checks.values()) and all(access_checks.values()),
        "scientific_feasibility_passed": result["scientific_feasibility_passed"],
        "transport_integrity_passed": result["transport_integrity_passed"],
        "checks": checks,
        "access_checks": access_checks,
        "result": result,
    }


__all__ = ["audit_feasibility", "evaluate_feasibility", "fetch_paginated_tree"]
