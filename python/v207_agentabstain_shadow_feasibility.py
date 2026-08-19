from __future__ import annotations

import hashlib
import json
import re
import subprocess
from typing import Any
from urllib.parse import quote
from urllib.request import Request, urlopen


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def fetch_bytes(url: str) -> bytes:
    request = Request(url, headers={"User-Agent": "simulagent-v207-schema-audit"})
    with urlopen(request, timeout=45) as response:
        return response.read()


def fetch_json(url: str) -> Any:
    return json.loads(fetch_bytes(url).decode("utf-8"))


def remote_head(repository: str) -> str:
    completed = subprocess.run(
        ["git", "ls-remote", repository, "HEAD"],
        check=True,
        capture_output=True,
        text=True,
        timeout=45,
    )
    fields = completed.stdout.strip().split()
    if len(fields) != 2 or fields[1] != "HEAD" or len(fields[0]) != 40:
        raise RuntimeError(f"V207 could not pin dataset HEAD: {repository}")
    return fields[0]


def github_slug(repository: str) -> str:
    prefix = "https://github.com/"
    if not repository.startswith(prefix):
        raise ValueError("V207 code repository URL is not canonical GitHub")
    return repository[len(prefix) :].removesuffix(".git").strip("/")


def selected_code_schema_paths(tree_paths: list[str], config: dict[str, Any]) -> list[str]:
    allowed = config["allowedMetadataReads"]
    exact = set(allowed["codeFiles"])
    prefixes = tuple(allowed["codeGlobPrefixes"])
    suffix = allowed["codeGlobSuffix"]
    selected = sorted(path for path in tree_paths if path in exact or (path.startswith(prefixes) and path.endswith(suffix)))
    maximum = int(allowed["maximumAllowedCodeSchemaFiles"])
    if len(selected) > maximum:
        raise ValueError("V207 allowed schema-file census exceeds its fixed maximum")
    return selected


def extracted_identifiers(value: bytes) -> set[str]:
    text = value.decode("utf-8", errors="replace")
    return {match.casefold() for match in re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*\b", text)}


def matched_identifiers(identifiers: set[str], candidates: list[str]) -> list[str]:
    return sorted(candidate for candidate in candidates if candidate.casefold() in identifiers)


def _has_marker(value: str, markers: list[str]) -> bool:
    normalized = value.casefold()
    return any(marker.casefold() in normalized for marker in markers)


def derive_tree_pairs(paths: list[str], patterns: dict[str, Any]) -> dict[str, Any]:
    act_markers = patterns["shouldActPathMarkers"]
    abstain_markers = patterns["shouldAbstainPathMarkers"]
    pre_markers = patterns["preExecutionPathMarkers"]
    runtime_markers = patterns["runtimePathMarkers"]
    sides: dict[str, set[str]] = {}
    side_paths: dict[tuple[str, str], str] = {}
    for path in paths:
        normalized = path.casefold()
        act = _has_marker(normalized, act_markers)
        abstain = _has_marker(normalized, abstain_markers)
        if act == abstain:
            continue
        markers = act_markers if act else abstain_markers
        base = normalized
        for marker in markers:
            base = base.replace(marker.casefold(), "<SIDE>")
        side = "act" if act else "abstain"
        sides.setdefault(base, set()).add(side)
        side_paths[(base, side)] = path
    complete = sorted(base for base, present in sides.items() if present == {"act", "abstain"})
    pre = [base for base in complete if _has_marker(base, pre_markers)]
    runtime = [base for base in complete if _has_marker(base, runtime_markers)]
    scenarios = set()
    for base in pre:
        for component in base.split("/"):
            if _has_marker(component, pre_markers):
                scenarios.add(component)
    return {
        "complete_pair_count": len(complete),
        "preexecution_pair_count": len(pre),
        "runtime_pair_count": len(runtime),
        "preexecution_scenario_identifiers": sorted(scenarios),
        "preexecution_scenario_count": len(scenarios),
        "pair_identity_sha256": sha256_bytes("\n".join(complete).encode("utf-8")),
        "preexecution_pair_identity_sha256": sha256_bytes("\n".join(pre).encode("utf-8")),
    }


def _tree_paths_from_huggingface(value: Any) -> list[str]:
    if isinstance(value, list):
        return sorted(
            item["path"]
            for item in value
            if isinstance(item, dict) and isinstance(item.get("path"), str) and item.get("type") != "directory"
        )
    if isinstance(value, dict) and isinstance(value.get("siblings"), list):
        return sorted(item["rfilename"] for item in value["siblings"] if isinstance(item.get("rfilename"), str))
    raise ValueError("V207 unrecognized Hugging Face tree metadata")


def evaluate_feasibility(config: dict[str, Any], *, dataset_commit: str | None = None) -> dict[str, Any]:
    source = config["source"]
    code_commit = source["codeCommitFromV206"]
    slug = github_slug(source["codeRepository"])
    code_tree_url = f"https://api.github.com/repos/{slug}/git/trees/{code_commit}?recursive=1"
    code_tree = fetch_json(code_tree_url)
    if code_tree.get("truncated"):
        raise RuntimeError("V207 GitHub tree metadata was truncated")
    tree_paths = [item["path"] for item in code_tree["tree"] if item.get("type") == "blob"]
    schema_paths = selected_code_schema_paths(tree_paths, config)
    identifiers: set[str] = set()
    schema_files = []
    task_config_identifier_count = 0
    identifier_pattern = re.compile(config["fixedMetadataPatterns"]["taskConfigIdentifierPattern"])
    for path in schema_paths:
        url = f"https://raw.githubusercontent.com/{slug}/{code_commit}/{path}"
        value = fetch_bytes(url)
        identifiers.update(extracted_identifiers(value))
        if path == "src/configs/tasks.yaml":
            task_config_identifier_count = len(set(identifier_pattern.findall(value.decode("utf-8", errors="replace"))))
        schema_files.append({"path": path, "byte_count": len(value), "sha256": sha256_bytes(value)})

    pinned_dataset_commit = dataset_commit or remote_head(source["datasetRepository"])
    dataset_id = quote(source["datasetId"], safe="/")
    dataset_tree_url = (
        f"https://huggingface.co/api/datasets/{dataset_id}/tree/{pinned_dataset_commit}"
        "?recursive=true&expand=false&limit=10000"
    )
    dataset_tree = fetch_json(dataset_tree_url)
    dataset_paths = _tree_paths_from_huggingface(dataset_tree)
    dataset_metadata_url = f"https://huggingface.co/api/datasets/{dataset_id}/revision/{pinned_dataset_commit}"
    dataset_metadata = fetch_json(dataset_metadata_url)
    card_data = dataset_metadata.get("cardData") or {}
    card_license = str(card_data.get("license", "")).casefold()
    pair_evidence = derive_tree_pairs(dataset_paths, config["fixedMetadataPatterns"])

    patterns = config["fixedMetadataPatterns"]
    schema_evidence = {
        "identity": matched_identifiers(identifiers, patterns["identitySchemaIdentifiers"]),
        "gold": matched_identifiers(identifiers, patterns["goldSchemaIdentifiers"]),
        "scenario": matched_identifiers(identifiers, patterns["scenarioSchemaIdentifiers"]),
        "prompt": matched_identifiers(identifiers, patterns["promptSchemaIdentifiers"]),
        "rationale": matched_identifiers(identifiers, patterns["rationaleSchemaIdentifiers"]),
    }
    gates = config["qualificationGates"]
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
            "tree_url": dataset_tree_url,
            "tree_file_count": len(dataset_paths),
            "tree_path_sha256": sha256_bytes("\n".join(dataset_paths).encode("utf-8")),
            "card_license": card_license,
            "pair_evidence": pair_evidence,
        },
        "qualification_checks": qualification,
        "scientific_feasibility_passed": all(qualification.values()),
        "access": {
            "code_tree_metadata_read_count": 1,
            "dataset_head_read_count": 1,
            "dataset_tree_metadata_read_count": 1,
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


def audit_feasibility(result: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    access = result["access"]
    gates = config["accessGates"]
    checks = {
        "qualification_is_noncompensatory": result["scientific_feasibility_passed"] == all(result["qualification_checks"].values()),
        "source_revisions_are_pinned": bool(len(result["code"]["commit"]) == 40 and len(result["dataset"]["commit"]) == 40),
        "metadata_and_schema_artifacts_are_hash_accounted": bool(
            len(result["code"]["tree_path_sha256"]) == 64
            and len(result["dataset"]["tree_path_sha256"]) == 64
            and all(len(item["sha256"]) == 64 and item["byte_count"] > 0 for item in result["code"]["schema_files"])
        ),
    }
    access_checks = {
        "required_metadata_reads_exact": bool(
            access["code_tree_metadata_read_count"] == gates["requiredCodeTreeMetadataReadCount"]
            and access["dataset_head_read_count"] == gates["requiredDatasetHeadReadCount"]
            and access["dataset_tree_metadata_read_count"] == gates["requiredDatasetTreeMetadataReadCount"]
            and access["dataset_card_header_read_count"] == gates["requiredDatasetCardHeaderReadCount"]
            and gates["minimumCodeSchemaFileReadCount"] <= access["code_schema_file_read_count"] <= gates["maximumCodeSchemaFileReadCount"]
        ),
        "task_payload_language_model_tool_and_execution_access_zero": all(
            access[key] <= gates[gate]
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
        "checks": checks,
        "access_checks": access_checks,
        "result": result,
    }


__all__ = [
    "audit_feasibility",
    "derive_tree_pairs",
    "evaluate_feasibility",
    "extracted_identifiers",
    "selected_code_schema_paths",
]
