from __future__ import annotations

import hashlib
import json
import subprocess
from typing import Any
from urllib.parse import quote
from urllib.request import Request, urlopen


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def fetch_json(url: str) -> Any:
    request = Request(url, headers={"User-Agent": "simulagent-v208-source-census"})
    with urlopen(request, timeout=45) as response:
        return json.loads(response.read().decode("utf-8"))


def remote_head(repository: str) -> str | None:
    completed = subprocess.run(
        ["git", "ls-remote", repository, "HEAD"],
        check=False,
        capture_output=True,
        text=True,
        timeout=45,
    )
    fields = completed.stdout.strip().split()
    if completed.returncode != 0 or len(fields) != 2 or fields[1] != "HEAD" or len(fields[0]) != 40:
        return None
    return fields[0]


def github_slug(repository: str) -> str:
    prefix = "https://github.com/"
    if not repository.startswith(prefix):
        raise ValueError("V208 requires a canonical GitHub repository")
    return repository[len(prefix) :].removesuffix(".git").strip("/")


def marker_counts(paths: list[str], groups: dict[str, list[str]]) -> dict[str, int]:
    normalized = [path.casefold() for path in paths]
    return {
        group: sum(1 for path in normalized if any(marker.casefold() in path for marker in markers))
        for group, markers in groups.items()
    }


def evaluate_candidate(
    candidate: dict[str, Any],
    config: dict[str, Any],
    *,
    pinned_code_commit: str | None = None,
    pinned_dataset_commit: str | None = None,
) -> dict[str, Any]:
    slug = github_slug(candidate["codeRepository"])
    code_commit = pinned_code_commit or remote_head(candidate["codeRepository"])
    repository_metadata = fetch_json(f"https://api.github.com/repos/{slug}")
    license_value = repository_metadata.get("license") or {}
    code_license = str(license_value.get("spdx_id") or "").strip()
    tree_paths: list[str] = []
    tree_truncated = True
    if code_commit:
        tree = fetch_json(f"https://api.github.com/repos/{slug}/git/trees/{code_commit}?recursive=1")
        tree_truncated = bool(tree.get("truncated"))
        tree_paths = sorted(item["path"] for item in tree.get("tree", []) if item.get("type") == "blob")

    dataset_repository = candidate.get("datasetRepository")
    dataset_commit = None
    dataset_license = None
    dataset_metadata_available = dataset_repository is None
    if dataset_repository:
        dataset_commit = pinned_dataset_commit or remote_head(dataset_repository)
        if dataset_commit:
            dataset_id = dataset_repository.split("/datasets/", 1)[1]
            metadata = fetch_json(
                f"https://huggingface.co/api/datasets/{quote(dataset_id, safe='/')}/revision/{dataset_commit}"
            )
            dataset_license = str((metadata.get("cardData") or {}).get("license") or "").strip()
            dataset_metadata_available = True

    facts = candidate["landingFacts"]
    gates = config["qualificationGates"]
    checks = {
        "immutable_code_revision": bool(code_commit and len(code_commit) == 40 and not tree_truncated),
        "immutable_external_dataset_revision_when_declared": bool(
            dataset_repository is None or (dataset_commit and len(dataset_commit) == 40 and dataset_metadata_available)
        ),
        "recognized_code_license_metadata": bool(code_license and code_license.casefold() not in {"noassertion", "other"}),
        "recognized_dataset_license_metadata_when_declared": bool(
            dataset_repository is None or (dataset_license and dataset_license.casefold() not in {"noassertion", "other"})
        ),
        "minimum_public_units": candidate["declaredPublicUnitCount"] >= gates["minimumDeclaredPublicUnitCount"],
        "minimum_pairs": candidate["declaredPairCount"] >= gates["minimumDeclaredPairCount"],
        "minimum_scenarios": candidate["declaredScenarioCount"] >= gates["minimumDeclaredScenarioCount"],
        "paired_or_matched_controls": facts["pairedOrMatchedControlsClaimed"],
        "explicit_machine_readable_pair_identity_field": facts["explicitMachineReadablePairIdentityFieldClaimed"],
        "explicit_preexecution_phase_identity": facts["explicitPreExecutionPhaseIdentityClaimed"],
        "deterministic_act_abstain_gold_field": facts["deterministicActAbstainGoldFieldClaimed"],
        "prompt_label_separation": facts["promptLabelSeparationClaimed"],
        "balanced_act_abstain_controls": facts["balancedActAbstainControlsClaimed"],
        "text_only_shadow_classification": facts["textOnlyShadowClassificationSupported"],
        "gold_independent_of_LLM_judge": facts["goldIndependentOfLLMJudgeClaimed"],
        "runtime_interaction_not_required": facts["runtimeInteractionRequired"] is gates["maximumRuntimeInteractionRequired"],
    }
    return {
        "candidate_id": candidate["candidateId"],
        "code": {
            "repository": candidate["codeRepository"],
            "commit": code_commit,
            "license_spdx": code_license,
            "tree_blob_count": len(tree_paths),
            "tree_truncated": tree_truncated,
            "tree_path_sha256": sha256_text("\n".join(tree_paths)),
            "tree_marker_counts": marker_counts(tree_paths, config["fixedTreePathMarkerGroups"]),
        },
        "dataset": {
            "repository": dataset_repository,
            "commit": dataset_commit,
            "license": dataset_license,
            "metadata_available": dataset_metadata_available,
        },
        "declared_counts": {
            "public_units": candidate["declaredPublicUnitCount"],
            "pairs": candidate["declaredPairCount"],
            "scenarios": candidate["declaredScenarioCount"],
        },
        "landing_facts": facts,
        "qualification_checks": checks,
        "scientific_feasibility_passed": all(checks.values()),
    }


def evaluate_census(
    config: dict[str, Any],
    *,
    pinned_revisions: dict[str, dict[str, str | None]] | None = None,
) -> dict[str, Any]:
    records = []
    for candidate in config["candidates"]:
        pinned = (pinned_revisions or {}).get(candidate["candidateId"], {})
        records.append(
            evaluate_candidate(
                candidate,
                config,
                pinned_code_commit=pinned.get("code"),
                pinned_dataset_commit=pinned.get("dataset"),
            )
        )
    repositories = [candidate["codeRepository"] for candidate in config["candidates"]]
    census_checks = {
        "minimum_candidate_count": len(records) >= config["qualificationGates"]["minimumCandidateCount"],
        "distinct_repository_per_candidate": len(repositories) == len(set(repositories)),
        "AgentAbstain_not_reopened": all("agentabstain" not in repository.casefold() for repository in repositories),
    }
    eligible = sorted(record["candidate_id"] for record in records if record["scientific_feasibility_passed"])
    return {
        "records": records,
        "eligible_candidate_ids": eligible,
        "eligible_candidate_count": len(eligible),
        "census_checks": census_checks,
        "scientific_feasibility_passed": bool(eligible) and all(census_checks.values()),
        "access": {
            "github_repository_object_metadata_read_count": len(records),
            "github_recursive_tree_metadata_read_count": len(records),
            "github_README_or_other_blob_body_read_count": 0,
            "github_license_blob_body_read_count": 0,
            "huggingFace_dataset_head_read_count": sum(record["dataset"]["repository"] is not None for record in records),
            "huggingFace_card_data_object_metadata_read_count": sum(record["dataset"]["repository"] is not None for record in records),
            "huggingFace_tree_or_payload_read_count": 0,
            "task_instruction_example_dialogue_rationale_response_read_count": 0,
            "model_or_policy_evaluation_count": 0,
            "API_call_count": 0,
            "training_run_count": 0,
            "tool_or_service_call_count": 0,
            "external_side_effect_count": 0,
            "actual_execution_count": 0,
        },
    }


def audit_census(summary: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    allowed = config["allowedEvaluationReads"]
    access = summary["access"]
    checks = {
        "census_is_noncompensatory": summary["scientific_feasibility_passed"] == bool(summary["eligible_candidate_ids"]) and all(summary["census_checks"].values()),
        "every_candidate_is_noncompensatory": all(
            record["scientific_feasibility_passed"] == all(record["qualification_checks"].values())
            for record in summary["records"]
        ),
        "all_code_tree_artifacts_hash_accounted": all(
            len(record["code"]["tree_path_sha256"]) == 64 for record in summary["records"]
        ),
        "allowed_metadata_counts_exact": bool(
            access["github_repository_object_metadata_read_count"] == len(config["candidates"])
            and access["github_recursive_tree_metadata_read_count"] == len(config["candidates"])
            and access["huggingFace_dataset_head_read_count"] == sum(candidate["datasetRepository"] is not None for candidate in config["candidates"])
            and access["huggingFace_card_data_object_metadata_read_count"] == sum(candidate["datasetRepository"] is not None for candidate in config["candidates"])
        ),
        "blob_payload_language_model_API_tool_and_execution_zero": bool(
            access["github_README_or_other_blob_body_read_count"] == allowed["githubREADMEOrOtherBlobBodyReadCount"]
            and access["github_license_blob_body_read_count"] == allowed["githubLicenseBlobBodyReadCount"]
            and access["huggingFace_tree_or_payload_read_count"] == allowed["huggingFaceTreeOrPayloadReadCount"]
            and access["task_instruction_example_dialogue_rationale_response_read_count"] == allowed["taskInstructionExampleDialogueRationaleResponseReadCount"]
            and access["model_or_policy_evaluation_count"] == allowed["modelOrPolicyEvaluationCount"]
            and all(
                access[key] == 0
                for key in (
                    "API_call_count",
                    "training_run_count",
                    "tool_or_service_call_count",
                    "external_side_effect_count",
                    "actual_execution_count",
                )
            )
        ),
    }
    return {"passed": all(checks.values()), "checks": checks, "summary": summary}


__all__ = ["audit_census", "evaluate_candidate", "evaluate_census", "marker_counts"]
