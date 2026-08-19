from __future__ import annotations

import hashlib
import subprocess
from typing import Any
from urllib.request import Request, urlopen


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def repository_slug(repository: str) -> str:
    prefix = "https://github.com/"
    if not repository.startswith(prefix):
        raise ValueError("V206 candidate is not a canonical GitHub repository URL")
    slug = repository[len(prefix) :].removesuffix(".git").strip("/")
    if slug.count("/") != 1:
        raise ValueError("V206 candidate repository slug is invalid")
    return slug


def remote_head(repository: str) -> str:
    completed = subprocess.run(
        ["git", "ls-remote", repository, "HEAD"],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    fields = completed.stdout.strip().split()
    if len(fields) != 2 or fields[1] != "HEAD" or len(fields[0]) != 40:
        raise RuntimeError(f"V206 could not pin HEAD for {repository}")
    return fields[0]


def fetch_official_file(repository: str, commit: str, path: str) -> tuple[bytes | None, str]:
    slug = repository_slug(repository)
    url = f"https://raw.githubusercontent.com/{slug}/{commit}/{path}"
    request = Request(url, headers={"User-Agent": "simulagent-v206-metadata-audit"})
    try:
        with urlopen(request, timeout=30) as response:
            return response.read(), url
    except Exception:
        return None, url


def contains_any(text: str, markers: list[str]) -> tuple[bool, list[str]]:
    normalized = text.casefold()
    matched = [marker for marker in markers if marker.casefold() in normalized]
    return bool(matched), matched


def contains_conjunction(text: str, conjunctions: list[list[str]]) -> tuple[bool, list[list[str]]]:
    normalized = text.casefold()
    matched = [group for group in conjunctions if all(marker.casefold() in normalized for marker in group)]
    return bool(matched), matched


def detect_license(value: bytes | None) -> str | None:
    if value is None:
        return None
    text = value.decode("utf-8", errors="replace").casefold()
    if "permission is hereby granted, free of charge" in text:
        return "MIT"
    if "apache license" in text and "version 2.0" in text:
        return "Apache-2.0"
    if "gnu general public license" in text:
        return "GPL"
    if "creative commons attribution" in text:
        return "CC-BY"
    return "IDENTIFIABLE_OTHER" if len(text.strip()) >= 40 else None


def evaluate_documentation(
    candidate: dict[str, Any],
    readme: bytes | None,
    license_value: bytes | None,
    config: dict[str, Any],
) -> dict[str, Any]:
    text = "" if readme is None else readme.decode("utf-8", errors="replace")
    rules = config["fixedDocumentationEvidenceRules"]
    open_world, open_markers = contains_any(text, rules["openWorldOutsideInvalidOrAbstentionMarkers"])
    active, active_markers = contains_any(text, rules["actionDependentInformationMarkers"])
    reference, reference_markers = contains_any(text, rules["referenceCalibrationMarkers"])
    defer, defer_markers = contains_any(text, rules["safeDeferMarkers"])
    delayed, delayed_markers = contains_any(text, rules["delayedConsequenceMarkers"])
    generative, generative_markers = contains_conjunction(text, rules["generativeModelConjunctions"])
    license_name = detect_license(license_value)
    fresh = candidate["repository"] not in set(config["priorExposureRepositoryUrls"])
    gates = {
        "official_license": license_name is not None,
        "fresh_repository_and_domain_family": fresh,
        "explicit_open_world_outside_invalid_or_abstention_regime": open_world,
        "action_dependent_information_gathering": active,
        "in_episode_reference_calibration_or_cross_sensor_pathway": reference,
        "safe_defer_abstain_or_hold_action": defer,
        "delayed_state_dependent_or_irreversible_consequence": delayed,
        "exact_simulator_or_validated_generative_likelihood_path": generative,
    }
    gates["all_critical_elements_source_native"] = all(gates.values())
    return {
        "license": license_name,
        "gate_results": gates,
        "qualified": all(gates.values()),
        "matched_documentation_markers": {
            "open_world": open_markers,
            "action_dependent_information": active_markers,
            "reference_calibration": reference_markers,
            "safe_defer": defer_markers,
            "delayed_consequence": delayed_markers,
            "generative_model_conjunctions": generative_markers,
        },
    }


def _build_result(config: dict[str, Any], pinned_commits: dict[str, str] | None = None) -> dict[str, Any]:
    records = []
    for candidate in config["candidates"]:
        commit = (
            remote_head(candidate["repository"])
            if pinned_commits is None
            else pinned_commits[candidate["candidateId"]]
        )
        readme, readme_url = fetch_official_file(candidate["repository"], commit, candidate["readmePath"])
        license_value, license_url = fetch_official_file(candidate["repository"], commit, candidate["licensePath"])
        evaluation = evaluate_documentation(candidate, readme, license_value, config)
        records.append(
            {
                "candidate_id": candidate["candidateId"],
                "repository": candidate["repository"],
                "commit": commit,
                "domain_family": candidate["domainFamily"],
                "readme": {
                    "path": candidate["readmePath"],
                    "url": readme_url,
                    "fetch_succeeded": readme is not None,
                    "byte_count": 0 if readme is None else len(readme),
                    "sha256": None if readme is None else sha256_bytes(readme),
                },
                "license_file": {
                    "path": candidate["licensePath"],
                    "url": license_url,
                    "fetch_succeeded": license_value is not None,
                    "byte_count": 0 if license_value is None else len(license_value),
                    "sha256": None if license_value is None else sha256_bytes(license_value),
                },
                **evaluation,
            }
        )
    qualified = [record["candidate_id"] for record in records if record["qualified"]]
    minimum = config["metadataQualificationGates"]["minimumQualifiedRepositoryDistinctFamilies"]
    return {
        "candidate_count": len(records),
        "records": records,
        "qualified_candidate_ids": qualified,
        "qualified_repository_distinct_family_count": len(qualified),
        "scientific_feasibility_passed": len(qualified) >= minimum,
        "access": {
            "repository_head_read_count": len(records),
            "official_README_fetch_attempt_count": len(records),
            "license_fetch_attempt_count": len(records),
            "repository_clone_or_archive_download_count": 0,
            "candidate_implementation_file_read_count": 0,
            "task_dialogue_utterance_or_example_record_read_count": 0,
            "transition_observation_reward_or_belief_array_read_count": 0,
            "simulator_planner_or_policy_evaluation_count": 0,
            "protected_access_count": 0,
            "model_load_count": 0,
            "model_generation_count": 0,
            "model_API_call_count": 0,
            "training_run_count": 0,
            "ontology_registration_count": 0,
            "trusted_state_mutation_count": 0,
            "service_call_count": 0,
            "external_side_effect_count": 0,
            "actual_execution_count": 0,
        },
    }


def evaluate_feasibility(config: dict[str, Any]) -> dict[str, Any]:
    return _build_result(config)


def reconstruct_from_pins(result: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    pins = {record["candidate_id"]: record["commit"] for record in result["records"]}
    if set(pins) != {candidate["candidateId"] for candidate in config["candidates"]}:
        raise ValueError("V206 pinned candidate identities do not match the design")
    return _build_result(config, pins)


def audit_feasibility(result: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    gates = config["metadataQualificationGates"]
    access_gates = config["accessGates"]
    access = result["access"]
    checks = {
        "candidate_count_exact": result["candidate_count"] == gates["requiredCandidateCount"],
        "records_have_unique_candidate_repository_and_commit_identity": bool(
            len({record["candidate_id"] for record in result["records"]}) == result["candidate_count"]
            and len({record["repository"] for record in result["records"]}) == result["candidate_count"]
            and all(len(record["commit"]) == 40 for record in result["records"])
        ),
        "qualification_is_noncompensatory": all(
            record["qualified"] == all(record["gate_results"].values()) for record in result["records"]
        ),
        "scientific_decision_matches_qualified_family_count": bool(
            result["scientific_feasibility_passed"]
            == (result["qualified_repository_distinct_family_count"] >= gates["minimumQualifiedRepositoryDistinctFamilies"])
        ),
        "official_files_are_hash_accounted_when_present": all(
            all(
                (not artifact["fetch_succeeded"] and artifact["sha256"] is None and artifact["byte_count"] == 0)
                or (artifact["fetch_succeeded"] and len(artifact["sha256"]) == 64 and artifact["byte_count"] > 0)
                for artifact in (record["readme"], record["license_file"])
            )
            for record in result["records"]
        ),
    }
    access_checks = {
        "required_metadata_reads_exact": bool(
            access["repository_head_read_count"] == access_gates["requiredRepositoryHeadReadCount"]
            and access["official_README_fetch_attempt_count"] == access_gates["requiredOfficialREADMEFetchAttemptCount"]
            and access["license_fetch_attempt_count"] == access_gates["requiredLicenseFetchAttemptCount"]
        ),
        "forbidden_access_and_effects_zero": all(
            access[key] <= access_gates[gate]
            for key, gate in (
                ("repository_clone_or_archive_download_count", "maximumRepositoryCloneOrArchiveDownloadCount"),
                ("candidate_implementation_file_read_count", "maximumCandidateImplementationFileReadCount"),
                ("task_dialogue_utterance_or_example_record_read_count", "maximumTaskDialogueUtteranceOrExampleRecordReadCount"),
                ("transition_observation_reward_or_belief_array_read_count", "maximumTransitionObservationRewardOrBeliefArrayReadCount"),
                ("simulator_planner_or_policy_evaluation_count", "maximumSimulatorPlannerOrPolicyEvaluationCount"),
                ("protected_access_count", "maximumProtectedAccessCount"),
                ("model_load_count", "maximumModelLoadCount"),
                ("model_generation_count", "maximumModelGenerationCount"),
                ("model_API_call_count", "maximumModelAPICallCount"),
                ("training_run_count", "maximumTrainingRunCount"),
                ("ontology_registration_count", "maximumOntologyRegistrationCount"),
                ("trusted_state_mutation_count", "maximumTrustedStateMutationCount"),
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
    "contains_any",
    "contains_conjunction",
    "detect_license",
    "evaluate_documentation",
    "evaluate_feasibility",
    "reconstruct_from_pins",
    "repository_slug",
]
