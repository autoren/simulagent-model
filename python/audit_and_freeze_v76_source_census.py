#!/usr/bin/env python3
"""Audit and freeze the metadata-only V76 source-census deferral."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT


PREREGISTRATION = "configs/v76-active-sensing-synthesis-lock.json"
INVENTORY = "configs/v76-discovery-clean-source-census-inventory.json"
REPORT = "docs/v76-discovery-clean-source-census-results.md"
DIRECTION = "docs/research-direction.md"
AUDITOR = "python/audit_and_freeze_v76_source_census.py"
AUDIT = "outputs/v76-discovery-clean-source-census/audit.json"
LOCK = "configs/v76-discovery-clean-source-census-lock.json"

EXPECTED_COMMITS = {
    "https://github.com/lsmcolab/ib-pomcp": "c825e524c71b7c77c113219f28bcb935ddf9177f",
    "https://github.com/proroklab/popgym": "410d5aa626dae8024f498354d8781a0d1870c399",
    "https://github.com/twni2016/pomdp-baselines": "e7c19c32a20033d75414b29fbc466c77c211e968",
    "https://github.com/penn-pal-lab/scaffolder": "80958717ded5e7f96043ffbffd268b46bb8cb7f9",
    "https://github.com/TimSchneider42/tactile-mnist": "9e4e59139e9349ab361a3b9297f4815724ad6387",
    "https://github.com/neuroergoISAE/POMDP-BCI": "9d576c3eae3e05d8bbe5fec1286d89c958242856",
    "https://github.com/gamma-opt/DecisionProgramming.jl": "105a25ee898cc806db65d5b475e4f1a613265653",
    "https://github.com/bonetblai/gpt-rewards": "b6a46c8bea7cd9b8244026d31ce4c259287c7da7",
    "https://github.com/infer-actively/pymdp": "58483604e56e330d3429df133a41e801099f6e9c",
    "https://github.com/ComputationalPsychiatry/ActiveInference.jl": "21216affd7f3ac20e28a4f7a11d68e13bc202f4f",
    "https://github.com/callmespring/Confounded-POMDP-OPE": "a6a5aa0fc2a7b9dbaa58670abea191684a327453",
    "https://github.com/opendatalab/Earth-Agent": "ca1dfaef750654331308a434966d307f1728fd90",
    "https://github.com/pemami4911/POMDPy": "ffa74ad627aade19cfa1773b01305e7736b2044d",
    "https://github.com/yaacovpariente/POMDPPlanners": "f18c5c9c214fab99f6e4c1e5b0ae31421e855ff7",
}


def canonical_hash(value: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def load_json(relative_path: str) -> dict[str, Any]:
    return json.loads((PROJECT_ROOT / relative_path).read_text())


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def payload_is_valid(lock: dict[str, Any]) -> bool:
    payload = {key: value for key, value in lock.items() if key != "lock_payload_sha256"}
    return canonical_hash(payload) == lock["lock_payload_sha256"]


def audit() -> dict[str, bool]:
    prereg = load_json(PREREGISTRATION)
    inventory = load_json(INVENTORY)
    candidates = inventory["candidates"]
    access = inventory["access"]
    gate_names = inventory["gateNames"]
    report = " ".join((PROJECT_ROOT / REPORT).read_text().split())
    direction = " ".join((PROJECT_ROOT / DIRECTION).read_text().split())

    repositories = [candidate["repository"] for candidate in candidates]
    identities = [candidate["candidateIdentity"] for candidate in candidates]
    commits_match = all(
        candidate["commit"] == EXPECTED_COMMITS[candidate["repository"]]
        and re.fullmatch(r"[0-9a-f]{40}", candidate["commit"]) is not None
        for candidate in candidates
    )
    gates_complete = all(
        len(candidate["gateResults"]) == len(gate_names)
        and all(isinstance(value, bool) for value in candidate["gateResults"])
        and candidate["eligible"] == all(candidate["gateResults"])
        and (candidate["eligible"] or any(not value for value in candidate["gateResults"]))
        for candidate in candidates
    )
    zero_access = all(value == 0 for value in access.values())

    return {
        "preregistration_is_valid_passed_and_authorizes_metadata_only_census": bool(
            prereg["passed"]
            and payload_is_valid(prereg)
            and prereg["authorization"]["execute_metadata_only_source_census"]
            and not prereg["authorization"]["clone_or_download_candidate_sources"]
            and not prereg["authorization"]["compute_candidate_policy_values_actions_regrets_EIG_or_mutual_information"]
        ),
        "search_began_after_durable_preregistration_commit": inventory["preregistrationGitCommit"]
        == "293bb769147b5b6c2856667155d71ab7faf89639",
        "complete_query_and_candidate_counts_are_bound": bool(
            inventory["searchScope"]["queryCount"] == 16
            and len(inventory["searchScope"]["queries"]) == 16
            and inventory["searchScope"]["candidateRepositoryCount"] == 14
            and len(candidates) == 14
        ),
        "candidate_repositories_identities_and_commits_are_unique_and_pinned": bool(
            len(set(repositories)) == len(repositories)
            and len(set(identities)) == len(identities)
            and set(repositories) == set(EXPECTED_COMMITS)
            and commits_match
        ),
        "all_eight_metadata_gates_are_complete_for_every_candidate": bool(
            len(gate_names) == 8 and gates_complete
        ),
        "no_candidate_is_eligible": bool(
            inventory["searchScope"]["eligibleRepositoryDistinctFamilyCount"] == 0
            and sum(candidate["eligible"] for candidate in candidates) == 0
        ),
        "empty_role_partition_follows_registered_minimum": bool(
            inventory["rolePartition"]["development"] is None
            and inventory["rolePartition"]["protectedConfirmation"] == []
            and "minimum is two" in inventory["rolePartition"]["reason"]
        ),
        "unexpected_search_snippet_is_disclosed_and_candidate_rejected": bool(
            inventory["unexpectedSnippetDisclosure"]["count"] == 1
            and inventory["unexpectedSnippetDisclosure"]["candidate"]
            == "infer-actively/pymdp#cue-chaining"
            and not next(
                candidate for candidate in candidates if candidate["repository"]
                == "https://github.com/infer-actively/pymdp"
            )["eligible"]
        ),
        "zero_forbidden_candidate_human_model_adapter_and_SMC2_access": zero_access,
        "decision_is_source_feasibility_deferral": bool(
            not inventory["passed"]
            and inventory["decision"]
            == "freeze_source_feasibility_deferral_before_any_candidate_implementation"
        ),
        "report_and_direction_record_deferral_without_gate_relaxation": all(
            marker in report
            for marker in (
                "found no eligible source pair",
                "Fourteen unique candidate repositories",
                "The deterministic role partition is empty",
                "active-sensing empirical branch should be reported and deferred",
            )
        )
        and all(
            marker in direction
            for marker in (
                "## V76 metadata-census outcome (2026-08-17)",
                "Fourteen repository candidates were recorded",
                "No development or protected-confirmation role was assigned",
                "report and defer the active-sensing empirical branch",
            )
        ),
    }


def build_lock(checks: dict[str, bool], audit_sha256: str) -> dict[str, Any]:
    inventory = load_json(INVENTORY)
    lock: dict[str, Any] = {
        "schema_version": "76-discovery-clean-source-census-outcome",
        "experiment": "v76_discovery_clean_source_census_outcome_lock",
        "passed": False,
        "decision": inventory["decision"],
        "checks": checks,
        "preregistration_lock": PREREGISTRATION,
        "preregistration_lock_sha256": file_sha256(PROJECT_ROOT / PREREGISTRATION),
        "preregistration_git_commit": inventory["preregistrationGitCommit"],
        "inventory": INVENTORY,
        "inventory_sha256": file_sha256(PROJECT_ROOT / INVENTORY),
        "report": REPORT,
        "report_sha256": file_sha256(PROJECT_ROOT / REPORT),
        "research_direction_snapshot": DIRECTION,
        "research_direction_snapshot_sha256": file_sha256(PROJECT_ROOT / DIRECTION),
        "audit": AUDIT,
        "audit_sha256": audit_sha256,
        "auditor": AUDITOR,
        "auditor_sha256": file_sha256(PROJECT_ROOT / AUDITOR),
        "outcome": {
            "query_count": 16,
            "candidate_repository_count": 14,
            "eligible_repository_distinct_family_count": 0,
            "development_role_assigned": False,
            "protected_confirmation_role_count": 0,
            "candidate_policy_value_count": 0,
            "candidate_implementation_file_open_count": 0,
        },
        "authorization": {
            "report_and_synthesize_source_feasibility_deferral": True,
            "inspect_or_select_any_V76_candidate_implementation": False,
            "compute_any_V76_candidate_policy_value_or_decision_statistic": False,
            "relax_V76_exposure_metadata_structural_resource_economic_or_role_gates": False,
            "reuse_V71_through_V75_sources_or_protected_models": False,
            "continue_active_sensing_empirical_branch_without_new_preregistration_or_new_public_sources": False,
            "human_data": False,
            "model_access": False,
            "adapter_training": False,
            "run_SMC2": False,
        },
    }
    lock["lock_payload_sha256"] = canonical_hash(lock)
    return lock


def main() -> None:
    checks = audit()
    audit_record = {
        "schema_version": "76-discovery-clean-source-census-audit",
        "experiment": "v76_discovery_clean_source_census_audit",
        "audit_passed": all(checks.values()),
        "scientific_gate_passed": False,
        "checks": checks,
    }
    audit_path = PROJECT_ROOT / AUDIT
    write_json(audit_path, audit_record)
    lock = build_lock(checks, file_sha256(audit_path))
    write_json(PROJECT_ROOT / LOCK, lock)
    print(json.dumps(audit_record, indent=2, sort_keys=True))
    if not audit_record["audit_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
