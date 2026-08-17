#!/usr/bin/env python3
"""Audit and freeze the V68r2–V70 development-to-confirmation synthesis."""
from __future__ import annotations

import hashlib
import json
import math
from typing import Any

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT


SOURCES = {
    "V68r2": "configs/v68r2-development-outcome-lock.json",
    "V69": "configs/v69-development-outcome-lock.json",
    "V70": "configs/v70-confirmatory-outcome-lock.json",
}
SYNTHESIS = "docs/v68r2-v70-development-confirmation-synthesis.md"
RESEARCH_DIRECTION = "docs/research-direction.md"
AUDITOR = "python/audit_and_freeze_v68r2_v70_synthesis.py"
LOCK = "configs/v68r2-v70-synthesis-lock.json"


def payload_hash(value: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def load_json(relative_path: str) -> dict[str, Any]:
    return json.loads((PROJECT_ROOT / relative_path).read_text())


def lock_payload_is_valid(lock: dict[str, Any]) -> bool:
    payload = {key: value for key, value in lock.items() if key != "lock_payload_sha256"}
    return payload_hash(payload) == lock["lock_payload_sha256"]


def close(actual: float, expected: float) -> bool:
    return math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-15)


def source_is_bound(lock_path: str, lock: dict[str, Any]) -> bool:
    result_path = PROJECT_ROOT / lock["result"]
    return bool(
        lock_payload_is_valid(lock)
        and result_path.is_file()
        and file_sha256(result_path) == lock["result_sha256"]
        and (PROJECT_ROOT / lock_path).is_file()
    )


def audit_sources(
    locks: dict[str, dict[str, Any]], results: dict[str, dict[str, Any]]
) -> dict[str, bool]:
    v68 = results["V68r2"]
    v69 = results["V69"]
    v70 = results["V70"]

    v68_metrics = v68["metrics"]
    v69_metrics = v69["metrics"]
    v70_metrics = v70["metrics"]

    v68_ok = bool(
        not v68["passed"]
        and v68["records"] == 59
        and v68["decision"]
        == "stop_unchanged_family_before_any_confirmatory_model_is_scored"
        and v68_metrics["exact_BA_MAP_root_action_disagreement_records"] == 0
        and v68_metrics["material_regret_record_counts"]["map"] == 0
        and v68_metrics["material_regret_record_counts"]["posterior_sampling"] == 0
        and close(
            v68_metrics["maximum_normalized_MAP_regret"],
            0.0015135820075061503,
        )
        and v68["access"]["confirmatory_models_scored"] == 0
        and sum(not passed for passed in v68["gate_results"].values()) == 4
    )

    v69_ok = bool(
        v69["passed"]
        and v69["records"] == 59
        and v69["decision"]
        == "authorize_preregistration_of_confirmatory_multi_environment_design_only"
        and all(v69["gate_results"].values())
        and len(v69["gate_results"]) == 18
        and v69_metrics["exact_BA_MAP_root_action_disagreement_records"] == 8
        and v69_metrics["material_regret_record_counts"]["map"] == 8
        and v69_metrics["material_regret_record_counts"]["posterior_sampling"] == 16
        and close(
            v69_metrics["maximum_normalized_MAP_regret"],
            0.027595609725981726,
        )
        and v69["access"]["confirmatory_models_scored"] == 0
    )

    by_model = v70["by_model"]
    fallback_free = ("4x3.POMDP", "network.POMDP")
    complete_procedure = (
        "fully_observable_tmaze2.POMDP",
        "hallway.POMDP",
    )
    fallback_evidence_ok = bool(
        all(
            by_model[model]["qualifies_paired_MAP_replication"]
            and by_model[model]["fallback_diagnostics"]["map"][
                "affected_record_count"
            ]
            == 0
            for model in fallback_free
        )
        and all(
            by_model[model]["qualifies_paired_MAP_replication"]
            and by_model[model]["fallback_diagnostics"]["map"][
                "qualifying_MAP_overlap_count"
            ]
            == by_model[model]["qualifying_MAP_record_count"]
            for model in complete_procedure
        )
    )

    cheese_models = ("cheese.95.POMDP", "cheese.95_nonterminating.POMDP")
    tier_b_ok = all(
        not by_model[model]["qualifies_paired_MAP_replication"]
        and by_model[model]["root_action_disagreement_count"] == 0
        and by_model[model]["material_regret_count"]["map"] == 9
        and by_model[model]["fallback_diagnostics"]["map"][
            "material_regret_overlap_count"
        ]
        == 9
        for model in cheese_models
    )

    v70_ok = bool(
        v70["passed"]
        and v70["records"] == 244
        and v70["decision"]
        == "confirm_multi_environment_replication_for_project_authored_V69_family"
        and all(v70["gate_results"].values())
        and len(v70["gate_results"]) == 22
        and v70_metrics["paired_MAP_qualifying_model_count"] == 4
        and v70_metrics["paired_MAP_qualifying_structurally_related_model_count"]
        == 2
        and v70_metrics["paired_MAP_qualifying_novel_model_count"] == 2
        and v70_metrics["material_posterior_sampling_model_count"] == 6
        and close(
            v70_metrics["maximum_normalized_MAP_regret"],
            0.09544903686417067,
        )
        and sum(model["root_action_disagreement_count"] for model in by_model.values())
        == 60
        and sum(model["qualifying_MAP_record_count"] for model in by_model.values())
        == 18
        and sum(model["material_regret_count"]["map"] for model in by_model.values())
        == 44
        and sum(
            model["material_regret_count"]["posterior_sampling"]
            for model in by_model.values()
        )
        == 52
        and v70["access"]["development_models_rescored"] == 0
        and v70["access"]["records_selected_rejected_or_replaced"] == 0
    )

    access_ok = all(
        result["access"]["SMC2_runs"] == 0
        and result["access"]["human_records"] == 0
        and result["access"]["model_forward_passes"] == 0
        and result["access"]["adapter_training_runs"] == 0
        for result in results.values()
    )

    source_hashes_ok = all(
        source_is_bound(SOURCES[stage], locks[stage]) for stage in SOURCES
    )
    source_outcomes_match_locks = all(
        locks[stage]["outcome"]["passed"] == results[stage]["passed"]
        and locks[stage]["outcome"]["decision"] == results[stage]["decision"]
        and locks[stage]["outcome"]["records"] == results[stage]["records"]
        for stage in SOURCES
    )

    return {
        "source_outcome_locks_and_results_are_hash_bound": source_hashes_ok,
        "source_outcomes_match_durable_locks": source_outcomes_match_locks,
        "v68r2_negative_development_decision": v68_ok,
        "v69_positive_development_only_decision": v69_ok,
        "v70_positive_model_level_confirmation": v70_ok,
        "fallback_free_and_complete_procedure_evidence_are_separated": fallback_evidence_ok,
        "cheese_pair_is_tier_b_only": tier_b_ok,
        "zero_SMC2_human_model_and_adapter_access": access_ok,
    }


def audit_documents() -> dict[str, bool]:
    synthesis = (PROJECT_ROOT / SYNTHESIS).read_text()
    direction = (PROJECT_ROOT / RESEARCH_DIRECTION).read_text()
    synthesis_markers = (
        "V68r2 and V69 evaluate different uncertainty families.",
        "Fallback-free | `4x3.POMDP`",
        "Fallback-free | `network.POMDP`",
        "The full four-model confirmation therefore supports the",
        "The two cheese models are Tier B only",
        "Here, “better” means higher finite-horizon posterior-expected value",
        "The V69 family and all V70 models are closed for development.",
    )
    direction_markers = (
        "## Status after V70 (2026-08-17)",
        "V68r2 was a valid negative result.",
        "V70 then applied the unchanged V69 family",
        "The evidence is deliberately tiered.",
        "The next direction is synthesis and fresh boundary testing.",
        "Do not modify or rerun V69 or V70",
    )
    return {
        "synthesis_contains_required_claim_boundaries": all(
            marker in synthesis for marker in synthesis_markers
        ),
        "research_direction_records_closed_sequence": all(
            marker in direction for marker in direction_markers
        ),
    }


def build_lock(
    locks: dict[str, dict[str, Any]],
    results: dict[str, dict[str, Any]],
    checks: dict[str, bool],
) -> dict[str, Any]:
    source_outcomes = {}
    for stage, relative_path in SOURCES.items():
        source = locks[stage]
        result = results[stage]
        source_outcomes[stage] = {
            "outcome_lock": relative_path,
            "outcome_lock_sha256": file_sha256(PROJECT_ROOT / relative_path),
            "result": source["result"],
            "result_sha256": source["result_sha256"],
            "passed": bool(result["passed"]),
            "decision": result["decision"],
            "records": result["records"],
        }

    lock: dict[str, Any] = {
        "schema_version": "68r2-70-development-confirmation-synthesis",
        "experiment": "v68r2_v70_development_confirmation_synthesis_lock",
        "source_outcomes": source_outcomes,
        "synthesis_document": SYNTHESIS,
        "synthesis_document_sha256": file_sha256(PROJECT_ROOT / SYNTHESIS),
        "research_direction_snapshot": RESEARCH_DIRECTION,
        "research_direction_snapshot_sha256": file_sha256(
            PROJECT_ROOT / RESEARCH_DIRECTION
        ),
        "auditor": AUDITOR,
        "auditor_sha256": file_sha256(PROJECT_ROOT / AUDITOR),
        "evidence": {
            "negative_development_stage": "V68r2",
            "positive_development_stage": "V69",
            "positive_confirmation_stage": "V70",
            "V70_confirmatory_models": 9,
            "V70_records": 244,
            "V70_paired_MAP_qualifying_models": 4,
            "V70_paired_MAP_qualifying_related_models": 2,
            "V70_paired_MAP_qualifying_novel_models": 2,
            "V70_material_posterior_sampling_models": 6,
            "V70_fallback_free_qualifying_models": [
                "4x3.POMDP",
                "network.POMDP",
            ],
            "V70_complete_procedure_qualifying_models": [
                "fully_observable_tmaze2.POMDP",
                "hallway.POMDP",
            ],
            "V70_tier_B_models": [
                "cheese.95.POMDP",
                "cheese.95_nonterminating.POMDP",
            ],
        },
        "claim_boundary": {
            "externally_sourced_base_environments": True,
            "project_authored_uncertainty_family": True,
            "finite_horizon_posterior_expected_value_only": True,
            "external_uncertainty_family_claim": False,
            "approximate_inference_portability_claim": False,
            "long_horizon_or_real_world_claim": False,
            "V68r2_to_V69_magnitude_comparison": False,
        },
        "authorization": {
            "report_and_synthesize_completed_sequence": True,
            "modify_or_rerun_V69_or_V70": False,
            "rescore_or_replace_V70_models": False,
            "use_V70_as_development_data": False,
            "revise_frozen_gates": False,
            "new_family_only_after_new_preregistration_and_fresh_models": True,
            "run_SMC2": False,
            "human_data": False,
            "model_access": False,
            "adapter_training": False,
        },
        "checks": checks,
        "passed": all(checks.values()),
    }
    lock["lock_payload_sha256"] = payload_hash(lock)
    return lock


def main() -> None:
    locks = {stage: load_json(path) for stage, path in SOURCES.items()}
    results = {
        stage: load_json(locks[stage]["result"])
        for stage in SOURCES
    }
    checks = audit_sources(locks, results) | audit_documents()
    lock = build_lock(locks, results, checks)
    if not lock["passed"]:
        print(json.dumps(lock, indent=2, sort_keys=True))
        raise SystemExit(1)

    lock_path = PROJECT_ROOT / LOCK
    serialized = json.dumps(lock, indent=2, sort_keys=True) + "\n"
    if lock_path.exists():
        if lock_path.read_text() != serialized:
            raise RuntimeError("existing V68r2–V70 synthesis lock differs")
    else:
        lock_path.write_text(serialized)
    print(json.dumps(lock, indent=2, sort_keys=True))
    print(json.dumps({"lock": LOCK, "sha256": file_sha256(lock_path)}, indent=2))


if __name__ == "__main__":
    main()
