from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from cross_track_evidence_audit import ROOT, payload_hash, read_json, sha256_file, valid_lock, write_json
from diagnose_dependency_drift import DOCUMENT_PATH as DRIFT_DOCUMENT
from diagnose_dependency_drift import OUTPUT_PATH as DRIFT_OUTPUT
from diagnose_dependency_drift import diagnose, render as render_drift
from model_free_reference_architecture import (
    ACCESS_PATH as ARCH_ACCESS,
    AUDIT_PATH as ARCH_AUDIT,
    CONFIG_PATH as ARCH_CONFIG,
    RESULT_PATH as ARCH_RESULT,
    RESULTS_DOCUMENT as ARCH_RESULTS_DOCUMENT,
    render_results as render_architecture_results,
    run_reference_architecture,
)
from verify_and_freeze_cross_track_evidence_audit import verify_artifacts as verify_cross_track
from verify_and_freeze_model_free_reference_architecture import verify as verify_architecture


CONFIG_PATH = ROOT / "configs/post-v224-consolidation.json"
OUTPUT_DIR = ROOT / "outputs/post-v224-consolidation"
RESULT_PATH = OUTPUT_DIR / "result.json"
AUDIT_PATH = OUTPUT_DIR / "audit.json"
ACCESS_PATH = OUTPUT_DIR / "access.json"
MANIFEST_PATH = OUTPUT_DIR / "manifest.json"
RESULTS_DOCUMENT = ROOT / "docs/post-v224-consolidation-results.md"


def build_consolidation() -> dict[str, Any]:
    config = read_json(CONFIG_PATH)
    cross = verify_cross_track()
    cross_lock_path = ROOT / config["source_cross_track_outcome"]
    cross_lock = read_json(cross_lock_path)

    drift = diagnose()
    drift_exact = read_json(DRIFT_OUTPUT) == drift and DRIFT_DOCUMENT.read_text(encoding="utf-8") == render_drift(drift)

    architecture = verify_architecture()
    architecture_lock_path = ROOT / "configs/model-free-reference-architecture-integration-outcome-lock.json"
    architecture_lock = read_json(architecture_lock_path)
    architecture_config = read_json(ARCH_CONFIG)
    component_ids = [row["component_id"] for row in architecture_config["components"]]

    roadmaps = sorted((ROOT / "docs").glob("research-roadmap-after-v*.md"))
    canonical_roadmap = config["only_canonical_roadmap"]
    roadmap_status = [
        {
            "path": path.relative_to(ROOT).as_posix(),
            "status": "canonical_current" if path.relative_to(ROOT).as_posix() == canonical_roadmap else "historical_snapshot_not_authorization",
            "file_sha256": sha256_file(path),
        }
        for path in roadmaps
    ]
    historical = [row for row in roadmap_status if row["status"].startswith("historical")]

    docs = {relative: (ROOT / relative).read_text(encoding="utf-8") for relative in config["canonical_documents"]}
    normalized_docs = {
        relative: " ".join(text.replace("**", "").replace(">", "").split())
        for relative, text in docs.items()
    }
    navigation_visible = (
        "Older roadmap files are historical snapshots" in normalized_docs["README.md"]
        and "supersede every earlier roadmap authorization" in normalized_docs["docs/README.md"]
        and "Authorize zero new experiments" in normalized_docs["docs/research-stopping-rule-after-v224.md"]
        and "zero new experiments" in normalized_docs["docs/research-roadmap-after-v224.md"]
    )
    architecture_nonexperimental = (
        architecture["result"]["artifact_role"] == "software_and_claim_consolidation_only_not_a_new_experiment"
        and "not a new experiment" in normalized_docs["docs/model-free-reference-architecture.md"]
        and architecture_lock["outcome"]["new_scientific_experiment_count"] == 0
    )
    access = {
        "protected_body_read_count": 0,
        "request_language_read_count": 0,
        "model_load_count": 0,
        "model_generation_count": 0,
        "api_call_count": 0,
        "training_run_count": 0,
        "simulated_human_semantic_gold_count": 0,
        "ontology_registration_count": 0,
        "trusted_real_state_mutation_count": 0,
        "external_service_action_count": 0,
        "actual_execution_count": 0,
    }

    gates = {
        "cross_track_audit_reconstructs_and_outcome_lock_valid": cross["reproducibility_audit"]["payload_valid_count"] == 198 and valid_lock(cross_lock),
        "all_eight_dependency_drifts_have_exact_append_only_provenance": drift_exact and drift["finding_count"] == 8 and all(row["frozen_expected_sha256"] != row["current_sha256"] for row in drift["findings"]),
        "no_unavailable_frozen_dependency_is_guessed_or_overwritten": drift["exact_frozen_dependency_recovery_count"] == 0 and drift["current_dirty_dependency_count"] == 0 and all(row["resolution"] == "do_not_guess_or_overwrite_preserve_append_only_addendum" for row in drift["findings"]),
        "model_free_reference_architecture_reconstructs_and_outcome_lock_valid": architecture["audit"]["passed"] and valid_lock(architecture_lock),
        "reference_architecture_covers_all_required_mechanism_interfaces": component_ids == config["required_architecture_components"] and all(architecture["audit"]["gates"].values()),
        "integration_is_explicitly_non_experimental_and_non_external_language_evidence": architecture_nonexperimental and "not externally validated" in normalized_docs["README.md"],
        "canonical_navigation_marks_pre_V224_roadmaps_historical_without_editing_them": navigation_visible and len(roadmap_status) == 28 and len(historical) == 27 and all(row["path"] != canonical_roadmap for row in historical),
        "V224_stopping_rule_remains_canonical": cross["stopping_decision"]["authorized_next_experiment_count"] == 0 and canonical_roadmap == "docs/research-roadmap-after-v224.md",
        "protected_request_language_model_API_training_registration_real_action_and_execution_counts_zero": all(value == 0 for value in access.values()),
    }
    if list(gates) != config["required_gates"]:
        raise AssertionError("Consolidation gate implementation differs from frozen config")
    passed = all(gates.values())
    result = {
        "schema_version": "post_v224_consolidation_result.v1",
        "consolidation_id": config["consolidation_id"],
        "artifact_role": config["artifact_role"],
        "gates": gates,
        "passed": passed,
        "dependency_drift": {
            "finding_count": drift["finding_count"],
            "recovered_count": drift["exact_frozen_dependency_recovery_count"],
            "dirty_current_count": drift["current_dirty_dependency_count"],
            "decision": drift["decision"],
            "addendum_sha256": payload_hash(drift),
        },
        "reference_architecture": {
            "component_ids": component_ids,
            "source_outcome_lock_count": len(architecture["result"]["source_lock_integrity"]),
            "integration_gate_count": len(architecture["audit"]["gates"]),
            "integration_passed": architecture["audit"]["passed"],
            "trusted_route": architecture["result"]["typed_version_space"]["routed_decision"],
            "other_route": architecture["result"]["other_defer"]["decision"],
            "external_semantic_adapter_status": "absent_pending_stopping_rule_reopening_condition",
        },
        "navigation": {
            "roadmap_count": len(roadmap_status),
            "historical_roadmap_count": len(historical),
            "canonical_roadmap": canonical_roadmap,
            "roadmaps": roadmap_status,
            "canonical_documents": config["canonical_documents"],
        },
        "decision": "freeze_post_V224_reproducibility_architecture_and_navigation_consolidation_without_experimental_escalation" if passed else "retain_consolidation_failure_and_repair_maintenance_artifacts_only",
        "next_state": "monitor_external_reopening_conditions_while_preserving_zero_experiment_authorization",
        "access": access,
    }
    audit = {
        "schema_version": "post_v224_consolidation_audit.v1",
        "passed": passed,
        "gates": gates,
        "result_sha256": payload_hash(result),
        "cross_track_outcome_sha256": sha256_file(cross_lock_path),
        "reference_architecture_outcome_sha256": sha256_file(architecture_lock_path),
        "dependency_addendum_sha256": sha256_file(DRIFT_OUTPUT),
    }
    return {"result": result, "audit": audit, "access": access}


def render_results(bundle: dict[str, Any]) -> str:
    result = bundle["result"]
    return "\n".join(
        [
            "# Post-V224 consolidation result",
            "",
            "## Outcome",
            "",
            f"The maintenance consolidation {'passed' if result['passed'] else 'failed'} all {len(result['gates'])} frozen gates. It creates no new experiment and does not change the V224 stopping decision.",
            "",
            "## Provenance",
            "",
            f"All {result['dependency_drift']['finding_count']} dependency drifts now have an append-only machine-readable diagnosis. Exact recoveries: `{result['dependency_drift']['recovered_count']}`; dirty current targets: `{result['dependency_drift']['dirty_current_count']}`. Because the frozen byte sequences are absent from reachable refs/reflogs, no historical lock or current target was overwritten.",
            "",
            "## Reference architecture",
            "",
            f"The specification maps {len(result['reference_architecture']['component_ids'])} interfaces to {result['reference_architecture']['source_outcome_lock_count']} valid frozen outcomes. The integration harness passed {result['reference_architecture']['integration_gate_count']} gates: the one-corruption trusted branch routed `{result['reference_architecture']['trusted_route']}`, while uninterpretable OTHER routed `{result['reference_architecture']['other_route']}` without sandbox entry. The external semantic adapter remains absent.",
            "",
            "## Navigation",
            "",
            f"The documentation index records {result['navigation']['historical_roadmap_count']} pre-V224 roadmaps as historical snapshots and `{result['navigation']['canonical_roadmap']}` as the sole current roadmap. Historical hash-linked documents were not edited.",
            "",
            "## Continuing status",
            "",
            "Experimental and model escalation remain frozen. Allowed work is reproducibility maintenance, reference-architecture maintenance, documentation, and monitoring for an independently admissible external semantic evidence source or qualified speaker/expert channel."
        ]
    ) + "\n"


def write_bundle(bundle: dict[str, Any]) -> None:
    write_json(RESULT_PATH, bundle["result"])
    write_json(AUDIT_PATH, bundle["audit"])
    write_json(ACCESS_PATH, bundle["access"])
    RESULTS_DOCUMENT.write_text(render_results(bundle), encoding="utf-8")
    artifacts = [
        CONFIG_PATH,
        RESULTS_DOCUMENT,
        ROOT / "docs/README.md",
        ROOT / "README.md",
        DRIFT_OUTPUT,
        DRIFT_DOCUMENT,
        ROOT / "configs/model-free-reference-architecture-integration-outcome-lock.json",
        ROOT / "python/post_v224_consolidation.py",
        ROOT / "python/run_post_v224_consolidation.py",
        ROOT / "python/test_post_v224_consolidation.py",
        ROOT / "python/verify_and_freeze_post_v224_consolidation.py",
        RESULT_PATH,
        AUDIT_PATH,
        ACCESS_PATH,
    ]
    write_json(
        MANIFEST_PATH,
        {
            "schema_version": "post_v224_consolidation_manifest.v1",
            "artifacts": [
                {
                    "path": path.resolve().relative_to(ROOT.resolve()).as_posix(),
                    "sha256": sha256_file(path),
                    "size_bytes": path.stat().st_size,
                }
                for path in sorted(artifacts)
            ],
        },
    )
