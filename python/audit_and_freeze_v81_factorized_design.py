#!/usr/bin/env python3
"""Audit and freeze the fresh V81 factorized local-model design."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT


def payload_hash(value: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def main() -> None:
    design_path = PROJECT_ROOT / "configs/v81-factorized-local-candidate-design.json"
    predecessor_path = (
        PROJECT_ROOT / "configs/v80-local-candidate-generation-outcome-lock.json"
    )
    plan_path = PROJECT_ROOT / "docs/v81-factorized-local-candidate-plan.md"
    auditor_path = PROJECT_ROOT / "python/audit_and_freeze_v81_factorized_design.py"
    audit_path = PROJECT_ROOT / "outputs/v81-factorized-local-candidate/design-audit.json"
    lock_path = PROJECT_ROOT / "configs/v81-factorized-local-candidate-design-lock.json"
    if audit_path.exists() or lock_path.exists():
        raise RuntimeError("V81 factorized design is already frozen")
    if (PROJECT_ROOT / "outputs/v81-factorized-local-candidate/evaluation").exists():
        raise RuntimeError("V81 outcome exists before design lock")
    audit_path.parent.mkdir(parents=True, exist_ok=True)

    design = json.loads(design_path.read_text())
    predecessor = json.loads(predecessor_path.read_text())
    predecessor_payload = {
        key: value
        for key, value in predecessor.items()
        if key != "lock_payload_sha256"
    }
    records = design["records"]
    v80_lock = json.loads(
        (PROJECT_ROOT / "configs/v80-local-candidate-generation-design-lock.json").read_text()
    )
    old_instructions = {
        record["instruction"] for record in v80_lock["config_payload"]["records"]
    }
    keys = design["labelKeysInRequiredOrder"]
    expected_candidates = design["candidateIdsInRequiredOrder"]

    def compose(labels: dict[str, bool]) -> list[str]:
        if labels["out_of_ontology"]:
            return ["none_of_the_above"]
        candidates: list[str] = []
        for recipient in ("alex_chen", "alex_kim"):
            for operation in ("schedule_review", "send_summary"):
                if labels[operation] and labels[recipient]:
                    candidates.append(f"{operation}__{recipient}")
        candidates.append("none_of_the_above")
        return candidates

    record_shape = all(
        list(record["goldLabels"]) == keys
        and all(type(value) is bool for value in record["goldLabels"].values())
        and record["goldLabels"]["out_of_ontology"]
        == (
            not record["goldLabels"]["schedule_review"]
            and not record["goldLabels"]["send_summary"]
        )
        and compose(record["goldLabels"]) == record["goldCandidateIds"]
        for record in records
    )
    counts: dict[str, int] = {}
    for record in records:
        counts[record["stratum"]] = counts.get(record["stratum"], 0) + 1
    checks = {
        "verified_V80_negative_authorizes_fresh_successor_design": bool(
            payload_hash(predecessor_payload) == predecessor["lock_payload_sha256"]
            and not predecessor["outcome"]["passed"]
            and predecessor["authorization"]["design_and_preregister_fresh_successor"]
            and not predecessor["authorization"]["run_V80_local_model_again"]
        ),
        "factorized_fixed_boolean_contract_and_deterministic_composition": bool(
            keys
            == [
                "schedule_review",
                "send_summary",
                "alex_chen",
                "alex_kim",
                "out_of_ontology",
            ]
            and expected_candidates[-1] == "none_of_the_above"
            and record_shape
        ),
        "fresh_complete_unique_population": bool(
            len(records) == 24
            and len({record["id"] for record in records}) == 24
            and len({record["instruction"] for record in records}) == 24
            and not ({record["instruction"] for record in records} & old_instructions)
            and counts == design["gates"]["requiredStratumCounts"]
        ),
        "local_frozen_model_and_deterministic_no_retry_decoding": bool(
            design["model"]["provider"] == "local_mlx"
            and design["model"]["frozen"]
            and not design["model"]["APIRequired"]
            and design["model"]["adapterPath"] is None
            and design["decoding"]["temperature"] == 0.0
            and design["decoding"]["samplesPerRecord"] == 1
            and not design["decoding"]["retryOnMalformedOutput"]
        ),
        "noncompensatory_quality_and_access_gates_present": bool(
            design["gates"]["minimumSchemaValidityRate"] == 1.0
            and design["gates"]["minimumMeanLabelAccuracy"] >= 0.95
            and design["gates"]["minimumOutOfOntologyExactCandidateSetAccuracy"]
            >= 0.75
            and design["gates"]["maximumModelGenerationCount"] == 24
            and design["gates"]["maximumAPICallCount"] == 0
            and design["gates"]["maximumAdapterTrainingRunCount"] == 0
            and design["gates"]["maximumHumanRecordAccessCount"] == 0
            and design["gates"]["maximumRealToolCallCount"] == 0
            and design["gates"]["maximumExternalSideEffectCount"] == 0
        ),
        "design_stage_authorizes_no_model_or_external_access": bool(
            design["stageAuthorization"]["auditAndFreezeProtocol"]
            and not design["stageAuthorization"]["runLocalModel"]
            and not design["stageAuthorization"]["runAPIModel"]
            and not design["stageAuthorization"]["trainAdapter"]
            and not design["stageAuthorization"]["collectHumanLanguage"]
            and not design["stageAuthorization"]["performRealToolCall"]
            and not design["stageAuthorization"]["performExternalSideEffect"]
        ),
    }
    passed = all(checks.values())
    audit = {
        "schema_version": "81-factorized-local-candidate-design-audit",
        "experiment": "v81_factorized_local_candidate_design_audit",
        "passed": passed,
        "decision": (
            "freeze_design_and_authorize_corpus_runner_implementation_only"
            if passed
            else "reject_V81_design"
        ),
        "checks": checks,
        "access": {
            "model_load_count": 0,
            "model_generation_count": 0,
            "API_call_count": 0,
            "adapter_training_run_count": 0,
            "human_record_access_count": 0,
            "real_tool_call_count": 0,
            "external_side_effect_count": 0,
        },
    }
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    if not passed:
        print(json.dumps(audit, indent=2, sort_keys=True))
        raise SystemExit(1)
    lock = {
        "schema_version": "81-factorized-local-candidate-design-lock",
        "experiment": "v81_factorized_local_candidate_design_lock",
        "design": str(design_path.relative_to(PROJECT_ROOT)),
        "design_sha256": file_sha256(design_path),
        "config_payload": design,
        "predecessor_outcome_lock": str(predecessor_path.relative_to(PROJECT_ROOT)),
        "predecessor_outcome_lock_sha256": file_sha256(predecessor_path),
        "plan": str(plan_path.relative_to(PROJECT_ROOT)),
        "plan_sha256": file_sha256(plan_path),
        "design_auditor": str(auditor_path.relative_to(PROJECT_ROOT)),
        "design_auditor_sha256": file_sha256(auditor_path),
        "design_audit": str(audit_path.relative_to(PROJECT_ROOT)),
        "design_audit_sha256": file_sha256(audit_path),
        "authorization": {
            "modify_design_prompt_records_model_decoding_or_gates": False,
            "construct_and_seal_corpus": True,
            "implement_and_audit_local_runner": True,
            "run_local_model": False,
            "run_API_model": False,
            "train_adapter": False,
            "collect_human_language": False,
            "perform_real_tool_call_or_external_side_effect": False,
        },
    }
    lock["lock_payload_sha256"] = payload_hash(lock)
    lock_path.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n")
    print(json.dumps(audit, indent=2, sort_keys=True))
    print(json.dumps({"lock": str(lock_path), "sha256": file_sha256(lock_path)}, indent=2))


if __name__ == "__main__":
    main()
