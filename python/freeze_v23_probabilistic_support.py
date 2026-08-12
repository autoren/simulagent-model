"""Freeze the V23 exposed-data probabilistic support replay."""

from __future__ import annotations

import argparse
import hashlib
import json

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT


IMPLEMENTATION = (
    "python/v23_probabilistic_relational.py",
    "python/evaluate_v23_probabilistic_support.py",
    "python/test_v23_probabilistic.py",
    "python/evaluate_v22r2_relational_grounding.py",
    "python/v22_relational.py",
    "python/v22r2_grounding.py",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/v23-probabilistic-support.json")
    parser.add_argument("--plan", default="docs/v23-probabilistic-support-plan.md")
    parser.add_argument("--output", default="configs/v23-probabilistic-support-lock.json")
    args = parser.parse_args()
    config_path = (PROJECT_ROOT / args.config).resolve()
    plan_path = (PROJECT_ROOT / args.plan).resolve()
    output_path = (PROJECT_ROOT / args.output).resolve()
    if output_path.exists():
        raise RuntimeError("V23 protocol lock already exists")
    config = json.loads(config_path.read_text())
    output_dir = PROJECT_ROOT / config["outputDir"]
    attempt_path = output_dir.parent / "v23-probabilistic-support-attempt.json"
    if output_dir.exists() or attempt_path.exists():
        raise RuntimeError("V23 replay artifacts exist before protocol lock")
    source_lock_path = PROJECT_ROOT / config["sourceV22r2aLock"]
    result_path = PROJECT_ROOT / config["sourceV22r2aResult"]
    audit_path = PROJECT_ROOT / config["sourceV22r2aPostAudit"]
    diagnostic_path = PROJECT_ROOT / config["sourceV22r2aDiagnostic"]
    metadata_path = PROJECT_ROOT / config["sourceFeatures"]
    heads_path = PROJECT_ROOT / config["sourceHeads"]
    source_lock = json.loads(source_lock_path.read_text())
    result = json.loads(result_path.read_text())
    audit = json.loads(audit_path.read_text())
    diagnostic = json.loads(diagnostic_path.read_text())
    metadata = json.loads(metadata_path.read_text())
    if not audit["passed"] or audit["decision"] != "accept_v22r2a_negative_result":
        raise RuntimeError("V22r2a integrity audit does not authorize diagnostic continuation")
    if result["decision"] != "develop_probabilistic_support_interface_no_lora":
        raise RuntimeError("V22r2a registered decision differs from V23 scope")
    if diagnostic["new_model_forward_passes"] != 0 or diagnostic["new_linear_fits"] != 0:
        raise RuntimeError("V22r2a diagnostic reports unregistered model work")
    if file_sha256(PROJECT_ROOT / metadata["feature_artifact"]) != metadata["feature_artifact_sha256"]:
        raise RuntimeError("Frozen V22r2 feature artifact changed")
    if file_sha256(heads_path) != result["heads_artifact_sha256"]:
        raise RuntimeError("Frozen V22r2a heads changed")
    lock = {
        "schema_version": 23,
        "experiment": config["experiment"],
        "config": str(config_path.relative_to(PROJECT_ROOT)),
        "config_sha256": file_sha256(config_path),
        "config_payload": config,
        "preregistration": str(plan_path.relative_to(PROJECT_ROOT)),
        "preregistration_sha256": file_sha256(plan_path),
        "branch_budgets": config["branchBudgets"],
        "credible_program_masses": config["credibleProgramMasses"],
        "registered_reference": config["registeredReference"],
        "gates": config["gates"],
        "limits": config["limits"],
        "source": {
            "v22r2a_lock": config["sourceV22r2aLock"],
            "v22r2a_lock_sha256": file_sha256(source_lock_path),
            "v22r2_original_lock": source_lock["source"]["original_lock"],
            "v22r2_original_lock_sha256": source_lock["source"]["original_lock_sha256"],
            "v22r2_result": config["sourceV22r2aResult"],
            "v22r2_result_sha256": file_sha256(result_path),
            "v22r2_post_audit": config["sourceV22r2aPostAudit"],
            "v22r2_post_audit_sha256": file_sha256(audit_path),
            "v22r2_diagnostic": config["sourceV22r2aDiagnostic"],
            "v22r2_diagnostic_sha256": file_sha256(diagnostic_path),
            "feature_metadata": config["sourceFeatures"],
            "feature_metadata_sha256": file_sha256(metadata_path),
            "feature_artifact": metadata["feature_artifact"],
            "feature_artifact_sha256": metadata["feature_artifact_sha256"],
            "heads": config["sourceHeads"],
            "heads_sha256": file_sha256(heads_path),
        },
        "implementation": {
            path: file_sha256(PROJECT_ROOT / path) for path in IMPLEMENTATION
        },
        "data_access_before_lock": {
            "all_v22r2_splits_exposed": True,
            "new_model_forward_passes": 0,
            "new_feature_extractions": 0,
            "new_linear_fits": 0,
            "hyperparameter_selections": 0,
            "adapter_training_runs": 0,
            "probabilistic_replay_runs": 0,
        },
    }
    lock["lock_payload_sha256"] = hashlib.sha256(
        json.dumps(lock, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    output_path.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n")
    print(json.dumps(lock, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
