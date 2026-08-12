"""Freeze the V28 marginal program MAP protocol."""

from __future__ import annotations

import argparse
import hashlib
import json

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT


IMPLEMENTATION = (
    "python/audit_v28_marginal_map.py",
    "python/v28_marginal_map.py",
    "python/evaluate_v28_marginal_map.py",
    "python/test_v28_marginal_map.py",
    "python/freeze_v28_marginal_map.py",
    "python/v27_support_map.py",
    "python/evaluate_v22r2_relational_grounding.py",
    "python/v22_relational.py",
    "python/v22r2_grounding.py",
    "python/v23_probabilistic_relational.py",
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/v28-marginal-map.json")
    parser.add_argument("--plan", default="docs/v28-marginal-map-plan.md")
    parser.add_argument("--audit", default="outputs/v28-marginal-map/pre-evaluation-audit.json")
    parser.add_argument("--output", default="configs/v28-marginal-map-lock.json")
    args = parser.parse_args()
    config_path = (PROJECT_ROOT / args.config).resolve()
    plan_path = (PROJECT_ROOT / args.plan).resolve()
    audit_path = (PROJECT_ROOT / args.audit).resolve()
    output_path = (PROJECT_ROOT / args.output).resolve()
    if output_path.exists():
        raise RuntimeError("V28 protocol lock already exists")
    output_root = PROJECT_ROOT / "outputs/v28-marginal-map"
    for path in (output_root / "evaluation", output_root / "evaluation-attempt.json"):
        if path.exists():
            raise RuntimeError(f"V28 evaluation artifact exists before lock: {path}")
    config = json.loads(config_path.read_text())
    audit = json.loads(audit_path.read_text())
    if not audit["passed"] or audit["decision"] != "authorize_v28_protocol_lock":
        raise RuntimeError("V28 pre-evaluation audit does not authorize lock")
    if audit["config_sha256"] != file_sha256(config_path):
        raise RuntimeError("V28 config changed after its audit")
    v27_lock_path = PROJECT_ROOT / config["sourceV27Lock"]
    v27_result_path = PROJECT_ROOT / config["sourceV27Result"]
    v27_result = json.loads(v27_result_path.read_text())
    source_keys = (
        "sourceV27Lock", "sourceV27Result", "sourceV27PostAudit",
        "sourceV27EdgeMetadata", "sourceV27EdgeScores", "sourceV27Predictions",
        "sourceV27Diagnostics", "sourceV27NativeMatchDiagnostic",
    )
    lock = {
        "schema_version": 28,
        "experiment": config["experiment"],
        "config": str(config_path.relative_to(PROJECT_ROOT)),
        "config_sha256": file_sha256(config_path),
        "config_payload": config,
        "preregistration": str(plan_path.relative_to(PROJECT_ROOT)),
        "preregistration_sha256": file_sha256(plan_path),
        "pre_evaluation_audit": str(audit_path.relative_to(PROJECT_ROOT)),
        "pre_evaluation_audit_sha256": file_sha256(audit_path),
        "marginal_map": config["marginalMap"],
        "integration_conditions": config["integrationConditions"],
        "gates": config["gates"],
        "limits": config["limits"],
        "source": {
            key: config[key] for key in source_keys
        } | {
            f"{key}_sha256": file_sha256(PROJECT_ROOT / config[key])
            for key in source_keys
        },
        "source_v27_reference": {
            "evaluation_support_exact_graph": v27_result["grounding"]["by_split"]["grounding_evaluation"]["exact_support_graph"],
            "frozen_support_oracle_query_exact": v27_result["integration"]["frozen_support_oracle_query"]["transition_set_exact_match"],
            "frozen_frozen_exact": v27_result["integration"]["frozen_support_frozen_query"]["transition_set_exact_match"],
            "target_retention": v27_result["integration"]["frozen_support_oracle_query"]["target_retention_rate"],
            "empty_version_space": v27_result["integration"]["frozen_support_oracle_query"]["empty_version_space_rate"],
        },
        "v27_source_lock_sha256": file_sha256(v27_lock_path),
        "implementation": {
            path: file_sha256(PROJECT_ROOT / path) for path in IMPLEMENTATION
        },
        "data_access_before_lock": {
            "all_v22r2_splits_exposed": True,
            "new_model_forward_passes": 0,
            "marginal_map_evaluations": 0,
            "head_fits": 0,
            "threshold_fits": 0,
            "hyperparameter_selections": 0,
            "adapter_training_runs": 0,
            "fresh_benchmark_records": 0,
        },
    }
    lock["lock_payload_sha256"] = hashlib.sha256(
        json.dumps(lock, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    output_path.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n")
    print(json.dumps(lock, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
