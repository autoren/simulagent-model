"""Freeze the V29 posterior-marginal support graph protocol."""

from __future__ import annotations

import argparse
import hashlib
import json

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT


IMPLEMENTATION = (
    "python/audit_v29_posterior_graph.py",
    "python/v29_posterior_graph.py",
    "python/evaluate_v29_posterior_graph.py",
    "python/test_v29_posterior_graph.py",
    "python/freeze_v29_posterior_graph.py",
    "python/v28_marginal_map.py",
    "python/v27_support_map.py",
    "python/evaluate_v22r2_relational_grounding.py",
    "python/v22_relational.py",
    "python/v22r2_grounding.py",
    "python/v23_probabilistic_relational.py",
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/v29-posterior-graph.json")
    parser.add_argument("--plan", default="docs/v29-posterior-graph-plan.md")
    parser.add_argument("--audit", default="outputs/v29-posterior-graph/pre-evaluation-audit.json")
    parser.add_argument("--output", default="configs/v29-posterior-graph-lock.json")
    args = parser.parse_args()
    config_path = (PROJECT_ROOT / args.config).resolve()
    plan_path = (PROJECT_ROOT / args.plan).resolve()
    audit_path = (PROJECT_ROOT / args.audit).resolve()
    output_path = (PROJECT_ROOT / args.output).resolve()
    if output_path.exists():
        raise RuntimeError("V29 protocol lock already exists")
    output_root = PROJECT_ROOT / "outputs/v29-posterior-graph"
    for path in (output_root / "evaluation", output_root / "evaluation-attempt.json"):
        if path.exists():
            raise RuntimeError(f"V29 evaluation artifact exists before lock: {path}")
    config = json.loads(config_path.read_text())
    audit = json.loads(audit_path.read_text())
    if not audit["passed"] or audit["decision"] != "authorize_v29_protocol_lock":
        raise RuntimeError("V29 pre-evaluation audit does not authorize lock")
    if audit["config_sha256"] != file_sha256(config_path):
        raise RuntimeError("V29 config changed after its audit")
    v28_result = json.loads((PROJECT_ROOT / config["sourceV28Result"]).read_text())
    source_keys = (
        "sourceV28Lock", "sourceV28Result", "sourceV28PostAudit",
        "sourceV28Predictions", "sourceV28Diagnostics",
    )
    lock = {
        "schema_version": 29,
        "experiment": config["experiment"],
        "config": str(config_path.relative_to(PROJECT_ROOT)),
        "config_sha256": file_sha256(config_path),
        "config_payload": config,
        "preregistration": str(plan_path.relative_to(PROJECT_ROOT)),
        "preregistration_sha256": file_sha256(plan_path),
        "pre_evaluation_audit": str(audit_path.relative_to(PROJECT_ROOT)),
        "pre_evaluation_audit_sha256": file_sha256(audit_path),
        "posterior_graph": config["posteriorGraph"],
        "integration_conditions": config["integrationConditions"],
        "gates": config["gates"],
        "limits": config["limits"],
        "source": {
            key: config[key] for key in source_keys
        } | {
            f"{key}_sha256": file_sha256(PROJECT_ROOT / config[key])
            for key in source_keys
        },
        "source_v28_reference": {
            "evaluation_support_exact_graph": v28_result["grounding"]["by_split"]["grounding_evaluation"]["exact_support_graph"],
            "frozen_support_oracle_query_exact": v28_result["integration"]["frozen_support_oracle_query"]["transition_set_exact_match"],
            "frozen_frozen_exact": v28_result["integration"]["frozen_support_frozen_query"]["transition_set_exact_match"],
            "target_retention": v28_result["integration"]["frozen_support_oracle_query"]["target_retention_rate"],
            "empty_version_space": v28_result["integration"]["frozen_support_oracle_query"]["empty_version_space_rate"],
        },
        "implementation": {
            path: file_sha256(PROJECT_ROOT / path) for path in IMPLEMENTATION
        },
        "data_access_before_lock": {
            "all_v22r2_splits_exposed": True, "new_model_forward_passes": 0,
            "posterior_graph_evaluations": 0, "head_fits": 0,
            "threshold_fits": 0, "hyperparameter_selections": 0,
            "adapter_training_runs": 0, "fresh_benchmark_records": 0,
        },
    }
    lock["lock_payload_sha256"] = hashlib.sha256(
        json.dumps(lock, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    output_path.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n")
    print(json.dumps(lock, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
