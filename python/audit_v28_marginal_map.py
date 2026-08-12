"""Pre-evaluation integrity audit for V28 marginal program MAP."""

from __future__ import annotations

import argparse
import json

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/v28-marginal-map.json")
    parser.add_argument("--output", default="outputs/v28-marginal-map/pre-evaluation-audit.json")
    args = parser.parse_args()
    config_path = (PROJECT_ROOT / args.config).resolve()
    config = json.loads(config_path.read_text())
    errors = []
    v27_lock_path = PROJECT_ROOT / config["sourceV27Lock"]
    v27_result_path = PROJECT_ROOT / config["sourceV27Result"]
    v27_audit_path = PROJECT_ROOT / config["sourceV27PostAudit"]
    v27_lock = json.loads(v27_lock_path.read_text())
    v27_result = json.loads(v27_result_path.read_text())
    v27_audit = json.loads(v27_audit_path.read_text())
    v27_diagnostics = [
        json.loads(line)
        for line in (PROJECT_ROOT / config["sourceV27Diagnostics"]).read_text().splitlines()
        if line.strip()
    ]
    evaluation_diagnostics = [
        row for row in v27_diagnostics if row["split"] == "grounding_evaluation"
    ]
    if not v27_audit["passed"] or v27_audit["decision"] != "accept_v27_exposed_development_result":
        errors.append("V27 integrity audit does not authorize reuse")
    if v27_result["decision"] != "support_map_improves_execution_continue_match_repair_no_lora":
        errors.append("V27 decision differs from the registered V28 premise")
    if v27_result["protocol_lock_sha256"] != file_sha256(v27_lock_path):
        errors.append("V27 result and lock differ")
    limits = config["limits"]
    expected_limits = {
        "newModelForwardPasses": 0, "marginalMapEvaluations": 1, "headFits": 0,
        "thresholdFits": 0, "hyperparameterSelections": 0, "adapterTrainingRuns": 0,
        "freshBenchmarkRecords": 0,
    }
    if limits != expected_limits:
        errors.append("V28 access limits differ from the registered zero-fit one-shot design")
    if config["marginalMap"]["credibleUnionPermitted"]:
        errors.append("V28 must not widen predictions with a credible union")
    source_paths = [
        "sourceV27Result", "sourceV27PostAudit", "sourceV27EdgeMetadata",
        "sourceV27EdgeScores", "sourceV27Predictions", "sourceV27Diagnostics",
        "sourceV27NativeMatchDiagnostic",
    ]
    for key in source_paths:
        if not (PROJECT_ROOT / config[key]).is_file():
            errors.append(f"Missing source artifact: {key}")
    result = {
        "schema_version": 28,
        "experiment": "v28_pre_evaluation_integrity_audit",
        "passed": not errors,
        "decision": "authorize_v28_protocol_lock" if not errors else "block_v28_protocol_lock",
        "errors": errors,
        "source_v27": {
            "support_scenes": v27_result["graph_search"]["support_scenes"],
            "mean_graph_branches": v27_result["graph_search"]["mean_graph_branches"],
            "target_graph_branch_coverage": v27_result["graph_search"]["target_graph_in_branch_rate"],
            "evaluation_target_program_selection": sum(
                row["target_program_selected"] for row in evaluation_diagnostics
            ) / len(evaluation_diagnostics),
            "new_model_forward_passes_available": 0,
        },
        "firewall": {
            "all_v22r2_splits_exposed": True,
            "new_model_forward_passes_before_lock": 0,
            "marginal_map_evaluations_before_lock": 0,
            "new_fit_or_threshold_before_lock": 0,
            "fresh_benchmark_records_created": 0,
        },
        "config_sha256": file_sha256(config_path),
    }
    output_path = (PROJECT_ROOT / args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
