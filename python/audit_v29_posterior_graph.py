"""Pre-evaluation integrity audit for V29 posterior graph decoding."""

from __future__ import annotations

import argparse
import json

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/v29-posterior-graph.json")
    parser.add_argument("--output", default="outputs/v29-posterior-graph/pre-evaluation-audit.json")
    args = parser.parse_args()
    config_path = (PROJECT_ROOT / args.config).resolve()
    config = json.loads(config_path.read_text())
    errors = []
    source_paths = (
        "sourceV28Lock", "sourceV28Result", "sourceV28PostAudit",
        "sourceV28Predictions", "sourceV28Diagnostics",
    )
    for key in source_paths:
        if not (PROJECT_ROOT / config[key]).is_file():
            errors.append(f"Missing source artifact: {key}")
    v28_lock_path = PROJECT_ROOT / config["sourceV28Lock"]
    v28_result_path = PROJECT_ROOT / config["sourceV28Result"]
    v28_audit_path = PROJECT_ROOT / config["sourceV28PostAudit"]
    v28_result = json.loads(v28_result_path.read_text())
    v28_audit = json.loads(v28_audit_path.read_text())
    if not v28_audit["passed"] or v28_audit["decision"] != "accept_v28_exposed_development_result":
        errors.append("V28 integrity audit does not authorize reuse")
    if v28_result["decision"] != "marginal_program_map_improves_support_continue_query_repair_no_lora":
        errors.append("V28 decision differs from the registered V29 premise")
    if v28_result["protocol_lock_sha256"] != file_sha256(v28_lock_path):
        errors.append("V28 result and lock differ")
    limits = config["limits"]
    expected_limits = {
        "newModelForwardPasses": 0, "posteriorGraphEvaluations": 1,
        "headFits": 0, "thresholdFits": 0, "hyperparameterSelections": 0,
        "adapterTrainingRuns": 0, "freshBenchmarkRecords": 0,
    }
    if limits != expected_limits:
        errors.append("V29 access limits differ from the registered zero-fit one-shot design")
    if config["posteriorGraph"]["credibleUnionPermitted"]:
        errors.append("V29 must emit one graph, not a credible union")
    result = {
        "schema_version": 29,
        "experiment": "v29_pre_evaluation_integrity_audit",
        "passed": not errors,
        "decision": "authorize_v29_protocol_lock" if not errors else "block_v29_protocol_lock",
        "errors": errors,
        "source_v28": {
            "evaluation_exact_support_graph": v28_result["grounding"]["by_split"]["grounding_evaluation"]["exact_support_graph"],
            "evaluation_target_program_selection": v28_result["marginal_search"]["evaluation_target_program_selection_rate"],
            "frozen_support_oracle_query_exact": v28_result["integration"]["frozen_support_oracle_query"]["transition_set_exact_match"],
        },
        "firewall": {
            "all_v22r2_splits_exposed": True,
            "new_model_forward_passes_before_lock": 0,
            "posterior_graph_evaluations_before_lock": 0,
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
