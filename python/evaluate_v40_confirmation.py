#!/usr/bin/env python3
"""Run the single sealed V40 confirmation against the unchanged V39 compiler."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import time

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from v32_language import compile_truth
from v39_compiler import compile_agent_input


def read(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def rate(values):
    return sum(values) / len(values) if values else 0.0


def score_confirmation(rows, v32):
    predictions = []
    coverage, exact, truth = [], [], []
    by_pack = defaultdict(list)
    by_operation_sign = defaultdict(list)
    for row in rows:
        result = compile_agent_input(row["agent_input"])
        covered = result.get("status") == "ok"
        correct = covered and result.get("parse") == row["target"]["parse"]
        compiled = None
        if covered:
            parsed = result["parse"]
            compiled = compile_truth(parsed["lexical_sign"], parsed["outer_operation"], v32)
        truth_correct = correct and compiled == row["target"]["truth_status"]
        coverage.append(covered)
        exact.append(correct)
        truth.append(truth_correct)
        by_pack[row["ontology_pack"]].append(correct)
        key = f"{row['oracle_metadata']['operation']}|{row['oracle_metadata']['sign']}"
        by_operation_sign[key].append(correct)
        predictions.append({"id": row["id"], "split": row["split"], "result": result, "compiled_truth": compiled})
    pack_rates = {key: rate(values) for key, values in sorted(by_pack.items())}
    cell_rates = {key: rate(values) for key, values in sorted(by_operation_sign.items())}
    metrics = {
        "records": len(rows),
        "coverage": rate(coverage),
        "exact_parse": rate(exact),
        "compiled_truth": rate(truth),
        "ontology_pack_exact_parse": pack_rates,
        "minimum_ontology_pack_exact_parse": min(pack_rates.values()),
        "operation_sign_exact_parse": cell_rates,
        "minimum_operation_sign_exact_parse": min(cell_rates.values()),
    }
    return metrics, predictions


def score_safety(rows):
    predictions = []
    all_safe = []
    by_condition = defaultdict(list)
    for row in rows:
        result = compile_agent_input(row["agent_input"])
        safe = result.get("status") in row["expected"]["statuses"]
        all_safe.append(safe)
        by_condition[row["expected"]["condition"]].append(safe)
        predictions.append({"id": row["id"], "split": row["split"], "result": result, "safe": safe})
    condition_rates = {key: rate(values) for key, values in sorted(by_condition.items())}
    return {
        "records": len(rows),
        "safe_rate": rate(all_safe),
        "condition_safe_rate": condition_rates,
        "minimum_condition_safe_rate": min(condition_rates.values()),
    }, predictions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus-seal", default="configs/v40-corpus-seal.json")
    parser.add_argument("--output-dir", default="outputs/v40-independent-compiler-confirmation/evaluation")
    args = parser.parse_args()
    seal_path = (PROJECT_ROOT / args.corpus_seal).resolve()
    output = (PROJECT_ROOT / args.output_dir).resolve()
    attempt = output.parent / "evaluation-attempt.json"
    if output.exists() or attempt.exists():
        raise RuntimeError("V40 confirmation already attempted")
    seal = json.loads(seal_path.read_text())
    implementation_path = PROJECT_ROOT / seal["implementation_lock"]
    implementation = json.loads(implementation_path.read_text())
    for path, expected in implementation["implementation"].items():
        if file_sha256(PROJECT_ROOT / path) != expected:
            raise RuntimeError(f"V40 implementation changed: {path}")
    if file_sha256(PROJECT_ROOT / implementation["frozen_compiler"]) != implementation["frozen_compiler_sha256"]:
        raise RuntimeError("Frozen V39 compiler changed")
    for artifact in seal["corpora"].values():
        if file_sha256(PROJECT_ROOT / artifact["path"]) != artifact["sha256"]:
            raise RuntimeError(f"V40 sealed corpus changed: {artifact['path']}")
    output.parent.mkdir(parents=True, exist_ok=True)
    attempt.write_text(json.dumps({"schema_version": 40, "status": "started", "confirmation_evaluation": 1, "corpus_seal_sha256": file_sha256(seal_path)}, indent=2, sort_keys=True) + "\n")
    started = time.perf_counter()
    core = read(PROJECT_ROOT / seal["corpora"]["independent_confirmation"]["path"])
    safety_rows = read(PROJECT_ROOT / seal["corpora"]["independent_safety"]["path"])
    confirmation, core_predictions = score_confirmation(core, implementation["v32_config_payload"])
    safety, safety_predictions = score_safety(safety_rows)
    gates = implementation["config_payload"]["gates"]
    checks = {
        "overall_coverage": confirmation["coverage"] >= gates["minimumOverallCoverage"],
        "overall_exact_parse": confirmation["exact_parse"] >= gates["minimumOverallExactParse"],
        "overall_compiled_truth": confirmation["compiled_truth"] >= gates["minimumOverallCompiledTruth"],
        "every_ontology_pack": confirmation["minimum_ontology_pack_exact_parse"] >= gates["minimumEveryOntologyPackExactParse"],
        "every_operation_sign_cell": confirmation["minimum_operation_sign_exact_parse"] >= gates["minimumEveryOperationSignCellExactParse"],
        "overall_safety": safety["safe_rate"] >= gates["minimumSafetyRate"],
        "every_safety_condition": safety["minimum_condition_safe_rate"] >= gates["minimumEverySafetyConditionRate"],
    }
    passed = all(checks.values())
    semantic_pass = all(checks[key] for key in ("overall_coverage", "overall_exact_parse", "overall_compiled_truth", "every_ontology_pack", "every_operation_sign_cell"))
    if passed:
        decision = "accept_declared_language_interface_preregister_relational_mechanic_confirmation"
    elif not semantic_pass:
        decision = "reject_v39_transfer_claim_no_v40_repair_or_repeat"
    else:
        decision = "reject_fail_closed_claim_no_v40_repair_or_repeat"
    output.mkdir(parents=True, exist_ok=False)
    prediction_path = output / "predictions.jsonl"
    prediction_path.write_text("".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in core_predictions + safety_predictions))
    result = {
        "schema_version": 40,
        "experiment": implementation["config_payload"]["experiment"],
        "corpus_seal": str(seal_path.relative_to(PROJECT_ROOT)),
        "corpus_seal_sha256": file_sha256(seal_path),
        "confirmation_evaluation_number": 1,
        "confirmation": confirmation,
        "safety": safety,
        "qualification": {"passed": passed, "checks": checks},
        "decision": decision,
        "predictions": str(prediction_path.relative_to(PROJECT_ROOT)),
        "predictions_sha256": file_sha256(prediction_path),
        "runtime_seconds": time.perf_counter() - started,
        "data_access": {"confirmation_evaluations": 1, "confirmation_records_scored": len(core), "safety_records_scored": len(safety_rows), "selection_on_confirmation": 0, "model_forward_passes": 0, "v32_evaluation_records_read": 0, "v28_runs": 0, "adapter_training_runs": 0},
        "authorization": {"preregister_relational_mechanic_confirmation": passed, "construct_relational_confirmation": False, "expand_to_open_paraphrase": False, "v32_evaluation": False, "v28": False, "adapter_training": False, "change_backbone": False},
    }
    result_path = output / "result.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    state = json.loads(attempt.read_text())
    state.update({"status": "completed", "result_sha256": file_sha256(result_path)})
    attempt.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
