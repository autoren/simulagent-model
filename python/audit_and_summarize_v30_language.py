#!/usr/bin/env python3
"""Reproduce, audit, and summarize the one-shot V30 language result."""

from __future__ import annotations

import argparse
import json

from audit_v30_signed_fact_language import read_rows
from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from v30_evaluation import lora_eligibility, primary_summary, truth_summary


def jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", default="configs/v30-signed-fact-language-lock.json")
    parser.add_argument("--result", default="outputs/v30-signed-fact-language/evaluation/result.json")
    parser.add_argument("--audit", default="outputs/v30-signed-fact-language/post-result-audit.json")
    parser.add_argument("--markdown", default="docs/v30-results.md")
    args = parser.parse_args()
    lock_path = (PROJECT_ROOT / args.lock).resolve()
    result_path = (PROJECT_ROOT / args.result).resolve()
    lock = json.loads(lock_path.read_text())
    result = json.loads(result_path.read_text())
    config = lock["config_payload"]
    rows = read_rows(PROJECT_ROOT / lock["source"]["corpus"], tuple(config["splits"]))
    primary_path = PROJECT_ROOT / result["primary_predictions"]
    baseline_path = PROJECT_ROOT / result["v26_baseline_predictions"]
    primary = primary_summary(rows, jsonl(primary_path), config)
    baseline = truth_summary(rows, jsonl(baseline_path), config)
    diagnostic = None
    diagnostic_path = None
    if result["candidate_nli_predictions"] is not None:
        diagnostic_path = PROJECT_ROOT / result["candidate_nli_predictions"]
        diagnostic = truth_summary(rows, jsonl(diagnostic_path), config)
    pre_audit = json.loads((PROJECT_ROOT / lock["source"]["pre_model_audit"]).read_text())
    eligibility = lora_eligibility(primary, diagnostic, pre_audit["passed"], config)
    expected_decision = (
        "signed_fact_pass_authorize_one_v28_reintegration" if primary["passed"]
        else "frozen_signed_fact_methods_fail_lora_eligible_separate_protocol_required"
        if eligibility["eligible"]
        else "frozen_signed_fact_insufficient_repair_interface_no_lora"
    )
    attempt_path = PROJECT_ROOT / "outputs/v30-signed-fact-language/evaluation-attempt.json"
    attempt = json.loads(attempt_path.read_text())
    expected_forwards = len(rows) * (6 if diagnostic is not None else 5)
    checks = {
        "lock_matches": result["protocol_lock_sha256"] == file_sha256(lock_path),
        "pre_model_audit_intact": (
            file_sha256(PROJECT_ROOT / lock["source"]["pre_model_audit"])
            == lock["source"]["pre_model_audit_sha256"]
        ),
        "attempt_completed_once": (
            attempt["attempt_number"] == 1 and attempt["status"] == "completed"
            and attempt["result_sha256"] == file_sha256(result_path)
        ),
        "primary_predictions_intact": result["primary_predictions_sha256"] == file_sha256(primary_path),
        "baseline_predictions_intact": result["v26_baseline_predictions_sha256"] == file_sha256(baseline_path),
        "diagnostic_predictions_intact": (
            diagnostic_path is None and result["candidate_nli_predictions_sha256"] is None
        ) or (
            diagnostic_path is not None
            and result["candidate_nli_predictions_sha256"] == file_sha256(diagnostic_path)
        ),
        "primary_metrics_reproduced": result["primary"] == primary,
        "baseline_metrics_reproduced": result["v26_baseline"] == baseline,
        "diagnostic_metrics_reproduced": result["candidate_nli_diagnostic"] == diagnostic,
        "conditional_diagnostic_rule": result["candidate_nli_triggered"] == (not primary["passed"]),
        "lora_eligibility_reproduced": result["lora_eligibility"] == eligibility,
        "pass_reproduced": result["passed"] == primary["passed"],
        "decision_reproduced": result["decision"] == expected_decision,
        "integration_authorization_reproduced": result["v28_integration_authorized"] == primary["passed"],
        "one_shot_zero_fit": (
            result["data_access"]["model_forward_passes"] == expected_forwards
            and result["data_access"]["primary_evaluations"] == 1
            and result["data_access"]["v26_baseline_evaluations"] == 1
            and result["data_access"]["candidate_nli_diagnostic_evaluations"] == int(diagnostic is not None)
            and result["data_access"]["head_fits"] == 0
            and result["data_access"]["threshold_fits"] == 0
            and result["data_access"]["hyperparameter_selections"] == 0
            and result["data_access"]["adapter_training_runs"] == 0
            and result["data_access"]["v28_integration_replays"] == 0
        ),
        "adapter_training_not_authorized": not result["adapter_training_authorized"],
    }
    audit = {
        "schema_version": 30,
        "experiment": "v30_language_post_result_integrity_audit",
        "passed": all(checks.values()),
        "decision": "accept_v30_language_result" if all(checks.values()) else "quarantine_v30_language_result",
        "checks": checks,
        "result": str(result_path.relative_to(PROJECT_ROOT)),
        "result_sha256": file_sha256(result_path),
        "reproduced_primary_checks": primary["checks"],
        "v28_integration_authorized": primary["passed"],
        "lora_eligible_for_separate_protocol": eligibility["eligible"],
    }
    audit_path = (PROJECT_ROOT / args.audit).resolve()
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")

    evaluation = primary["by_split"]["language_evaluation"]
    baseline_eval = baseline["by_split"]["language_evaluation"]
    lines = [
        "# V30 results: sealed canonical signed-fact language interface", "",
        f"Decision: `{result['decision']}`.", "",
        "V30 is a fresh language-only development study. The primary frozen extractor sees the",
        "declared ontology, entity inventory, and clause, but no candidate fact. Fit and calibration",
        "are report-only; the sealed surface-family evaluation was opened once after hash lock.", "",
        "## Sealed evaluation", "",
        "| Metric | Primary | Gate |", "|---|---:|---:|",
        f"| Predicate accuracy | {evaluation['predicate_accuracy']:.3f} | {config['gates']['languageEvaluation']['minimumPredicateAccuracy']:.3f} |",
        f"| Argument-1 accuracy | {evaluation['argument_1_accuracy']:.3f} | {config['gates']['languageEvaluation']['minimumArgument1Accuracy']:.3f} |",
        f"| Relation-order accuracy | {evaluation['relation_argument_order_accuracy']:.3f} | {config['gates']['languageEvaluation']['minimumRelationArgumentOrderAccuracy']:.3f} |",
        f"| Truth-status accuracy | {evaluation['truth_status_accuracy']:.3f} | {config['gates']['languageEvaluation']['minimumTruthStatusAccuracy']:.3f} |",
        f"| Exact signed fact | {evaluation['exact_signed_fact_accuracy']:.3f} | {config['gates']['languageEvaluation']['minimumExactSignedFactAccuracy']:.3f} |",
        f"| Exact scene | {evaluation['exact_scene_accuracy']:.3f} | {config['gates']['languageEvaluation']['minimumExactSceneAccuracy']:.3f} |", "",
        f"The oracle-atom V26 baseline truth accuracy is {baseline_eval['truth_status_accuracy']:.3f}; it is not credited with predicate or argument discovery.",
    ]
    if diagnostic is not None:
        nli_eval = diagnostic["by_split"]["language_evaluation"]
        lines.extend([
            f"The preregistered conditional oracle-atom NLI diagnostic was triggered and reached {nli_eval['truth_status_accuracy']:.3f} truth accuracy with {nli_eval['truth_target_retention_top2']:.3f} top-two retention.",
        ])
    else:
        lines.append("The conditional candidate-NLI diagnostic was not triggered because every primary gate passed.")
    lines.extend([
        "", "## Decision integrity", "",
        f"Primary gates passed: `{str(primary['passed']).lower()}`.",
        f"One V28 reintegration replay authorized: `{str(primary['passed']).lower()}`.",
        f"LoRA eligible for a separate preregistration: `{str(eligibility['eligible']).lower()}`.",
        "Adapter training performed: `false`.",
        f"Post-result integrity audit: `{'pass' if audit['passed'] else 'fail'}`.", "",
    ])
    (PROJECT_ROOT / args.markdown).write_text("\n".join(lines))
    print(json.dumps(audit, indent=2, sort_keys=True))
    if not audit["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
