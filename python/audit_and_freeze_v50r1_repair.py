#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from decimal import Decimal

from evaluate_v50r1_history import effective_count_decimal
from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/v50r1-execution-repair.json")
    parser.add_argument("--plan", default="docs/v50r1-execution-repair-plan.md")
    parser.add_argument("--audit", default="outputs/v50r1-execution-repair/repair-audit.json")
    parser.add_argument("--lock", default="configs/v50r1-repair-lock.json")
    args = parser.parse_args()
    config_path, plan_path, audit_path, lock_path = tuple(
        (PROJECT_ROOT / value).resolve() for value in (args.config, args.plan, args.audit, args.lock)
    )
    if lock_path.exists():
        raise RuntimeError("V50r1 repair already frozen")
    config = json.loads(config_path.read_text())
    seal_path = PROJECT_ROOT / config["sourceCorpusSeal"]
    implementation_path = PROJECT_ROOT / config["sourceImplementationLock"]
    attempt_path = PROJECT_ROOT / config["failedAttempt"]
    seal = json.loads(seal_path.read_text())
    implementation = json.loads(implementation_path.read_text())
    attempt = json.loads(attempt_path.read_text())
    original_result = PROJECT_ROOT / "outputs/v50-history-dependent-belief-filtering/development/result.json"
    original_metrics = PROJECT_ROOT / "outputs/v50-history-dependent-belief-filtering/development/mechanic-metrics.jsonl"
    errors = []
    if seal["implementation_lock_sha256"] != file_sha256(implementation_path):
        errors.append("V50 source seal is not bound to source implementation")
    if attempt.get("status") != "started" or attempt.get("development_run") != 1:
        errors.append("V50 failed-attempt state is not preserved")
    if original_result.exists() or original_metrics.exists():
        errors.append("V50 scientific outputs exist despite the documented pre-result failure")
    if config["scientificOutcomeAccessed"] or config["mechanicMetricsMaterialized"] or config["resultMaterialized"]:
        errors.append("V50r1 repair incorrectly claims scientific outcome access")
    permitted = config["permittedChange"]
    if any(permitted[key] for key in (
        "predictionOrPosteriorSemanticsChanged", "populationChanged", "metricsChanged", "gatesChanged",
        "heldoutOutcomesChanged",
    )):
        errors.append("V50r1 repair exceeds the diagnostic-only boundary")
    tiny = Decimal("1e-500")
    repaired_count = effective_count_decimal([Decimal(1) - tiny, tiny])
    numerical_repair = 1.0 <= repaired_count <= 2.0
    if not numerical_repair:
        errors.append("V50r1 Decimal effective-count repair is invalid")
    if (PROJECT_ROOT / "outputs/v50r1-execution-repair/development").exists():
        errors.append("V50r1 downstream result exists before repair lock")

    audit = {
        "schema_version": "50r1",
        "experiment": "v50r1_repair_audit",
        "passed": not errors,
        "decision": "authorize_v50r1_repair_lock" if not errors else "reject_v50r1_repair",
        "errors": errors,
        "checks": {
            "source_corpus_sealed": seal["authorization"]["run_development_once"],
            "failed_attempt_preserved": attempt.get("status") == "started",
            "no_scientific_result_materialized": not original_result.exists() and not original_metrics.exists(),
            "diagnostic_only_change": not any(permitted[key] for key in (
                "predictionOrPosteriorSemanticsChanged", "populationChanged", "metricsChanged", "gatesChanged",
                "heldoutOutcomesChanged",
            )),
            "decimal_underflow_repaired": numerical_repair,
            "repair_downstream_absent": not (PROJECT_ROOT / "outputs/v50r1-execution-repair/development").exists(),
        },
        "data_access": {
            "failed_execution_attempts": 1,
            "completed_development_runs": 0,
            "mechanic_metrics_accessed": 0,
            "model_forward_passes": 0,
            "adapter_training_runs": 0,
        },
    }
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    if errors:
        print(json.dumps(audit, indent=2, sort_keys=True))
        raise SystemExit(1)
    repair_implementation = {
        path: file_sha256(PROJECT_ROOT / path)
        for path in (
            "python/evaluate_v50r1_history.py",
            "python/freeze_v50r1_outcome.py",
        )
    }
    lock = {
        "schema_version": "50r1",
        "experiment": "v50r1_repair_lock",
        "config": str(config_path.relative_to(PROJECT_ROOT)),
        "config_sha256": file_sha256(config_path),
        "plan": str(plan_path.relative_to(PROJECT_ROOT)),
        "plan_sha256": file_sha256(plan_path),
        "repair_audit": str(audit_path.relative_to(PROJECT_ROOT)),
        "repair_audit_sha256": file_sha256(audit_path),
        "source_corpus_seal": str(seal_path.relative_to(PROJECT_ROOT)),
        "source_corpus_seal_sha256": file_sha256(seal_path),
        "source_implementation_lock": str(implementation_path.relative_to(PROJECT_ROOT)),
        "source_implementation_lock_sha256": file_sha256(implementation_path),
        "failed_attempt": str(attempt_path.relative_to(PROJECT_ROOT)),
        "failed_attempt_sha256": file_sha256(attempt_path),
        "repair_implementation": repair_implementation,
        "authorization": {
            "run_repair_development_once": True,
            "model_access": False,
            "adapter_training": False,
            "final_evaluation": False,
        },
    }
    lock["lock_payload_sha256"] = hashlib.sha256(
        json.dumps(lock, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    lock_path.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"audit": audit, "lock": lock}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
