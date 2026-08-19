#!/usr/bin/env python3
from __future__ import annotations

import importlib.metadata as metadata
import json
from pathlib import Path
from typing import Any

from audit_and_freeze_v122_prequery_signal_inventory import payload_hash, valid_lock, write_json
from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT


def main() -> None:
    config_path = PROJECT_ROOT / "configs/v126-sgd-retrieval-selectivity.json"
    plan_path = PROJECT_ROOT / "docs/v126-sgd-retrieval-selectivity-plan.md"
    protocol_path = PROJECT_ROOT / "python/v126_sgd_retrieval_selectivity.py"
    tests_path = PROJECT_ROOT / "python/test_v126_sgd_retrieval_selectivity.py"
    runner_path = PROJECT_ROOT / "python/run_v126_sgd_retrieval_selectivity.py"
    verifier_path = PROJECT_ROOT / "python/verify_and_freeze_v126_sgd_retrieval_selectivity_outcome.py"
    auditor_path = PROJECT_ROOT / "python/audit_and_freeze_v126_sgd_retrieval_selectivity.py"
    audit_path = PROJECT_ROOT / "outputs/v126-sgd-retrieval-selectivity/design-audit.json"
    lock_path = PROJECT_ROOT / "configs/v126-sgd-retrieval-selectivity-lock.json"
    result_path = PROJECT_ROOT / "outputs/v126-sgd-retrieval-selectivity/evaluation/result.json"
    if any(path.exists() for path in (audit_path, lock_path, result_path)):
        raise RuntimeError("V126 already frozen or evaluated")
    config = json.loads(config_path.read_text())
    parent_path = PROJECT_ROOT / config["parentV125OutcomeLock"]
    parent = json.loads(parent_path.read_text())
    parent_lock_path = PROJECT_ROOT / parent["analysis_lock"]
    parent_lock = json.loads(parent_lock_path.read_text())
    archive_path = PROJECT_ROOT / config["sourceArchive"]
    inventory_path = PROJECT_ROOT / config["sourceInventory"]
    catalog_path = PROJECT_ROOT / config["choiceCatalog"]
    populations_path = PROJECT_ROOT / config["selectedPopulations"]
    baseline_path = PROJECT_ROOT / config["baselineConfig"]
    v119_path = PROJECT_ROOT / config["V119Config"]
    v112_config = json.loads((PROJECT_ROOT / "configs/v112-open-world-full-policy-transfer.json").read_text())
    v119_config = json.loads(v119_path.read_text())
    catalog = json.loads(catalog_path.read_text())
    populations = json.loads(populations_path.read_text())
    auth = parent["authorization"]
    checks = {
        "V125_is_valid_and_authorizes_only_separate_selectivity_design": bool(
            valid_lock(parent) and valid_lock(parent_lock)
            and parent["outcome"]["passed"] and parent["outcome"]["audit_pass"]
            and auth["preregister_cross_dataset_retrieval_geometry_selectivity_design"]
            and not auth["extract_selected_language_or_evaluate_before_next_lock"]
            and not auth["run_language_model_or_open_protected"]
            and not auth["run_API_training_action_or_execution"]
        ),
        "catalog_populations_and_source_are_exact": bool(
            file_sha256(catalog_path) == parent["choice_catalog_sha256"]
            and file_sha256(populations_path) == parent["selected_populations_sha256"]
            and catalog["choice_count"] == 11
            and populations["training_record_count"] == config["extraction"]["expectedTrainingRecordCount"] == 4881
            and populations["evaluation_record_count"] == config["extraction"]["expectedEvaluationRecordCount"] == 576
            and archive_path.is_file() and inventory_path.is_file()
        ),
        "retrieval_trigger_and_channel_are_frozen_without_fit_or_selection": bool(
            config["frozenRetrieval"]["knownThreshold"] == v112_config["fixedRetrievalThresholds"]["known"] == 0.8
            and config["frozenRetrieval"]["unsupportedThreshold"] == v112_config["fixedRetrievalThresholds"]["unsupported"] == 0.3
            and config["frozenRetrieval"]["fitCount"] == 0
            and config["primaryTrigger"]["candidateCount"] == 1
            and config["primaryTrigger"]["selectionCount"] == 0
            and config["queryChannel"]["marginalCorrectness"] == 0.95
            and config["queryChannel"]["totalCost"] == v119_config["adaptiveTree"]["totalCostEveryPath"] == 0.3
            and config["queryChannel"]["sharedFailureCorrelations"] == [0.0, 0.25, 0.5]
        ),
        "language_model_authority_and_execution_boundaries_are_closed": bool(
            not config["extraction"]["persistSelectedLanguage"]
            and config["extraction"]["maximumIndividualLanguageRecordEmissionCount"] == 0
            and config["accessGates"]["maximumPersistedSelectedLanguageRecordCount"] == 0
            and config["accessGates"]["maximumManualLanguageOrRawResponseInspectionCount"] == 0
            and config["accessGates"]["maximumModelLoadCount"] == 0
            and config["accessGates"]["maximumModelGenerationCount"] == 0
            and config["authorityBoundary"]["completeSafeCompositeHypothesisUniverseAlwaysRetained"]
            and config["authorityBoundary"]["actualExecutionCount"] == 0
            and not config["decisionRule"]["passAuthorizesThresholdFitAlternativeTriggerOrModelRun"]
            and not config["decisionRule"]["passAuthorizesProtectedInductionAPITrainingActionOrExecution"]
        ),
        "code_runtime_and_output_absence_hold": bool(
            all(path.is_file() for path in (plan_path, protocol_path, tests_path, runner_path, verifier_path, auditor_path, baseline_path, v119_path))
            and metadata.version("numpy") == "2.5.1" and metadata.version("scikit-learn") == "1.9.0"
            and not result_path.exists()
        ),
    }
    passed = all(checks.values())
    audit = {"schema_version": "126-sgd-retrieval-selectivity-design-audit", "experiment": config["experiment"], "passed": passed, "checks": checks, "decision": "freeze_and_authorize_one_cross_dataset_model_free_run" if passed else "reject_V126_design", "prelock_access": {"source_archive_read_count": 0, "selected_language_parse_count": 0, "manual_language_inspection_count": 0, "model_load_count": 0, "model_generation_count": 0, "actual_execution_count": 0}}
    write_json(audit_path, audit)
    if not passed:
        print(json.dumps(audit, indent=2, sort_keys=True)); raise SystemExit(1)
    deps = {"config": config_path, "parent_outcome": parent_path, "parent_analysis_lock": parent_lock_path, "source_archive": archive_path, "source_inventory": inventory_path, "choice_catalog": catalog_path, "selected_populations": populations_path, "baseline_config": baseline_path, "V119_config": v119_path, "plan": plan_path, "protocol": protocol_path, "tests": tests_path, "runner": runner_path, "verifier": verifier_path, "auditor": auditor_path, "design_audit": audit_path}
    lock: dict[str, Any] = {"schema_version": "126-sgd-retrieval-selectivity-lock", "experiment": config["experiment"], "config_payload": config, "authorization": {"run_one_cross_dataset_model_free_retrieval_selectivity_evaluation": True, "modify_extraction_retrieval_trigger_channel_cost_gates_or_decision": False, "persist_or_manually_inspect_language_or_individual_records": False, "fit_or_select_trigger_or_load_model": False, "grant_protected_induction_authority_or_execution": False}}
    for key, path in deps.items():
        lock[key] = str(path.relative_to(PROJECT_ROOT)); lock[f"{key}_sha256"] = file_sha256(path)
    lock["lock_payload_sha256"] = payload_hash(lock); write_json(lock_path, lock)
    print(json.dumps(audit, indent=2, sort_keys=True)); print(json.dumps({"lock": str(lock_path.relative_to(PROJECT_ROOT)), "sha256": file_sha256(lock_path)}, indent=2))


if __name__ == "__main__": main()
