#!/usr/bin/env python3
import json
from typing import Any

from audit_and_freeze_v122_prequery_signal_inventory import payload_hash
from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from v133_sgd_capability_label_identifiability import run_audit


def main() -> None:
    lock_path = PROJECT_ROOT / "configs/v133-sgd-capability-label-identifiability-lock.json"
    result_path = PROJECT_ROOT / "outputs/v133-sgd-capability-label-identifiability/evaluation/result.json"
    doc_path = PROJECT_ROOT / "docs/v133-sgd-capability-label-identifiability-results.md"
    audit_path = PROJECT_ROOT / "outputs/v133-sgd-capability-label-identifiability/outcome-audit.json"
    outcome_path = PROJECT_ROOT / "configs/v133-sgd-capability-label-identifiability-outcome-lock.json"
    verifier_path = PROJECT_ROOT / "python/verify_and_freeze_v133_sgd_capability_label_identifiability_outcome.py"
    if audit_path.exists() or outcome_path.exists(): raise RuntimeError("V133 outcome already frozen")
    if not doc_path.is_file(): raise RuntimeError("write V133 result document first")
    lock = json.loads(lock_path.read_text()); result = json.loads(result_path.read_text())
    archive_bytes = (PROJECT_ROOT / lock["source_archive"]).read_bytes(); catalog = json.loads((PROJECT_ROOT / lock["choice_catalog"]).read_text()); population = json.loads((PROJECT_ROOT / lock["fixture_population"]).read_text())
    expected = run_audit(archive_bytes, catalog, population, lock["config_payload"])
    dependencies = [key for key in lock if not key.endswith("_sha256") and f"{key}_sha256" in lock]
    checks = {"lock_and_dependencies_exact": bool(payload_hash({key: value for key, value in lock.items() if key != "lock_payload_sha256"}) == lock["lock_payload_sha256"] and all(file_sha256(PROJECT_ROOT / lock[key]) == lock[f"{key}_sha256"] for key in dependencies)), "independent_schema_recomputation_exact": all(result[key] == expected[key] for key in expected), "access_pass": result["access_pass"], "decision_exact": result["decision"] == expected["decision"], "zero_language_model_execution": result["summary"]["utterance_or_slot_value_read_count"] == result["summary"]["model_load_count"] == result["summary"]["model_generation_count"] == result["summary"]["actual_execution_count"] == 0}
    passed = all(checks.values()); audit = {"schema_version": "133-sgd-capability-label-identifiability-outcome-audit", "experiment": lock["config_payload"]["experiment"], "passed": passed, "checks": checks, "recomputed": expected}
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    if not passed: raise SystemExit(1)
    positive = expected["identifiability_pass"]
    paths = {"analysis_lock": lock_path, "result": result_path, "verifier": verifier_path, "audit": audit_path, "results_document": doc_path}
    outcome: dict[str, Any] = {"schema_version": "133-sgd-capability-label-identifiability-outcome-lock", "experiment": "v133_sgd_capability_label_identifiability_outcome_lock", "outcome": {"passed": True, "audit_pass": True, "identifiability_pass": positive, "decision": expected["decision"], "summary": expected}, "authorization": {"modify_rerun_redefine_or_mine_V133": False, "preregister_text_free_semantically_noncolliding_source_design": not positive, "preregister_independent_source_confirmation": positive, "rerun_model_revise_prompt_or_scale": False, "open_protected_or_begin_induction_or_richer_planning": False, "run_API_training_action_or_execution": False}}
    for key, path in paths.items(): outcome[key] = str(path.relative_to(PROJECT_ROOT)); outcome[f"{key}_sha256"] = file_sha256(path)
    outcome["lock_payload_sha256"] = payload_hash(outcome); outcome_path.write_text(json.dumps(outcome, indent=2, sort_keys=True) + "\n")
    print(json.dumps(audit, indent=2, sort_keys=True)); print(json.dumps({"lock": str(outcome_path.relative_to(PROJECT_ROOT)), "sha256": file_sha256(outcome_path)}, indent=2))


if __name__ == "__main__": main()
