#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from audit_and_freeze_v165_factored_ontology_identifiability_population import valid_lock
from audit_and_freeze_v167_exact_evidence_gathering_planner import payload_hash
from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main() -> None:
    paths = {
        "repair_config": PROJECT_ROOT / "configs/v221r1-parser-config-repair.json",
        "plan": PROJECT_ROOT / "docs/v221r1-parser-config-repair-plan.md",
        "failure_record": PROJECT_ROOT / "docs/v221-initial-attempt-failure.md",
        "runner": PROJECT_ROOT / "python/run_v221r1_parser_config_repair.py",
        "verifier": PROJECT_ROOT / "python/verify_and_freeze_v221r1_parser_config_repair_outcome.py",
        "auditor": PROJECT_ROOT / "python/audit_and_freeze_v221r1_parser_config_repair.py",
    }
    output_root = PROJECT_ROOT / "outputs/v221r1-parser-config-repair"
    audit_path = output_root / "design-audit.json"
    lock_path = PROJECT_ROOT / "configs/v221r1-parser-config-repair-lock.json"
    outcome_path = PROJECT_ROOT / "configs/v221r1-parser-config-repair-outcome-lock.json"
    if output_root.exists() or lock_path.exists() or outcome_path.exists():
        raise RuntimeError("V221r1 is already audited, frozen, run, or outcome-frozen")
    repair = json.loads(paths["repair_config"].read_text())
    parent_path = PROJECT_ROOT / repair["parentV221DesignLock"]
    parent = json.loads(parent_path.read_text())
    source_path = PROJECT_ROOT / repair["sourceV220DesignLock"]
    source = json.loads(source_path.read_text())
    failure = repair["failureBoundary"]
    change = repair["repair"]
    parser = source["config_payload"]["parserDesign"]
    base_artifacts = {key: PROJECT_ROOT / value for key, value in parent["config_payload"]["artifacts"].items()}
    checks = {
        "V221_design_lock_is_valid_exact_and_still_immutable": bool(
            valid_lock(parent)
            and file_sha256(parent_path) == repair["parentV221DesignLockSha256"]
            and parent["authorization"]["run_one_development_only_deterministic_evaluation"]
            and not parent["authorization"]["run_model_or_open_protected"]
        ),
        "initial_attempt_failed_before_catalog_observations_or_candidate_evaluation": bool(
            failure["exceptionType"] == "KeyError" and failure["missingKey"] == "parserDesign"
            and failure["failedFunction"] == "build_catalog"
            and not failure["catalogManifestWritten"] and failure["observationCount"] == 0
            and failure["candidateMethodEvaluationCount"] == 0
            and failure["developmentPublicPriorLoadCount"] == 1
            and failure["developmentTruthPriorLoadCount"] == 1
            and failure["protectedJSONLBodyLoadCount"] == 0 and failure["modelRunCount"] == 0
            and not any(base_artifacts[key].exists() for key in ("catalogManifest", "observations", "residualManifest", "summary", "result"))
        ),
        "repair_injects_exact_V220_parser_semantics_only": bool(
            valid_lock(source)
            and file_sha256(source_path) == repair["sourceV220DesignLockSha256"]
            and change["operation"] == "inject_exact_V220_parserDesign_into_runtime_copy_of_locked_V221_config"
            and change["parserDesign"] == parser
            and parser["duplicateTermIdPolicy"] == "FAIL"
            and parser["remoteImportPolicy"] == "FORBID_AND_DO_NOT_RESOLVE"
            and not parser["assertedStateIsInferredOWLEquivalence"]
        ),
        "roles_methods_budgets_controller_gates_residual_and_inputs_are_unchanged": bool(
            not change["roleManifestChanged"] and not change["methodPortfolioChanged"]
            and not change["candidateBudgetsChanged"] and not change["primaryBudgetChanged"]
            and not change["controllerChanged"] and not change["evaluationGatesChanged"]
            and not change["residualDefinitionChanged"] and not change["inputHashesChanged"]
        ),
        "repair_authority_is_one_evaluation_only_with_protected_model_and_effects_closed": bool(
            repair["decisionRule"]["ifRepairAuditPasses"].startswith("run_one_repaired")
            and not repair["decisionRule"]["authorizesProtectedOrModelRun"]
            and not repair["decisionRule"]["authorizesRegistrationMutationServiceActionOrExecution"]
            and all(path.is_file() for path in (*paths.values(), parent_path, source_path))
            and not output_root.exists() and not outcome_path.exists()
        ),
    }
    passed = all(checks.values())
    audit = {
        "schema_version": "221r1-parser-config-repair-design-audit",
        "experiment": repair["experiment"], "passed": passed,
        "decision": "freeze_exact_parser_injection_and_authorize_one_repaired_evaluation" if passed else "reject_V221r1_repair",
        "checks": checks,
    }
    write_json(audit_path, audit)
    if not passed:
        print(json.dumps(audit, indent=2, sort_keys=True))
        raise SystemExit(1)
    dependencies = {
        **paths, "parent_V221_design": parent_path, "source_V220_design": source_path,
        "design_audit": audit_path,
    }
    lock: dict[str, Any] = {
        "schema_version": "221r1-parser-config-repair-lock",
        "experiment": repair["experiment"], "repair_config_payload": repair,
        "authorization": {
            "run_one_repaired_V221_development_evaluation": True,
            "hash_but_do_not_load_V220_protected": True,
            "run_model_or_open_protected": False,
            "register_mutate_service_act_execute": False,
        },
    }
    for key, path in dependencies.items():
        lock[key] = str(path.relative_to(PROJECT_ROOT))
        lock[f"{key}_sha256"] = file_sha256(path)
    lock["lock_payload_sha256"] = payload_hash(lock)
    write_json(lock_path, lock)
    print(json.dumps(audit, indent=2, sort_keys=True))
    print(json.dumps({"lock": str(lock_path.relative_to(PROJECT_ROOT)), "sha256": file_sha256(lock_path)}, indent=2))


if __name__ == "__main__":
    main()
