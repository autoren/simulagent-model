#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from audit_and_freeze_v122_prequery_signal_inventory import payload_hash, valid_lock, write_json
from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT


def main() -> None:
    config_path = PROJECT_ROOT / "configs/v139-repaired-direct-vs-thinking-realization.json"
    plan_path = PROJECT_ROOT / "docs/v139-repaired-direct-vs-thinking-realization-plan.md"
    protocol_path = PROJECT_ROOT / "python/v139_repaired_direct_vs_thinking_realization.py"
    parser_path = PROJECT_ROOT / "python/v138_thinking_parser_contract.py"
    tests_path = PROJECT_ROOT / "python/test_v139_repaired_direct_vs_thinking_realization.py"
    auditor_path = PROJECT_ROOT / "python/audit_and_freeze_v139_repaired_direct_vs_thinking_realization.py"
    runner_path = PROJECT_ROOT / "python/run_v139_repaired_direct_vs_thinking_realization.py"
    verifier_path = PROJECT_ROOT / "python/verify_and_freeze_v139_repaired_direct_vs_thinking_realization_outcome.py"
    audit_path = PROJECT_ROOT / "outputs/v139-repaired-direct-vs-thinking-realization/design-audit.json"
    lock_path = PROJECT_ROOT / "configs/v139-repaired-direct-vs-thinking-realization-lock.json"
    output_dir = PROJECT_ROOT / "outputs/v139-repaired-direct-vs-thinking-realization/model-realization"
    if audit_path.exists() or lock_path.exists() or output_dir.exists():
        raise RuntimeError("V139 already frozen or run")

    config = json.loads(config_path.read_text())
    parent_path = PROJECT_ROOT / config["parentV138OutcomeLock"]
    parent = json.loads(parent_path.read_text())
    v135_outcome_path = PROJECT_ROOT / config["V135OutcomeLock"]
    v135_outcome = json.loads(v135_outcome_path.read_text())
    public_path = PROJECT_ROOT / v135_outcome["public_fixtures"]
    hidden_path = PROJECT_ROOT / v135_outcome["hidden_fixtures"]
    catalog_path = PROJECT_ROOT / v135_outcome["choice_catalog"]
    public_rows = json.loads(public_path.read_text())
    hidden_rows = json.loads(hidden_path.read_text())
    selected_public = [row for row in public_rows if row["split"] == config["population"]["split"]]
    selected_hidden = [row for row in hidden_rows if row["split"] == config["population"]["split"]]
    other_hidden = [row for row in hidden_rows if row["split"] != config["population"]["split"]]
    manifest_path = PROJECT_ROOT / config["modelManifest"]
    manifest = json.loads(manifest_path.read_text())
    snapshot = Path(manifest["snapshot_path"])
    conditions = config["conditions"]
    checks = {
        "V138_valid_and_authorizes_one_fresh_recovery": bool(
            valid_lock(parent)
            and parent["outcome"]["passed"]
            and parent["outcome"]["technical_contract_mismatch_confirmed"]
            and parent["authorization"]["preregister_one_fresh_repaired_V135_development_comparison"]
            and not parent["authorization"]["rerun_reparse_or_modify_V137"]
        ),
        "development_population_exact_and_disjoint_from_V137_test": bool(
            valid_lock(v135_outcome)
            and len(selected_public) == len(selected_hidden) == config["population"]["fixtureCount"]
            and {row["fixture_id"] for row in selected_public} == {row["fixture_id"] for row in selected_hidden}
            and not ({row["fixture_id"] for row in selected_hidden} & {row["fixture_id"] for row in other_hidden})
            and len({row["group_id"] for row in selected_hidden}) == config["population"]["groupCount"]
            and sum(row["phase"].startswith("clear_") for row in selected_hidden) == config["population"]["clearFixtureCount"]
            and sum(row["phase"] == "ambiguous" for row in selected_hidden) == config["population"]["ambiguousFixtureCount"]
            and sum(row["phase"].startswith("clarified_") for row in selected_hidden) == config["population"]["clarifiedFixtureCount"]
            and config["population"]["priorModelGenerationCountOnSplit"] == 0
        ),
        "pinned_model_snapshot_exact": bool(
            manifest["repository"] == config["model"]["repository"]
            and manifest["revision"] == config["model"]["revision"]
            and manifest["quantization_bits"] == config["model"]["quantizationBits"]
            and manifest["weight_bytes"] == manifest["expected_weight_bytes"]
            and snapshot.is_dir()
        ),
        "two_deterministic_no_retry_conditions": bool(
            len(conditions) == 2
            and {row["id"] for row in conditions} == {"direct", "thinking"}
            and all(row["temperature"] == 0.0 and row["samplesPerPrompt"] == 1 and row["retryCount"] == 0 for row in conditions)
            and next(row for row in conditions if row["id"] == "direct")["promptThinkOpened"] is False
            and next(row for row in conditions if row["id"] == "thinking")["promptThinkOpened"] is True
            and config["prompt"]["sameSemanticPromptAcrossConditions"]
        ),
        "V138_parser_contract_is_explicit": bool(
            config["parsing"]["contract"] == "V138_prompt_opened_stateful_thinking_suffix"
            and config["parsing"]["thinkingGeneratedSuffixInitialDepth"] == 1
        ),
        "raw_reasoning_never_persisted_or_inspected": bool(
            not config["parsing"]["persistRawResponseOrThinkingTrace"]
            and config["accessGates"]["maximumManualRawResponseOrTraceInspectionCount"] == 0
            and config["accessGates"]["maximumPersistedRawResponseOrTraceCount"] == 0
        ),
        "exact_generation_budget": config["accessGates"]["maximumModelGenerationCount"] == len(selected_public) * len(conditions) == 200,
        "authority_and_external_access_remain_closed": bool(
            not config["decisionRule"]["outcomeAuthorizesV134LanguageOrRerun"]
            and not config["decisionRule"]["outcomeAuthorizesAPIInductionTrainingAuthorityActionOrExecution"]
            and config["qualificationGates"]["requiredTrueHypothesisRetention"] == 1.0
            and config["qualificationGates"]["maximumActualExecutionCount"] == 0
        ),
        "code_and_output_hold": all(path.is_file() for path in (config_path, plan_path, protocol_path, parser_path, tests_path, auditor_path, runner_path, verifier_path)) and not output_dir.exists(),
    }
    passed = all(checks.values())
    audit = {
        "schema_version": "139-repaired-direct-vs-thinking-realization-design-audit",
        "experiment": config["experiment"],
        "passed": passed,
        "checks": checks,
        "decision": "freeze_V139_and_authorize_one_exact_recovery_comparison" if passed else "reject_V139_without_model_run",
        "summary": {
            "development_fixture_count": len(selected_public),
            "condition_count": len(conditions),
            "generation_count": len(selected_public) * len(conditions),
            "model": config["model"],
        },
    }
    write_json(audit_path, audit)
    if not passed:
        print(json.dumps(audit, indent=2, sort_keys=True))
        raise SystemExit(1)
    dependencies = {
        "config": config_path,
        "parent_outcome": parent_path,
        "V135_outcome": v135_outcome_path,
        "choice_catalog": catalog_path,
        "public_fixtures": public_path,
        "hidden_fixtures": hidden_path,
        "V136_config": PROJECT_ROOT / config["V136Config"],
        "model_manifest": manifest_path,
        "plan": plan_path,
        "protocol": protocol_path,
        "parser": parser_path,
        "tests": tests_path,
        "auditor": auditor_path,
        "runner": runner_path,
        "verifier": verifier_path,
        "design_audit": audit_path,
    }
    lock: dict[str, Any] = {
        "schema_version": "139-repaired-direct-vs-thinking-realization-lock",
        "experiment": config["experiment"],
        "config_payload": config,
        "authorization": {
            "run_exactly_one_pinned_repaired_comparison": True,
            "modify_prompt_population_model_conditions_parser_gates_or_decision": False,
            "retry_rerun_or_inspect_raw_responses_or_traces": False,
            "touch_V134_or_external_language": False,
            "run_API_induction_training_authority_action_or_execution": False,
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
