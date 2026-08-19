#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from audit_and_freeze_v122_prequery_signal_inventory import payload_hash, valid_lock, write_json
from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT


def main() -> None:
    config_path = PROJECT_ROOT / "configs/v144-local-certificate-realization.json"
    plan_path = PROJECT_ROOT / "docs/v144-local-certificate-realization-plan.md"
    protocol_path = PROJECT_ROOT / "python/v144_local_certificate_realization.py"
    tests_path = PROJECT_ROOT / "python/test_v144_local_certificate_realization.py"
    auditor_path = PROJECT_ROOT / "python/audit_and_freeze_v144_local_certificate_realization.py"
    runner_path = PROJECT_ROOT / "python/run_v144_local_certificate_realization.py"
    verifier_path = PROJECT_ROOT / "python/verify_and_freeze_v144_local_certificate_realization_outcome.py"
    output_dir = PROJECT_ROOT / "outputs/v144-local-certificate-realization/preregistration"
    audit_path = PROJECT_ROOT / "outputs/v144-local-certificate-realization/design-audit.json"
    lock_path = PROJECT_ROOT / "configs/v144-local-certificate-realization-lock.json"
    if output_dir.exists() or audit_path.exists() or lock_path.exists():
        raise RuntimeError("V144 already preregistered")
    config = json.loads(config_path.read_text())
    parent_path = PROJECT_ROOT / config["parentV143OutcomeLock"]
    parent = json.loads(parent_path.read_text())
    v142_config_path = PROJECT_ROOT / config["V142Config"]
    v142_config = json.loads(v142_config_path.read_text())
    v136_path = PROJECT_ROOT / config["V136Config"]
    manifest_path = PROJECT_ROOT / config["modelManifest"]
    manifest = json.loads(manifest_path.read_text())
    catalog_path = PROJECT_ROOT / parent["choice_catalog"]
    full_public_path = PROJECT_ROOT / parent["hidden_fixtures"].replace("hidden-fixtures", "public-fixtures")
    full_hidden_path = PROJECT_ROOT / parent["hidden_fixtures"]
    public = [row for row in json.loads(full_public_path.read_text()) if row["split"] == config["population"]["split"]]
    hidden = [row for row in json.loads(full_hidden_path.read_text()) if row["split"] == config["population"]["split"]]
    public_ids = {row["fixture_id"] for row in public}
    hidden_ids = {row["fixture_id"] for row in hidden}
    public_forbidden = {"group_id", "family_id", "stage", "language_class", "truth_choice_id", "compatible_choice_ids", "variant_index"}
    qualification = config["qualificationGates"]
    access = config["accessGates"]
    checks = {
        "V143_valid_and_authorizes_only_preregistration": bool(
            valid_lock(parent)
            and parent["outcome"]["passed"]
            and parent["outcome"]["oracle_certificate_policy_pass"]
            and parent["authorization"]["preregister_one_pinned_local_V142_development_realization"]
            and not parent["authorization"]["run_model_before_separate_preregistration"]
            and not parent["authorization"]["open_V142_test_split_language"]
        ),
        "development_population_exact_and_public_hidden_aligned": bool(
            len(public) == config["population"]["fixtureCount"]
            and len(hidden) == config["population"]["fixtureCount"]
            and public_ids == hidden_ids
            and len({row["group_id"] for row in hidden}) == config["population"]["groupCount"]
            and sum(row["stage"] == "ambiguous" for row in hidden) == config["population"]["ambiguousFixtureCount"]
            and sum(row["stage"] != "ambiguous" for row in hidden) == config["population"]["decidableFixtureCount"]
        ),
        "public_fixtures_hide_ground_truth": all(not (public_forbidden & set(row)) for row in public),
        "pinned_model_manifest_exact_and_local_snapshot_exists": bool(
            manifest["repository"] == config["model"]["repository"]
            and manifest["revision"] == config["model"]["revision"]
            and Path(manifest["snapshot_path"]).is_dir()
        ),
        "single_thinking_generation_no_retry_contract": bool(
            config["model"]["enableThinking"]
            and config["model"]["promptThinkOpened"]
            and config["model"]["samplesPerPrompt"] == 1
            and config["model"]["retryCount"] == 0
            and access["maximumModelGenerationCount"] == len(public)
            and access["maximumGenerationCountPerFixture"] == 1
            and access["maximumTestFixtureModelGenerationCount"] == 0
        ),
        "noncompensatory_semantic_and_sequential_gates_present": all(
            key in qualification for key in (
                "minimumCertificateStructuralValidity",
                "minimumOverallFinalExactAccuracy",
                "minimumEveryLanguageClassFinalExactAccuracy",
                "minimumCompatibleSetExactAccuracy",
                "minimumCertificateTrueOptionRetention",
                "minimumAmbiguitySensitivity",
                "minimumDecidableSpecificity",
                "minimumConditionalProposalCorrectness",
                "maximumFalseKnownRateOnNonKnownTruths",
                "maximumSequentialMeanDecisionCost",
                "minimumSequentialImprovementOverNoQuery",
            )
        ),
        "fail_closed_zero_authority_execution_access": bool(
            qualification["requiredDeterministicFinalOutputValidity"] == 1.0
            and qualification["requiredAuthoritativeTrueHypothesisRetention"] == 1.0
            and all(access[key] == 0 for key in (
                "maximumV134LanguageReadCount",
                "maximumExternalLanguageReadCount",
                "maximumTestFixtureModelGenerationCount",
                "maximumRetryCount",
                "maximumManualRawResponseOrTraceInspectionCount",
                "maximumPersistedRawResponseOrTraceCount",
                "maximumAPICallCount",
                "maximumTrainingRunCount",
                "maximumRealServiceCallCount",
                "maximumExternalSideEffectCount",
                "maximumActualExecutionCount",
            ))
        ),
        "required_preregistration_files_exist": all(
            path.is_file() for path in (config_path, plan_path, protocol_path, tests_path, auditor_path, runner_path, verifier_path)
        ),
    }
    passed = all(checks.values())
    output_dir.mkdir(parents=True, exist_ok=False)
    development_public_path = output_dir / "development-public-fixtures.json"
    development_hidden_path = output_dir / "development-hidden-fixtures.json"
    write_json(development_public_path, public)
    write_json(development_hidden_path, hidden)
    audit = {
        "schema_version": "144-local-certificate-realization-design-audit",
        "experiment": config["experiment"],
        "passed": passed,
        "checks": checks,
        "development_fixture_count": len(public),
        "test_fixture_model_generation_count": 0,
        "decision": "authorize_exact_single_V144_development_run" if passed else "close_V144_before_model_run",
        "access": {
            "model_load_count": 0,
            "model_generation_count": 0,
            "test_fixture_model_generation_count": 0,
            "V134_language_read_count": 0,
            "external_language_read_count": 0,
            "API_call_count": 0,
            "training_run_count": 0,
            "actual_execution_count": 0,
        },
    }
    write_json(audit_path, audit)
    if not passed:
        print(json.dumps(audit, indent=2, sort_keys=True))
        raise SystemExit(1)
    dependencies = {
        "config": config_path,
        "parent_outcome": parent_path,
        "V142_config": v142_config_path,
        "V136_config": v136_path,
        "model_manifest": manifest_path,
        "choice_catalog": catalog_path,
        "development_public_fixtures": development_public_path,
        "development_hidden_fixtures": development_hidden_path,
        "plan": plan_path,
        "protocol": protocol_path,
        "tests": tests_path,
        "auditor": auditor_path,
        "runner": runner_path,
        "verifier": verifier_path,
        "design_audit": audit_path,
    }
    lock: dict[str, Any] = {
        "schema_version": "144-local-certificate-realization-lock",
        "experiment": config["experiment"],
        "config_payload": config,
        "authorization": {
            "run_exact_single_pinned_local_development_realization": True,
            "modify_retry_rerun_reprompt_or_mine_V144": False,
            "open_or_generate_on_V142_test_split": False,
            "touch_V134_or_external_language": False,
            "run_API_training_induction_authority_action_or_execution": False,
        },
    }
    for key, path in dependencies.items():
        lock[key] = str(path.relative_to(PROJECT_ROOT))
        lock[f"{key}_sha256"] = file_sha256(path)
    lock["lock_payload_sha256"] = payload_hash(lock)
    write_json(lock_path, lock)
    print(json.dumps({"passed": passed, "checks": checks, "decision": audit["decision"]}, indent=2, sort_keys=True))
    print(json.dumps({"lock": str(lock_path.relative_to(PROJECT_ROOT)), "sha256": file_sha256(lock_path)}, indent=2))


if __name__ == "__main__":
    main()
