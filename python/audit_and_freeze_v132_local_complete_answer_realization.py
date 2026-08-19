#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from audit_and_freeze_v122_prequery_signal_inventory import payload_hash, valid_lock, write_json
from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT


def main() -> None:
    config_path = PROJECT_ROOT / "configs/v132-local-complete-answer-realization.json"
    plan_path = PROJECT_ROOT / "docs/v132-local-complete-answer-realization-plan.md"
    protocol_path = PROJECT_ROOT / "python/v132_local_complete_answer_realization.py"
    tests_path = PROJECT_ROOT / "python/test_v132_local_complete_answer_realization.py"
    runner_path = PROJECT_ROOT / "python/run_v132_local_complete_answer_realization.py"
    verifier_path = PROJECT_ROOT / "python/verify_and_freeze_v132_local_complete_answer_realization_outcome.py"
    auditor_path = PROJECT_ROOT / "python/audit_and_freeze_v132_local_complete_answer_realization.py"
    audit_path = PROJECT_ROOT / "outputs/v132-local-complete-answer-realization/design-audit.json"
    lock_path = PROJECT_ROOT / "configs/v132-local-complete-answer-realization-lock.json"
    if audit_path.exists() or lock_path.exists(): raise RuntimeError("V132 already frozen")
    config = json.loads(config_path.read_text())
    parent_path = PROJECT_ROOT / config["parentV131OutcomeLock"]
    parent = json.loads(parent_path.read_text()); parent_lock_path = PROJECT_ROOT / parent["analysis_lock"]
    parent_lock = json.loads(parent_lock_path.read_text())
    manifest_path = PROJECT_ROOT / config["modelManifest"]
    manifest = json.loads(manifest_path.read_text())
    snapshot = Path(manifest["snapshot_path"])
    catalog_path = PROJECT_ROOT / config["choiceCatalog"]
    population_path = PROJECT_ROOT / config["fixturePopulation"]
    catalog = json.loads(catalog_path.read_text()); population = json.loads(population_path.read_text())
    auth = parent["authorization"]
    checks = {
        "V131_is_valid_and_authorizes_only_protocol_preregistration": bool(
            valid_lock(parent) and valid_lock(parent_lock) and parent["outcome"]["passed"]
            and parent["outcome"]["audit_pass"]
            and auth["preregister_local_model_complete_answer_realization_protocol"]
            and not auth["extract_language_or_run_model_before_next_lock"]
            and not auth["run_API_training_action_or_execution"]
        ),
        "frozen_population_is_exact_66_cell_census": bool(
            population["fixture_count"] == config["condition"]["totalFixtureCount"] == 264
            and population["cell_count"] == 66 and set(population["cell_counts"].values()) == {4}
            and catalog["choice_count"] == 11 and catalog["complete_safe_composite_hypothesis_universe"]
        ),
        "pinned_local_model_snapshot_matches": bool(
            manifest["repository"] == config["condition"]["repository"]
            and manifest["revision"] == config["condition"]["revision"]
            and manifest["quantization_bits"] == config["condition"]["quantizationBits"]
            and snapshot.is_dir() and manifest["weight_bytes"] == manifest["expected_weight_bytes"]
        ),
        "one_pass_no_retry_and_exact_V130_boundary": bool(
            config["condition"]["generationCountPerFixture"] == 1
            and config["condition"]["retryCount"] == 0
            and config["decoding"]["samplesPerPrompt"] == 1
            and not config["decoding"]["retryOnMalformedOutput"]
            and config["decoding"]["temperature"] == 0.0
            and config["evidenceGates"]["minimumOverallExactChoiceAccuracy"] == 0.9725
            and config["downstreamGates"]["answerReliabilityUsedByFrozenPlanner"] == 0.9725
        ),
        "hidden_labels_and_novel_members_are_not_prompted": bool(
            config["extraction"]["hideNovelMemberIntentIdentifiers"]
            and config["extraction"]["hideGroundTruthClassServiceIntentAndDomainFromPrompt"]
            and not config["extraction"]["useDialogueHistory"] and not config["extraction"]["useSlotValues"]
        ),
        "permanently_non_authoritative_and_zero_execution": bool(
            config["authorityBoundary"]["modelIsPermanentlyNonAuthoritativeEvidenceOnly"]
            and config["authorityBoundary"]["completeSafeHypothesisUniverseAlwaysRetained"]
            and config["authorityBoundary"]["authoritativeStatePosteriorCapabilitiesAndPolicyAreImmutable"]
            and config["authorityBoundary"]["allActionsAreCounterfactualShadowActions"]
            and config["authorityBoundary"]["actualExecutionCount"] == 0
            and not config["decisionRule"]["passAuthorizesHumanEquivalenceOrRepeatedSampleIndependenceClaim"]
            and not config["decisionRule"]["passAuthorizesProtectedInductionRicherPlanningAPITrainingActionOrExecution"]
        ),
        "code_and_output_hold": all(path.is_file() for path in (plan_path, protocol_path, tests_path, runner_path, verifier_path, auditor_path)) and not (PROJECT_ROOT / "outputs/v132-local-complete-answer-realization/model-realization").exists(),
    }
    passed = all(checks.values())
    audit = {
        "schema_version": "132-local-complete-answer-realization-design-audit",
        "experiment": config["experiment"], "passed": passed, "checks": checks,
        "decision": "freeze_V132_and_authorize_one_local_run" if passed else "reject_V132_design",
        "summary": {"fixture_count": population["fixture_count"], "cell_count": population["cell_count"], "choice_count": catalog["choice_count"], "generation_count": config["condition"]["totalGenerationCount"], "minimum_exact_accuracy": config["evidenceGates"]["minimumOverallExactChoiceAccuracy"]},
        "pre_run_access": {"language_read_count": 0, "model_load_count": 0, "model_generation_count": 0, "LLM_API_call_count": 0, "actual_execution_count": 0},
    }
    write_json(audit_path, audit)
    if not passed: print(json.dumps(audit, indent=2, sort_keys=True)); raise SystemExit(1)
    deps = {
        "config": config_path, "parent_outcome": parent_path, "parent_analysis_lock": parent_lock_path,
        "source_archive": PROJECT_ROOT / config["sourceArchive"], "choice_catalog": catalog_path,
        "fixture_population": population_path, "model_manifest": manifest_path,
        "V130_config": PROJECT_ROOT / config["V130Config"], "baseline_config": PROJECT_ROOT / config["baselineConfig"],
        "plan": plan_path, "protocol": protocol_path, "tests": tests_path, "runner": runner_path,
        "verifier": verifier_path, "auditor": auditor_path, "design_audit": audit_path,
    }
    lock: dict[str, Any] = {
        "schema_version": "132-local-complete-answer-realization-lock", "experiment": config["experiment"],
        "config_payload": config,
        "authorization": {
            "run_exactly_one_pinned_local_complete_answer_condition": True,
            "modify_prompt_catalog_population_model_decoding_gates_or_decision": False,
            "retry_rerun_or_inspect_raw_responses": False,
            "run_API_train_grant_authority_or_execute": False,
        },
    }
    for key, path in deps.items():
        lock[key] = str(path.relative_to(PROJECT_ROOT)); lock[f"{key}_sha256"] = file_sha256(path)
    lock["lock_payload_sha256"] = payload_hash(lock); write_json(lock_path, lock)
    print(json.dumps(audit, indent=2, sort_keys=True))
    print(json.dumps({"lock": str(lock_path.relative_to(PROJECT_ROOT)), "sha256": file_sha256(lock_path)}, indent=2))


if __name__ == "__main__":
    main()
