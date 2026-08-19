#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

from audit_and_freeze_v122_prequery_signal_inventory import payload_hash, valid_lock, write_json
from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from v145_finite_certificate_codebook import oracle_code
from v147_closed_alternative_scoring import alias_mapping, evaluate, render_prompt, select_scored_code


def main() -> None:
    config_path = PROJECT_ROOT / "configs/v147-closed-alternative-scoring.json"
    plan_path = PROJECT_ROOT / "docs/v147-closed-alternative-scoring-plan.md"
    protocol_path = PROJECT_ROOT / "python/v147_closed_alternative_scoring.py"
    tests_path = PROJECT_ROOT / "python/test_v147_closed_alternative_scoring.py"
    auditor_path = PROJECT_ROOT / "python/audit_and_freeze_v147_closed_alternative_scoring.py"
    runner_path = PROJECT_ROOT / "python/run_v147_closed_alternative_scoring.py"
    verifier_path = PROJECT_ROOT / "python/verify_and_freeze_v147_closed_alternative_scoring_outcome.py"
    output_dir = PROJECT_ROOT / "outputs/v147-closed-alternative-scoring/preregistration"
    audit_path = PROJECT_ROOT / "outputs/v147-closed-alternative-scoring/design-audit.json"
    lock_path = PROJECT_ROOT / "configs/v147-closed-alternative-scoring-lock.json"
    if output_dir.exists() or audit_path.exists() or lock_path.exists():
        raise RuntimeError("V147 already preregistered")

    config = json.loads(config_path.read_text())
    parent_path = PROJECT_ROOT / config["parentV146OutcomeLock"]
    parent = json.loads(parent_path.read_text())
    v136_path = PROJECT_ROOT / config["V136Config"]
    v136 = json.loads(v136_path.read_text())
    manifest_path = PROJECT_ROOT / config["modelManifest"]
    manifest = json.loads(manifest_path.read_text())
    catalog_path = PROJECT_ROOT / parent["choice_catalog"]
    codebook_path = PROJECT_ROOT / parent["certificate_codebook"]
    catalog = json.loads(catalog_path.read_text())
    codebook = json.loads(codebook_path.read_text())["entries"]
    full_public = json.loads((PROJECT_ROOT / parent["public_fixtures"]).read_text())
    full_hidden = json.loads((PROJECT_ROOT / parent["hidden_fixtures"]).read_text())
    public = sorted(
        (row for row in full_public if row["split"] == config["population"]["split"]),
        key=lambda row: row["fixture_id"],
    )
    hidden = sorted(
        (row for row in full_hidden if row["split"] == config["population"]["split"]),
        key=lambda row: row["fixture_id"],
    )
    hidden_by_id = {row["fixture_id"]: row for row in hidden}
    public_forbidden = {
        "group_id", "family_id", "stage", "language_class", "truth_choice_id",
        "compatible_choice_ids", "variant_index",
    }

    snapshot = Path(manifest["snapshot_path"])
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    aliases = config["scoring"]["aliases"]
    alias_tokens = {
        alias: tokenizer.encode(alias, add_special_tokens=False)
        for alias in aliases
    }
    prompt_token_counts: list[int] = []
    boundary_checks: list[bool] = []
    mappings: list[tuple[tuple[str, str], ...]] = []
    for fixture in public:
        payload = render_prompt(catalog, codebook, fixture, config)
        prompt = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": config["prompt"]["system"]},
                {"role": "user", "content": payload},
            ],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=config["model"]["enableThinking"],
        )
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        prompt_token_counts.append(len(prompt_ids))
        mappings.append(tuple(sorted(alias_mapping(fixture["fixture_id"], codebook, aliases).items())))
        for alias in aliases:
            full_ids = tokenizer.encode(prompt + alias, add_special_tokens=False)
            boundary_checks.append(
                full_ids[: len(prompt_ids)] == prompt_ids
                and full_ids[len(prompt_ids) :] == alias_tokens[alias]
            )

    oracle_completed: dict[str, dict[str, Any]] = {}
    for fixture in public:
        fixture_id = fixture["fixture_id"]
        mapping = alias_mapping(fixture_id, codebook, aliases)
        truth_code = oracle_code(hidden_by_id[fixture_id])
        truth_alias = next(alias for alias, code in mapping.items() if code == truth_code)
        scores = {alias: -10.0 for alias in aliases}
        scores[truth_alias] = -1.0
        oracle_completed[fixture_id] = {
            **select_scored_code(fixture_id, scores, codebook, config),
            "scoring_seconds": 0.0,
        }
    oracle_access = {
        "V134_language_read_count": 0,
        "external_language_read_count": 0,
        "tokenizer_load_count": 1,
        "model_load_count": 1,
        "model_generation_count": 0,
        "model_scoring_fixture_count": len(public),
        "candidate_sequence_score_count": len(public) * len(aliases),
        "test_fixture_score_count": 0,
        "retry_count": 0,
        "manual_raw_response_or_trace_inspection_count": 0,
        "persisted_raw_response_or_trace_count": 0,
        "API_call_count": 0,
        "training_run_count": 0,
        "real_service_call_count": 0,
        "external_side_effect_count": 0,
        "actual_execution_count": 0,
    }
    oracle_summary = evaluate(oracle_completed, hidden, catalog, v136, oracle_access, config)
    qualification = config["qualificationGates"]
    access = config["accessGates"]
    checks = {
        "V146_outcome_valid_and_authorizes_scoring_preregistration_only": bool(
            valid_lock(parent)
            and parent["outcome"]["passed"]
            and parent["outcome"]["fresh_population_and_codebook_pass"]
            and parent["authorization"]["preregister_closed_alternative_scoring_protocol"]
            and not parent["authorization"]["run_language_or_model_before_separate_preregistration"]
            and not parent["authorization"]["run_API_training_induction_authority_action_or_execution"]
        ),
        "development_population_exact_public_hidden_aligned": bool(
            len(public) == config["population"]["fixtureCount"]
            and len(hidden) == config["population"]["fixtureCount"]
            and [row["fixture_id"] for row in public] == [row["fixture_id"] for row in hidden]
            and len({row["group_id"] for row in hidden}) == config["population"]["groupCount"]
            and sum(row["stage"] == "ambiguous" for row in hidden) == config["population"]["ambiguousFixtureCount"]
            and sum(row["stage"] != "ambiguous" for row in hidden) == config["population"]["decidableFixtureCount"]
        ),
        "public_fixtures_hide_ground_truth": all(not (public_forbidden & set(row)) for row in public),
        "V146_test_is_retired_and_receives_zero_scores": bool(
            config["population"]["V146TestSplitRetiredDueToPrePreregistrationInspection"]
            and config["population"]["futureEvidenceRequiresNewBlindPopulation"]
            and config["population"]["testFixtureScoreCount"] == 0
            and access["maximumTestFixtureScoreCount"] == 0
            and config["decisionRule"]["passDoesNotAuthorizeV146TestUse"]
        ),
        "pinned_local_model_manifest_exact": bool(
            manifest["repository"] == config["model"]["repository"]
            and manifest["revision"] == config["model"]["revision"]
            and manifest["quantization_bits"] == config["model"]["quantizationBits"]
            and snapshot.is_dir()
        ),
        "all_aliases_exactly_three_tokens": all(
            len(tokens) == config["scoring"]["requiredTokensPerAlias"]
            for tokens in alias_tokens.values()
        ),
        "all_prompt_alias_token_boundaries_exact": bool(boundary_checks and all(boundary_checks)),
        "all_prompts_within_frozen_budget": bool(
            prompt_token_counts
            and max(prompt_token_counts) <= config["prompt"]["maximumPromptTokens"]
        ),
        "fixture_specific_alias_mapping_is_complete_and_varies": bool(
            all(len(mapping) == len(aliases) for mapping in mappings)
            and len(set(mappings)) > 1
        ),
        "score_only_zero_generation_single_load_budget": bool(
            not config["model"]["enableThinking"]
            and config["scoring"]["freeFormGenerationCount"] == 0
            and config["scoring"]["retryCount"] == 0
            and access["maximumTokenizerLoadCount"] == 1
            and access["maximumModelLoadCount"] == 1
            and access["maximumModelGenerationCount"] == 0
            and access["maximumGenerationCountPerFixture"] == 0
            and access["maximumTestFixtureModelGenerationCount"] == 0
            and access["maximumModelScoringFixtureCount"] == len(public)
            and access["maximumCandidateSequenceScoreCount"] == len(public) * len(aliases)
        ),
        "oracle_mock_passes_full_noncompensatory_evaluator": bool(
            oracle_summary["qualified"]
            and all(oracle_summary["qualification_gates"].values())
            and all(oracle_summary["access_gates"].values())
        ),
        "noncompensatory_semantic_and_sequential_gates_present": all(
            key in qualification
            for key in (
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
            and all(
                access[key] == 0
                for key in (
                    "maximumV134LanguageReadCount",
                    "maximumExternalLanguageReadCount",
                    "maximumModelGenerationCount",
                    "maximumGenerationCountPerFixture",
                    "maximumTestFixtureModelGenerationCount",
                    "maximumTestFixtureScoreCount",
                    "maximumRetryCount",
                    "maximumManualRawResponseOrTraceInspectionCount",
                    "maximumPersistedRawResponseOrTraceCount",
                    "maximumAPICallCount",
                    "maximumTrainingRunCount",
                    "maximumRealServiceCallCount",
                    "maximumExternalSideEffectCount",
                    "maximumActualExecutionCount",
                )
            )
        ),
        "required_preregistration_files_exist": all(
            path.is_file()
            for path in (config_path, plan_path, protocol_path, tests_path, auditor_path, runner_path, verifier_path)
        ),
    }
    passed = all(checks.values())
    output_dir.mkdir(parents=True, exist_ok=False)
    development_public_path = output_dir / "development-public-fixtures.json"
    development_hidden_path = output_dir / "development-hidden-fixtures.json"
    write_json(development_public_path, public)
    write_json(development_hidden_path, hidden)
    audit = {
        "schema_version": "147-closed-alternative-scoring-design-audit",
        "experiment": config["experiment"],
        "passed": passed,
        "checks": checks,
        "development_fixture_count": len(public),
        "alias_token_ids": alias_tokens,
        "minimum_prompt_token_count": min(prompt_token_counts),
        "maximum_prompt_token_count": max(prompt_token_counts),
        "prompt_alias_boundary_check_count": len(boundary_checks),
        "V146_test_fixture_score_count": 0,
        "V146_test_split_retired": True,
        "model_load_count": 0,
        "model_generation_count": 0,
        "model_scoring_fixture_count": 0,
        "decision": "authorize_exact_single_V147_development_scoring_run" if passed else "close_V147_before_model_run",
    }
    write_json(audit_path, audit)
    if not passed:
        print(json.dumps(audit, indent=2, sort_keys=True))
        raise SystemExit(1)

    dependencies = {
        "config": config_path,
        "parent_outcome": parent_path,
        "V136_config": v136_path,
        "model_manifest": manifest_path,
        "choice_catalog": catalog_path,
        "certificate_codebook": codebook_path,
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
        "schema_version": "147-closed-alternative-scoring-lock",
        "experiment": config["experiment"],
        "config_payload": config,
        "authorization": {
            "run_exact_single_pinned_local_development_scoring_realization": True,
            "modify_retry_rescore_rerun_reprompt_or_mine_V147": False,
            "score_or_use_V146_test_split": False,
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
