#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import math
from typing import Any

from audit_and_freeze_v122_prequery_signal_inventory import payload_hash, write_json
from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from v147_closed_alternative_scoring import alias_mapping, evaluate, select_scored_code


def main() -> None:
    lock_path = PROJECT_ROOT / "configs/v147-closed-alternative-scoring-lock.json"
    result_path = PROJECT_ROOT / "outputs/v147-closed-alternative-scoring/model-scoring-realization/result.json"
    access_path = PROJECT_ROOT / "outputs/v147-closed-alternative-scoring/model-scoring-realization/access.json"
    doc_path = PROJECT_ROOT / "docs/v147-closed-alternative-scoring-results.md"
    audit_path = PROJECT_ROOT / "outputs/v147-closed-alternative-scoring/outcome-audit.json"
    outcome_path = PROJECT_ROOT / "configs/v147-closed-alternative-scoring-outcome-lock.json"
    verifier_path = PROJECT_ROOT / "python/verify_and_freeze_v147_closed_alternative_scoring_outcome.py"
    if audit_path.exists() or outcome_path.exists():
        raise RuntimeError("V147 outcome already frozen")
    if not doc_path.is_file():
        raise RuntimeError("write the V147 results document before freezing the outcome")

    lock = json.loads(lock_path.read_text())
    config = lock["config_payload"]
    result = json.loads(result_path.read_text())
    access = json.loads(access_path.read_text())
    catalog = json.loads((PROJECT_ROOT / lock["choice_catalog"]).read_text())
    codebook = json.loads((PROJECT_ROOT / lock["certificate_codebook"]).read_text())["entries"]
    hidden = json.loads((PROJECT_ROOT / lock["development_hidden_fixtures"]).read_text())
    v136 = json.loads((PROJECT_ROOT / lock["V136_config"]).read_text())
    expected = evaluate(result["fixtures"], hidden, catalog, v136, access, config)
    dependencies = [key for key in lock if not key.endswith("_sha256") and f"{key}_sha256" in lock]
    aliases = config["scoring"]["aliases"]
    raw_fields = {"raw_response", "thinking_trace", "final_text", "prompt", "payload", "conversation"}

    reselection_checks = []
    for fixture_id, row in result["fixtures"].items():
        scores = row.get("scores_by_alias")
        mapping = alias_mapping(fixture_id, codebook, aliases)
        expected_mapping_hash = hashlib.sha256(
            json.dumps(mapping, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        valid_scores = bool(
            isinstance(scores, dict)
            and set(scores) == set(aliases)
            and all(
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(value)
                for value in scores.values()
            )
        )
        if not valid_scores:
            reselection_checks.append(False)
            continue
        selected = select_scored_code(fixture_id, scores, codebook, config)
        fields = (
            "code_valid",
            "validation_reason",
            "normalized_certificate",
            "final_choice_id",
            "final_json",
            "final_output_structurally_valid",
            "selection_valid",
            "selection_reason",
            "selected_alias",
            "selected_certificate_code",
            "relative_probabilities_by_code",
            "top_relative_probability",
            "score_margin",
        )
        reselection_checks.append(
            all(row.get(field) == selected.get(field) for field in fields)
            and row.get("alias_mapping_sha256") == expected_mapping_hash
            and row.get("candidate_sequence_score_count") == len(aliases)
        )

    invalid = [row for row in result["fixtures"].values() if not row["code_valid"]]
    checks = {
        "preregistration_lock_and_dependencies_exact": bool(
            payload_hash({key: value for key, value in lock.items() if key != "lock_payload_sha256"}) == lock["lock_payload_sha256"]
            and all(file_sha256(PROJECT_ROOT / lock[key]) == lock[f"{key}_sha256"] for key in dependencies)
        ),
        "completed_exact_development_scoring_realization": bool(
            result["completed_condition"]
            and len(result["fixtures"]) == config["population"]["fixtureCount"]
            and access["model_scoring_fixture_count"] == config["population"]["fixtureCount"]
            and access["candidate_sequence_score_count"] == config["population"]["fixtureCount"] * len(aliases)
        ),
        "summary_and_decision_exact": result["summary"] == expected and result["decision"] == expected["decision"],
        "all_score_vectors_exact_finite_and_reselectable": bool(reselection_checks and all(reselection_checks)),
        "single_local_load_zero_generation_retry_API_training_execution": bool(
            access["tokenizer_load_count"] == 1
            and access["model_load_count"] == 1
            and access["model_generation_count"] == 0
            and access["retry_count"] == 0
            and access["API_call_count"] == 0
            and access["training_run_count"] == 0
            and access["actual_execution_count"] == 0
        ),
        "retired_V146_test_received_zero_scores_and_no_authorization": bool(
            access["test_fixture_score_count"] == 0
            and access["test_fixture_model_generation_count"] == 0
            and result["V146_test_fixture_score_count"] == 0
            and result["V146_test_split_retired"]
            and not lock["authorization"]["score_or_use_V146_test_split"]
        ),
        "no_raw_prompt_response_trace_final_text_or_conversation_persisted": all(
            not (raw_fields & set(row)) and not row["raw_response_or_trace_persisted"]
            for row in result["fixtures"].values()
        ),
        "only_valid_registered_certificates_are_normalized": all(
            (row["normalized_certificate"] is not None) == row["code_valid"]
            for row in result["fixtures"].values()
        ),
        "invalid_or_tied_selections_fail_closed": all(
            row["final_choice_id"] == "A00" and row["final_output_structurally_valid"]
            for row in invalid
        ),
        "all_outputs_non_authoritative_nonexecuting": all(
            row["permanently_non_authoritative"]
            and not row["authoritative_hypothesis_universe_pruned"]
            and not row["capability_defined"]
            and not row["executable"]
            and row["actual_execution_count"] == 0
            for row in result["fixtures"].values()
        ),
        "all_access_gates_pass": all(expected["access_gates"].values()),
        "relative_score_diagnostics_not_claimed_calibrated": expected["calibration_diagnostics"]["relative_scores_are_not_claimed_calibrated"],
    }
    passed = all(checks.values())
    audit = {
        "schema_version": "147-closed-alternative-scoring-outcome-audit",
        "experiment": config["experiment"],
        "passed": passed,
        "checks": checks,
        "qualified": expected["qualified"],
        "decision": expected["decision"],
        "metrics": expected["metrics"],
        "calibration_diagnostics": expected["calibration_diagnostics"],
        "qualification_gates": expected["qualification_gates"],
        "access_gates": expected["access_gates"],
    }
    write_json(audit_path, audit)
    if not passed:
        print(json.dumps(audit, indent=2, sort_keys=True))
        raise SystemExit(1)

    paths = {
        "analysis_lock": lock_path,
        "result": result_path,
        "access": access_path,
        "verifier": verifier_path,
        "audit": audit_path,
        "results_document": doc_path,
    }
    outcome: dict[str, Any] = {
        "schema_version": "147-closed-alternative-scoring-outcome-lock",
        "experiment": config["experiment"],
        "outcome": {
            "passed": True,
            "audit_pass": True,
            "realization_completed": True,
            "qualified": expected["qualified"],
            "decision": expected["decision"],
            "metrics": expected["metrics"],
            "calibration_diagnostics": expected["calibration_diagnostics"],
            "qualification_gates": expected["qualification_gates"],
        },
        "authorization": {
            "retain_as_project_authored_synthetic_development_evidence_only": True,
            "preregister_new_blind_successor_population": expected["qualified"],
            "score_or_use_retired_V146_test": False,
            "modify_retry_rescore_rerun_reprompt_tune_or_mine_V147": False,
            "touch_V134_external_language_or_run_API": False,
            "run_training_induction_authority_action_or_execution": False,
        },
    }
    for key, path in paths.items():
        outcome[key] = str(path.relative_to(PROJECT_ROOT))
        outcome[f"{key}_sha256"] = file_sha256(path)
    outcome["lock_payload_sha256"] = payload_hash(outcome)
    write_json(outcome_path, outcome)
    print(json.dumps(audit, indent=2, sort_keys=True))
    print(json.dumps({"lock": str(outcome_path.relative_to(PROJECT_ROOT)), "sha256": file_sha256(outcome_path)}, indent=2))


if __name__ == "__main__":
    main()
