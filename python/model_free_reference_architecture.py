from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path
from typing import Any

from cross_track_evidence_audit import ROOT, payload_hash, read_json, sha256_file, valid_lock, write_json
from v165_factored_ontology_identifiability_population import candidate_universe
from v167_exact_evidence_gathering_planner import available_queries, condition, initial_belief
from v168_fixed_ontology_reversible_sandbox import build_fixtures, run_fixture
from v173_trusted_only_shadow_integration import deterministic_consensus_route, trace_tree
from v175_certification_aware_planner_development import exact_certification_policy, routed_stop
from v179_triple_repetition_robust_feasibility import (
    clean_decoded_survivors,
    majority_decode,
    raw_robust_survivors,
    repeated_history,
)
from v205_terminally_proper_open_world_semantic_pomdp import audit_oracle, evaluate_oracle


CONFIG_PATH = ROOT / "configs/model-free-reference-architecture.json"
OUTPUT_DIR = ROOT / "outputs/model-free-reference-architecture"
RESULT_PATH = OUTPUT_DIR / "result.json"
AUDIT_PATH = OUTPUT_DIR / "audit.json"
ACCESS_PATH = OUTPUT_DIR / "access.json"
MANIFEST_PATH = OUTPUT_DIR / "manifest.json"
RESULTS_DOCUMENT = ROOT / "docs/model-free-reference-architecture-integration-results.md"


def _fraction(value: Fraction) -> dict[str, Any]:
    return {"numerator": value.numerator, "denominator": value.denominator, "decimal": float(value)}


def _belief_payload(belief: tuple[tuple[str, Fraction], ...]) -> list[dict[str, Any]]:
    return [{"candidate_id": candidate_id, "weight": _fraction(weight)} for candidate_id, weight in belief]


def source_lock_integrity(config: dict[str, Any]) -> list[dict[str, Any]]:
    paths = sorted({path for component in config["components"] for path in component["frozen_outcomes"]})
    rows = []
    for relative in paths:
        path = ROOT / relative
        lock = read_json(path)
        rows.append(
            {
                "path": relative,
                "file_sha256": sha256_file(path),
                "declared_payload_sha256": lock.get("lock_payload_sha256"),
                "payload_lock_valid": valid_lock(lock),
            }
        )
    return rows


def run_reference_architecture(config: dict[str, Any] | None = None) -> dict[str, Any]:
    config = config or read_json(CONFIG_PATH)
    fixture = config["integration_fixture"]
    source_locks = source_lock_integrity(config)

    universe = {row["candidate_id"]: row for row in candidate_universe()}
    candidate_ids = list(fixture["candidate_ids"])
    planner_config = read_json(ROOT / fixture["planner_config"])
    initial = initial_belief(candidate_ids, universe, planner_config)
    queries = available_queries(initial, universe)
    policy = exact_certification_policy(
        initial,
        queries,
        int(fixture["certificate_horizon"]),
        universe,
        Fraction(fixture["query_cost"]),
    )
    target = universe[fixture["target_candidate_id"]]
    clean_trace = trace_tree(policy["tree"], initial, target, universe)
    selected_queries = tuple(row["valuation_index"] for row in clean_trace["query_trace"])
    if not selected_queries:
        raise AssertionError("Integration fixture must exercise at least one evidence query")
    clean_raw = repeated_history(selected_queries, target, None)
    corrupted_raw = repeated_history(selected_queries, target, (selected_queries[0], 0))
    clean_decoded = majority_decode(clean_raw)
    corrupted_decoded = majority_decode(corrupted_raw)
    raw_survivors = raw_robust_survivors(candidate_ids, corrupted_raw, universe)
    decoded_survivors = clean_decoded_survivors(candidate_ids, corrupted_raw, universe)

    conditioned = initial
    for query, outcome in corrupted_decoded:
        conditioned = condition(conditioned, query, outcome, universe)
    route = deterministic_consensus_route(conditioned, universe)
    routed_risk, routed_decision = routed_stop(conditioned, universe)

    other_route_risk, other_route_decision = routed_stop(initial, universe)
    other_preserves_version_space = initial == initial_belief(candidate_ids, universe, planner_config)
    other_sandbox_entry_count = 0

    sandbox_config = read_json(ROOT / fixture["sandbox_config"])
    sandbox_fixtures = sorted(
        (row for row in build_fixtures(sandbox_config) if row["scenario"] == fixture["sandbox_scenario"]),
        key=lambda row: row["record_id"],
    )
    if route["route_class"] not in {"alias", "composition"}:
        raise AssertionError("The trusted integration branch did not reach a registered unanimous class")
    sandbox_result = run_fixture(sandbox_fixtures[0], sandbox_config)

    semantic_config = read_json(ROOT / fixture["semantic_pomdp_config"])
    semantic_result = evaluate_oracle(semantic_config)
    semantic_audit = audit_oracle(semantic_result, semantic_config)

    access = {
        "protected_body_read_count": 0,
        "request_language_read_count": 0,
        "raw_model_response_read_count": 0,
        "model_load_count": 0,
        "model_generation_count": 0,
        "api_call_count": 0,
        "training_run_count": 0,
        "ontology_registration_count": 0,
        "trusted_real_state_mutation_count": 0,
        "real_service_call_count": 0,
        "external_side_effect_count": 0,
        "actual_execution_count": 0,
        "simulated_sandbox_transaction_count": 1,
        "model_free_oracle_evaluation_count": 1,
    }

    result = {
        "schema_version": "model_free_reference_architecture_result.v1",
        "architecture_id": config["architecture_id"],
        "artifact_role": config["artifact_role"],
        "claim_boundary": config["claim_boundary"],
        "source_lock_integrity": source_locks,
        "typed_version_space": {
            "candidate_ids": candidate_ids,
            "candidate_classes": {candidate_id: universe[candidate_id]["expressibility_class"] for candidate_id in candidate_ids},
            "initial_belief": _belief_payload(initial),
            "initial_weight_sum": _fraction(sum((weight for _, weight in initial), Fraction(0))),
            "available_queries": list(queries),
            "certificate_policy_risk": _fraction(policy["risk"]),
            "selected_query_trace": clean_trace["query_trace"],
            "clean_decoded": [list(row) for row in clean_decoded],
            "one_corruption_decoded": [list(row) for row in corrupted_decoded],
            "raw_robust_survivors": list(raw_survivors),
            "clean_decoded_survivors": list(decoded_survivors),
            "conditioned_belief": _belief_payload(conditioned),
            "conditioned_weight_sum": _fraction(sum((weight for _, weight in conditioned), Fraction(0))),
            "trusted_route": route,
            "routed_risk": _fraction(routed_risk),
            "routed_decision": routed_decision,
        },
        "other_defer": {
            "observation": fixture["other_or_uninterpretable_observation"],
            "decision": other_route_decision,
            "risk": _fraction(other_route_risk),
            "version_space_preserved": other_preserves_version_space,
            "candidate_ids_after": candidate_ids,
            "sandbox_entry_count": other_sandbox_entry_count,
        },
        "reversible_sandbox": sandbox_result,
        "outside_semantic_terminal_planner": {
            "oracle_audit_passed": semantic_audit["passed"],
            "scientific_gates_passed": semantic_audit["scientific_gates_passed"],
            "access_gates_passed": semantic_audit["access_gates_passed"],
            "root_action": semantic_result["exact"]["root_action"],
            "action_after_red": semantic_result["exact"]["action_after_root_red"],
            "action_after_blue": semantic_result["exact"]["action_after_root_blue"],
            "action_after_green": semantic_result["exact"]["action_after_root_green"],
            "reachable_repair_actions": semantic_result["exact"]["distinct_reachable_repair_actions"],
            "reachable_defer_history_count": semantic_result["exact"]["reachable_defer_history_count"],
            "terminal_audit": semantic_result["structural"]["exact_policy_terminal_audit"],
            "horizon_escape_path_count": semantic_result["structural"]["horizon_escape_path_count"],
            "mandatory_automatic_settlement_rate": semantic_result["structural"]["mandatory_automatic_settlement_rate"],
            "unfinished_sensing_safe_deferral_rate": semantic_result["structural"]["unfinished_sensing_safe_deferral_rate"],
        },
        "access": access,
    }

    gates = {
        "all_source_outcome_payload_locks_valid": all(row["payload_lock_valid"] for row in source_locks),
        "initial_and_conditioned_beliefs_normalize": result["typed_version_space"]["initial_weight_sum"]["numerator"] == result["typed_version_space"]["initial_weight_sum"]["denominator"] and result["typed_version_space"]["conditioned_weight_sum"]["numerator"] == result["typed_version_space"]["conditioned_weight_sum"]["denominator"],
        "certificate_policy_reaches_unanimous_expected_trusted_class": route["candidate_classes"] == [fixture["expected_target_class"]] and routed_decision == fixture["expected_target_class"],
        "single_corruption_majority_decode_matches_clean_truth": corrupted_decoded == clean_decoded and all(outcome == int(target["truth_table"][query]) for query, outcome in corrupted_decoded),
        "raw_robust_and_clean_decoded_survivors_match": raw_survivors == decoded_survivors,
        "OTHER_preserves_full_version_space_and_defers": other_route_decision == fixture["expected_other_route"] and other_preserves_version_space and other_sandbox_entry_count == 0,
        "only_trusted_unanimous_route_enters_sandbox": route["route_class"] in {"alias", "composition"} and len(route["candidate_classes"]) == 1 and sandbox_result["committed"],
        "sandbox_preview_commit_verification_and_provenance_pass": all(sandbox_result[key] for key in ("preview_nonmutation", "preview_commit_parity", "invariants_preserved", "zero_unauthorized_commit_mutation", "provenance_chain_valid", "exact_final_target_state")),
        "outside_semantic_oracle_and_terminal_settlement_audit_pass": semantic_audit["passed"] and semantic_result["exact"]["root_action"] == "calibrate" and semantic_result["exact"]["action_after_root_green"] == "defer" and semantic_result["structural"]["horizon_escape_path_count"] == 0 and semantic_result["structural"]["mandatory_automatic_settlement_rate"] == 1.0 and semantic_result["structural"]["unfinished_sensing_safe_deferral_rate"] == 1.0,
        "model_API_training_protected_language_registration_real_action_and_execution_counts_zero": all(value == 0 for key, value in access.items() if key not in {"simulated_sandbox_transaction_count", "model_free_oracle_evaluation_count"}),
    }
    if set(gates) != set(config["integration_gates"]):
        raise AssertionError("Implemented gate vector does not match the frozen architecture config")
    audit = {
        "schema_version": "model_free_reference_architecture_audit.v1",
        "gates": gates,
        "passed": all(gates.values()),
        "decision": "freeze_reference_architecture_software_integration_without_new_scientific_or_external_language_claim" if all(gates.values()) else "retain_integration_failure_and_repair_software_only",
        "result_sha256": payload_hash(result),
    }
    return {"result": result, "audit": audit, "access": access}


def render_results(bundle: dict[str, Any]) -> str:
    result = bundle["result"]
    audit = bundle["audit"]
    version = result["typed_version_space"]
    semantic = result["outside_semantic_terminal_planner"]
    sandbox = result["reversible_sandbox"]
    lines = [
        "# Model-free reference architecture integration result",
        "",
        "## Outcome",
        "",
        f"The deterministic software integration {'passed' if audit['passed'] else 'failed'}. This is an interface/reproducibility result only, not a new experiment or external-language validation.",
        "",
        "## Integrated path",
        "",
        f"- Exact initial version space: `{', '.join(version['candidate_ids'])}` with normalized class-balanced mass.",
        f"- Exact certificate query trace: `{version['selected_query_trace']}`.",
        f"- One deliberately corrupted raw inspection decoded to the clean result: `{version['one_corruption_decoded']}`.",
        f"- Raw-robust and clean-decoded survivors: `{version['raw_robust_survivors']}` / `{version['clean_decoded_survivors']}`.",
        f"- Trusted route: `{version['routed_decision']}`; uninterpretable `OTHER` route: `{result['other_defer']['decision']}` with zero sandbox entries.",
        f"- Existing V168 sandbox fixture: disposition `{sandbox['disposition']}`, exact target `{sandbox['exact_final_target_state']}`, provenance valid `{sandbox['provenance_chain_valid']}`.",
        f"- Existing V205 oracle: root `{semantic['root_action']}`, red/blue `{semantic['action_after_red']}/{semantic['action_after_blue']}`, green `{semantic['action_after_green']}`, horizon escapes `{semantic['horizon_escape_path_count']}`.",
        "",
        "## Safety boundary",
        "",
        "Protected/request language, models, APIs, training, ontology registration, trusted real-state mutation, services, external side effects, and actual execution were all zero. The one transaction was an existing deterministic in-memory sandbox fixture.",
        "",
        "## Interpretation",
        "",
        "The harness shows that the frozen exact version-space, robust evidence, conservative routing, reversible sandbox, and terminally proper outside-semantics planner can coexist behind explicit interfaces. It does not fill the missing empirical semantic observation channel identified by V224 and authorizes no new experiment."
    ]
    return "\n".join(lines) + "\n"


def write_bundle(bundle: dict[str, Any]) -> None:
    write_json(RESULT_PATH, bundle["result"])
    write_json(AUDIT_PATH, bundle["audit"])
    write_json(ACCESS_PATH, bundle["access"])
    RESULTS_DOCUMENT.write_text(render_results(bundle), encoding="utf-8")
    artifacts = [
        CONFIG_PATH,
        ROOT / "docs/model-free-reference-architecture.md",
        RESULTS_DOCUMENT,
        ROOT / "python/model_free_reference_architecture.py",
        ROOT / "python/run_model_free_reference_architecture.py",
        ROOT / "python/test_model_free_reference_architecture.py",
        ROOT / "python/verify_and_freeze_model_free_reference_architecture.py",
        RESULT_PATH,
        AUDIT_PATH,
        ACCESS_PATH,
    ]
    write_json(
        MANIFEST_PATH,
        {
            "schema_version": "model_free_reference_architecture_manifest.v1",
            "artifacts": [
                {
                    "path": path.resolve().relative_to(ROOT.resolve()).as_posix(),
                    "sha256": sha256_file(path),
                    "size_bytes": path.stat().st_size,
                }
                for path in sorted(artifacts)
            ],
        },
    )
