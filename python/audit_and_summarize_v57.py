#!/usr/bin/env python3
"""Reproduce V57 scoring from stored records without rerunning the compiler."""
from __future__ import annotations

import argparse
import json

from evaluate_v57_definition_transfer import aggregate, read_jsonl, verify_population
from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result",
        default=(
            "outputs/v57-definition-augmented-ontology-transfer/"
            "evaluation/result.json"
        ),
    )
    parser.add_argument(
        "--audit",
        default=(
            "outputs/v57-definition-augmented-ontology-transfer/"
            "post-result-audit.json"
        ),
    )
    parser.add_argument("--summary", default="docs/v57-results.md")
    args = parser.parse_args()
    result_path, audit_path, summary_path = tuple(
        (PROJECT_ROOT / value).resolve()
        for value in (args.result, args.audit, args.summary)
    )
    if audit_path.exists() or summary_path.exists():
        raise FileExistsError("V57 post-result artifacts already exist")
    result = json.loads(result_path.read_text())
    lock_path = PROJECT_ROOT / result["evaluation_implementation_lock"]
    lock = json.loads(lock_path.read_text())
    seal_path = PROJECT_ROOT / result["population_seal"]
    seal = json.loads(seal_path.read_text())
    attempt_path = result_path.parent.parent / "evaluation-attempt.json"
    errors: list[str] = []

    manifest, population_mismatches = verify_population(seal)
    sealed = (
        result["evaluation_run"] == 1
        and result["evaluation_implementation_lock_sha256"]
        == file_sha256(lock_path)
        and result["population_seal_sha256"] == file_sha256(seal_path)
        and result["manifest_sha256"]
        == file_sha256(PROJECT_ROOT / seal["manifest"])
        and lock["population_seal_sha256"] == file_sha256(seal_path)
        and seal["manifest_sha256"]
        == file_sha256(PROJECT_ROOT / seal["manifest"])
        and population_mismatches == 0
        and attempt_path.exists()
        and json.loads(attempt_path.read_text())["evaluation_run"] == 1
    )
    if not sealed:
        errors.append("V57 result is not bound to the one-shot sealed population")

    source_core = read_jsonl(PROJECT_ROOT / seal["artifacts"]["core"]["path"])
    source_safety = read_jsonl(
        PROJECT_ROOT / seal["artifacts"]["safety"]["path"]
    )
    core_records = result["records"]["core"]
    safety_records = result["records"]["safety"]
    core_order = [row["id"] for row in source_core]
    safety_order = [row["id"] for row in source_safety]
    records_ok = (
        len(core_records) == len(core_order) == 1920
        and len(safety_records) == len(safety_order) == 224
        and [row["id"] for row in core_records] == core_order
        and [row["id"] for row in safety_records] == safety_order
        and len({row["id"] for row in core_records}) == 1920
        and len({row["id"] for row in safety_records}) == 224
    )
    if not records_ok:
        errors.append("V57 record counts, order, or IDs are invalid")

    implementation_path = PROJECT_ROOT / seal["implementation_lock"]
    implementation = json.loads(implementation_path.read_text())
    config = implementation["config_payload"]
    reproduced = aggregate(
        core_records,
        safety_records,
        config,
        result["metrics"]["integrity"],
        implementation["mutation_kill_rate"],
    )
    metrics_ok = reproduced["metrics"] == result["metrics"]
    checks_ok = reproduced["checks"] == result["qualification"]["checks"]
    qualification_ok = (
        reproduced["passed"] == result["qualification"]["passed"]
    )
    if not metrics_ok:
        errors.append("V57 aggregate metrics do not reproduce")
    if not checks_ok or not qualification_ok:
        errors.append("V57 qualification does not reproduce")

    core_by_id = {row["id"]: row for row in source_core}
    safety_by_id = {row["id"]: row for row in source_safety}
    algebra_ok = all(
        row["exact_ast"]
        == (
            row["status"] == "ok"
            and row["parse"] == core_by_id[row["id"]]["target"]["parse"]
        )
        and (
            row["concept_kind"] == "bound_action"
            or row["predicate_compiled_truth_exact"] is not None
        )
        and (
            row["concept_kind"] != "bound_action"
            or row["exact_bound_action"] == row["exact_ast"]
        )
        for row in core_records
    ) and all(
        row["passed"]
        == (row["status"] in safety_by_id[row["id"]]["expected"]["statuses"])
        for row in safety_records
    )
    if not algebra_ok:
        errors.append("V57 stored record-level scoring algebra is inconsistent")

    integrity = result["metrics"]["integrity"]
    integrity_ok = all(value == 0 for value in integrity.values())
    if not integrity_ok:
        errors.append("V57 population, compiler-truth, or attempt integrity failed")

    audit = {
        "schema_version": 57,
        "experiment": "v57_post_result_audit",
        "passed": not errors,
        "decision": "accept_v57_result" if not errors else "invalidate_v57_result",
        "errors": errors,
        "result": str(result_path.relative_to(PROJECT_ROOT)),
        "result_sha256": file_sha256(result_path),
        "evaluation_attempt": str(attempt_path.relative_to(PROJECT_ROOT)),
        "evaluation_attempt_sha256": file_sha256(attempt_path),
        "evaluation_implementation_lock": str(
            lock_path.relative_to(PROJECT_ROOT)
        ),
        "evaluation_implementation_lock_sha256": file_sha256(lock_path),
        "population_seal": str(seal_path.relative_to(PROJECT_ROOT)),
        "population_seal_sha256": file_sha256(seal_path),
        "qualification": result["qualification"],
        "checks": {
            "one_shot_sealed_bindings": sealed,
            "record_count_order_and_ids": records_ok,
            "metric_aggregation_reproduced": metrics_ok,
            "qualification_reproduced": checks_ok and qualification_ok,
            "stored_record_scoring_algebra": algebra_ok,
            "population_compiler_truth_and_attempt_integrity": integrity_ok,
        },
        "data_access": {
            "additional_v57_candidate_evaluation_runs": 0,
            "compiler_invocations": 0,
            "model_forward_passes": 0,
            "human_authored_records_collected": 0,
            "stored_target_comparisons_for_audit": len(core_records),
        },
    }
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")

    core = result["metrics"]["core"]
    controls = result["metrics"]["definition_dependence_controls"]
    safety = result["metrics"]["safety"]
    if result["qualification"]["passed"]:
        decision = (
            "V57 qualifies exact definition-conditioned transfer to unseen typed "
            "symbols inside the sealed controlled definition and evidence languages. "
            "It authorizes only the separate V58 human-authored, known-ontology "
            "collection protocol."
        )
    else:
        decision = (
            "V57 does not qualify controlled definition-conditioned transfer; "
            "the failed gate must be localized in a new preregistered revision."
        )
    summary_path.write_text(
        "# V57 results: definition-augmented ontology transfer\n\n"
        f"Decision: {decision}\n\n"
        "## Sealed results\n\n"
        f"- All 15 noncompensatory gates passed: `{result['qualification']['passed']}`.\n"
        f"- Core coverage / exact AST: `{core['coverage']}` / `{core['exact_ast']}`.\n"
        f"- Predicate compiled truth: `{core['predicate_compiled_truth']}`.\n"
        f"- Exact bound action: `{core['exact_bound_action']}`.\n"
        f"- Worst pack / kind / definition family: "
        f"`{core['worst_ontology_pack_exact_ast']}` / "
        f"`{core['worst_concept_kind_exact_ast']}` / "
        f"`{core['worst_definition_template_exact_ast']}`.\n"
        f"- No-definition / opaque-name-only abstention: "
        f"`{controls['no_definition_abstention_rate']}` / "
        f"`{controls['opaque_name_only_abstention_rate']}`.\n"
        f"- Shuffled-schema exact AST: `{controls['shuffled_schema_exact_ast']}`.\n"
        f"- Safety / worst safety condition: `{safety['rate']}` / "
        f"`{safety['worst_condition_rate']}`.\n"
        f"- Mutation kill rate: "
        f"`{result['metrics']['implementation_controls']['mutation_kill_rate']}`.\n"
        f"- Integrity violations: `{sum(integrity.values())}`.\n\n"
        "## Claim boundary\n\n"
        "This result covers new opaque unary predicates, binary relations, and "
        "bound actions only when complete typed definitions are supplied in the "
        "declared controlled language and the evidence uses the familiar declared "
        "grammar. It does not establish human-authored language robustness, open "
        "definition understanding, joint new-language/new-concept transfer, or any "
        "probabilistic-inference or planning improvement.\n"
    )
    print(json.dumps(audit, indent=2, sort_keys=True))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
