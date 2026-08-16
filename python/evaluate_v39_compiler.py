#!/usr/bin/env python3
"""Perform the single locked V39 compiler and safety evaluation."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import time

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from v32_language import compile_truth
from v39_compiler import compile_agent_input


def read(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _rate(values):
    return sum(values) / len(values) if values else 0.0


def score_supported(rows, v32):
    predictions = []
    coverage = []
    exact = []
    truth_exact = []
    by_cell = defaultdict(list)
    for row in rows:
        result = compile_agent_input(row["agent_input"])
        covered = result.get("status") == "ok"
        parse_exact = covered and result.get("parse") == row["target"]["parse"]
        compiled_truth = None
        if covered:
            parse = result["parse"]
            compiled_truth = compile_truth(parse["lexical_sign"], parse["outer_operation"], v32)
        truth_ok = parse_exact and compiled_truth == row["target"]["truth_status"]
        coverage.append(covered)
        exact.append(parse_exact)
        truth_exact.append(truth_ok)
        by_cell[row["oracle_metadata"]["composition_cell"]].append(parse_exact)
        predictions.append({"id": row["id"], "split": row["split"], "result": result, "compiled_truth": compiled_truth})
    cell_rates = {cell: _rate(values) for cell, values in sorted(by_cell.items())}
    metrics = {
        "records": len(rows),
        "coverage": _rate(coverage),
        "exact_parse": _rate(exact),
        "compiled_truth": _rate(truth_exact),
        "composition_cells": len(cell_rates),
        "minimum_composition_cell_exact_parse": min(cell_rates.values()),
        "composition_cell_exact_parse": cell_rates,
    }
    return metrics, predictions


def score_safety(rows):
    predictions = []
    by_kind = defaultdict(list)
    for row in rows:
        result = compile_agent_input(row["agent_input"])
        safe = result.get("status") in row["expected"]["statuses"]
        by_kind[row["expected"]["challenge_kind"]].append(safe)
        predictions.append({"id": row["id"], "split": row["split"], "result": result, "safe": safe})
    rates = {kind: _rate(values) for kind, values in sorted(by_kind.items())}
    metrics = {
        "records": len(rows),
        "by_challenge": rates,
        "malformed_abstention": rates["malformed_declared_grammar"],
        "ambiguity_safety": rates["two_equally_marked_focus_literals"],
        "unknown_predicate_abstention": rates["unknown_predicate_lexeme"],
        "unknown_operator_abstention": rates["unknown_operator_cue"],
        "unknown_lexeme_abstention": min(rates["unknown_predicate_lexeme"], rates["unknown_operator_cue"]),
    }
    return metrics, predictions


def score_paraphrases(rows):
    predictions = []
    statuses = Counter()
    reference_exact = []
    for row in rows:
        result = compile_agent_input(row["agent_input"])
        statuses[result.get("status", "missing")] += 1
        reference_exact.append(result.get("status") == "ok" and result.get("parse") == row["expected"]["reference_parse"])
        predictions.append({"id": row["id"], "split": row["split"], "result": result})
    return {
        "records": len(rows),
        "status_counts": dict(sorted(statuses.items())),
        "reference_exact_parse_non_gating": _rate(reference_exact),
    }, predictions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus-seal", default="configs/v39-corpus-seal.json")
    parser.add_argument("--output-dir", default="outputs/v39-declared-language-compiler/evaluation")
    args = parser.parse_args()
    seal_path = (PROJECT_ROOT / args.corpus_seal).resolve()
    output = (PROJECT_ROOT / args.output_dir).resolve()
    attempt = output.parent / "evaluation-attempt.json"
    if output.exists() or attempt.exists():
        raise RuntimeError("V39 evaluation already attempted")
    seal = json.loads(seal_path.read_text())
    implementation_path = PROJECT_ROOT / seal["implementation_lock"]
    implementation = json.loads(implementation_path.read_text())
    for path, expected in implementation["implementation"].items():
        if file_sha256(PROJECT_ROOT / path) != expected:
            raise RuntimeError(f"V39 implementation changed: {path}")
    for artifact in seal["corpora"].values():
        if file_sha256(PROJECT_ROOT / artifact["path"]) != artifact["sha256"]:
            raise RuntimeError(f"V39 sealed corpus changed: {artifact['path']}")
    output.parent.mkdir(parents=True, exist_ok=True)
    attempt.write_text(json.dumps({"schema_version": 39, "status": "started", "evaluation_attempt": 1, "corpus_seal_sha256": file_sha256(seal_path)}, indent=2, sort_keys=True) + "\n")
    started = time.perf_counter()
    supported = read(PROJECT_ROOT / seal["corpora"]["supported_evaluation"]["path"])
    safety = read(PROJECT_ROOT / seal["corpora"]["compiler_safety"]["path"])
    paraphrases = read(PROJECT_ROOT / seal["corpora"]["novel_paraphrase_diagnostic"]["path"])
    supported_metrics, supported_predictions = score_supported(supported, implementation["v32_config_payload"])
    safety_metrics, safety_predictions = score_safety(safety)
    paraphrase_metrics, paraphrase_predictions = score_paraphrases(paraphrases)
    config = implementation["config_payload"]
    gates = config["gates"]
    checks = {
        "supported_coverage": supported_metrics["coverage"] >= gates["minimumSupportedCoverage"],
        "supported_exact_parse": supported_metrics["exact_parse"] >= gates["minimumSupportedExactParse"],
        "supported_compiled_truth": supported_metrics["compiled_truth"] >= gates["minimumSupportedCompiledTruth"],
        "every_composition_cell": supported_metrics["minimum_composition_cell_exact_parse"] >= gates["minimumEveryCompositionCellExactParse"],
        "malformed_abstention": safety_metrics["malformed_abstention"] >= gates["minimumMalformedAbstention"],
        "unknown_lexeme_abstention": safety_metrics["unknown_lexeme_abstention"] >= gates["minimumUnknownLexemeAbstention"],
        "ambiguity_safety": safety_metrics["ambiguity_safety"] >= gates["minimumAmbiguitySafety"],
    }
    passed = all(checks.values())
    if passed:
        decision = "declared_language_compiler_pass_preregister_fresh_confirmation"
    elif not checks["malformed_abstention"] or not checks["unknown_lexeme_abstention"] or not checks["ambiguity_safety"]:
        decision = "repair_compiler_safety_before_confirmation"
    else:
        decision = "repair_declared_grammar_or_compiler_before_confirmation"
    output.mkdir(parents=True, exist_ok=False)
    prediction_path = output / "predictions.jsonl"
    all_predictions = supported_predictions + safety_predictions + paraphrase_predictions
    prediction_path.write_text("".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in all_predictions))
    result = {
        "schema_version": 39,
        "experiment": config["experiment"],
        "corpus_seal": str(seal_path.relative_to(PROJECT_ROOT)),
        "corpus_seal_sha256": file_sha256(seal_path),
        "evaluation_number": 1,
        "supported": supported_metrics,
        "safety": safety_metrics,
        "novel_paraphrase_diagnostic": paraphrase_metrics,
        "qualification": {"passed": passed, "checks": checks},
        "decision": decision,
        "predictions": str(prediction_path.relative_to(PROJECT_ROOT)),
        "predictions_sha256": file_sha256(prediction_path),
        "runtime_seconds": time.perf_counter() - started,
        "secondary_comparisons": {"run": False, "reason": "no selection role and no new model access required for the primary compiler claim"},
        "data_access": {
            "evaluation_attempts": 1,
            "supported_records_scored": len(supported),
            "safety_records_scored": len(safety),
            "novel_paraphrase_records_scored": len(paraphrases),
            "model_forward_passes": 0,
            "selection_on_evaluation": 0,
            "v32_evaluation_records_read": 0,
            "v28_runs": 0,
            "adapter_training_runs": 0,
        },
        "authorization": {
            "preregister_fresh_supported_language_confirmation": passed,
            "construct_confirmation": False,
            "end_to_end_relational_suite": False,
            "v32_evaluation": False,
            "v28": False,
            "adapter_training": False,
            "change_backbone": False,
        },
    }
    result_path = output / "result.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    state = json.loads(attempt.read_text())
    state.update({"status": "completed", "result_sha256": file_sha256(result_path)})
    attempt.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
