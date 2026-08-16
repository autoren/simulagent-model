#!/usr/bin/env python3
"""Exactly reproduce, integrity-audit, and summarize the one-shot V32 result."""

from __future__ import annotations

import argparse
import json
import numpy as np

from audit_v32_factorized_semantics import read_rows
from evaluate_v32_sealed_mlx import material_comparison
from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from v32_evaluation import family_bootstrap_delta, summarize_seed, system_summary


def jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trained-lock", default="configs/v32-trained-systems-lock.json")
    parser.add_argument("--result", default="outputs/v32-factorized-semantics/sealed-evaluation/result.json")
    parser.add_argument("--audit", default="outputs/v32-factorized-semantics/post-result-audit.json")
    parser.add_argument("--markdown", default="docs/v32-results.md")
    args = parser.parse_args()
    trained_path, result_path = (PROJECT_ROOT / args.trained_lock).resolve(), (PROJECT_ROOT / args.result).resolve()
    trained, result = json.loads(trained_path.read_text()), json.loads(result_path.read_text())
    protocol_path = PROJECT_ROOT / trained["protocol_lock"]
    protocol, config = json.loads(protocol_path.read_text()), json.loads(protocol_path.read_text())["config_payload"]
    errors = []
    if result["protocol_lock_sha256"] != file_sha256(protocol_path): errors.append("V32 protocol-lock hash mismatch")
    if result["trained_system_lock_sha256"] != file_sha256(trained_path): errors.append("V32 trained-lock hash mismatch")
    for path, expected in protocol["implementation"].items():
        if file_sha256(PROJECT_ROOT / path) != expected: errors.append(f"V32 implementation changed: {path}")
    root = result_path.parent
    for name, expected in result["prediction_artifacts"].items():
        if file_sha256(root / name) != expected: errors.append(f"V32 prediction artifact changed: {name}")
    if file_sha256(PROJECT_ROOT / result["evaluation_features"]) != result["evaluation_features_sha256"]: errors.append("V32 sealed features changed")
    rows = sorted(read_rows(PROJECT_ROOT / protocol["source"]["corpus"], ("factor_evaluation_paraphrase", "factor_evaluation_composition")), key=lambda row: row["id"])
    predictions = {system: {} for system in ("monolithic", "auxiliaryDirect", "factorizedCompiled")}
    seed_summaries = {system: {} for system in predictions}
    for system in predictions:
        for seed in config["training"]["seeds"]:
            values = jsonl(root / f"{system}-seed-{seed}-predictions.jsonl")
            predictions[system][str(seed)] = values
            seed_summaries[system][str(seed)] = summarize_seed(rows, values, config, True)
    systems = {system: system_summary(values, config) for system, values in seed_summaries.items()}
    comp_rows = [row for row in rows if row["split"] == "factor_evaluation_composition"]
    comp_predictions = {system: {seed: [row for row in values if row["split"] == "factor_evaluation_composition"] for seed, values in seeds.items()} for system, seeds in predictions.items()}
    comp_summaries = {system: {seed: summarize_seed(comp_rows, values, config, False) for seed, values in seeds.items()} for system, seeds in comp_predictions.items()}
    factor_vs_aux = family_bootstrap_delta(comp_rows, comp_predictions["auxiliaryDirect"], comp_predictions["factorizedCompiled"], config)
    aux_vs_mono = family_bootstrap_delta(comp_rows, comp_predictions["monolithic"], comp_predictions["auxiliaryDirect"], config)
    factor_values, gates = list(comp_summaries["factorizedCompiled"].values()), config["gates"]["scientificFactorization"]
    factor_checks = {
        "mean_composition_lexical_sign_accuracy": bool(np.mean([row["lexical_sign_accuracy"] for row in factor_values]) >= gates["minimumMeanCompositionLexicalSignAccuracy"]),
        "mean_composition_outer_operation_accuracy": bool(np.mean([row["outer_operation_accuracy"] for row in factor_values]) >= gates["minimumMeanCompositionOuterOperationAccuracy"]),
        "mean_composition_compiled_truth_accuracy": bool(np.mean([row["truth_status_accuracy"] for row in factor_values]) >= gates["minimumMeanCompositionCompiledTruthAccuracy"]),
        "minimum_seed_composition_compiled_truth_accuracy": bool(min(row["truth_status_accuracy"] for row in factor_values) >= gates["minimumMinimumSeedCompositionCompiledTruthAccuracy"]),
        "factorized_minus_auxiliary_composition_exact_fact": bool(factor_vs_aux["mean_exact_signed_fact_delta"] >= gates["minimumFactorizedMinusAuxiliaryCompositionExactFact"]),
        "paired_family_bootstrap_lower_bound": bool(factor_vs_aux["bootstrap_95_interval"][0] > gates["minimumPairedFamilyBootstrapLowerBound"]),
        "oracle_compiler_accuracy": bool(gates["requiredOracleCompilerAccuracy"] == 1.0),
    }
    gates = config["gates"]["scientificIntermediateSupervision"]
    intermediate_checks = {"auxiliary_minus_monolithic_composition_exact_fact": bool(aux_vs_mono["mean_exact_signed_fact_delta"] >= gates["minimumAuxiliaryMinusMonolithicCompositionExactFact"]), "paired_family_bootstrap_lower_bound": bool(aux_vs_mono["bootstrap_95_interval"][0] > gates["minimumPairedFamilyBootstrapLowerBound"])}
    selected, trace = None, []
    for candidate in ("monolithic", "auxiliaryDirect", "factorizedCompiled"):
        if not systems[candidate]["passed"]:
            trace.append({"candidate": candidate, "absolute_pass": False, "selected": False}); continue
        if selected is None:
            selected = candidate; trace.append({"candidate": candidate, "absolute_pass": True, "selected": True, "reason": "first_simplest_absolute_pass"}); continue
        comparison = material_comparison(rows, predictions[selected], predictions[candidate], config)
        replace = comparison["material_advantage"]
        trace.append({"candidate": candidate, "absolute_pass": True, "selected": replace, "challenged": selected, "comparison": comparison})
        if replace: selected = candidate
    comparisons = {
        "systems": systems == result["systems"], "composition_seed_summaries": comp_summaries == result["composition_seed_summaries"],
        "factor_checks": factor_checks == result["scientific_factorization"]["checks"], "factor_pass": all(factor_checks.values()) == result["scientific_factorization"]["passed"],
        "factor_delta": factor_vs_aux == result["scientific_factorization"]["factorized_minus_auxiliary"],
        "intermediate_checks": intermediate_checks == result["scientific_intermediate_supervision"]["checks"], "intermediate_pass": all(intermediate_checks.values()) == result["scientific_intermediate_supervision"]["passed"],
        "intermediate_delta": aux_vs_mono == result["scientific_intermediate_supervision"]["auxiliary_minus_monolithic"],
        "selection": selected == result["selected_system"], "selection_trace": trace == result["selection_trace"],
        "authorization": (selected is not None) == result["v28_integration_authorized"],
    }
    if not all(comparisons.values()): errors.append("V32 result does not exactly reproduce")
    access = result["data_access"]
    if access["frozen_feature_model_forward_passes"] != protocol["planned_evaluation"]["frozen_feature_forward_passes"]: errors.append("V32 model-forward ledger differs from lock")
    if access["sealed_evaluations"] != 1 or access["seed_selections"] or access["checkpoint_selections"] or access["hyperparameter_selections"] or access["v28_integration_replays"]: errors.append("V32 sealed access ledger violates protocol")
    audit = {"schema_version": 32, "experiment": "v32_post_result_integrity_audit", "result": str(result_path.relative_to(PROJECT_ROOT)), "result_sha256": file_sha256(result_path), "passed": not errors, "decision": "accept_v32_result" if not errors else "reject_v32_result", "errors": errors, "reproduction_checks": comparisons, "scientific_factorization_passed": all(factor_checks.values()), "scientific_intermediate_supervision_passed": all(intermediate_checks.values()), "selected_system": selected, "absolute_language_passed": selected is not None, "v28_integration_authorized": selected is not None, "observed_model_forward_passes": access["frozen_feature_model_forward_passes"], "expected_model_forward_passes": protocol["planned_evaluation"]["frozen_feature_forward_passes"]}
    audit_path = (PROJECT_ROOT / args.audit).resolve()
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    factor_mean = {key: float(np.mean([row[key] for row in factor_values])) for key in ("lexical_sign_accuracy", "outer_operation_accuracy", "truth_status_accuracy", "exact_signed_fact_accuracy")}
    lines = [
        "# V32 factorized polarity and scope", "", "## Verdict", "",
        f"Scientific factorization gate: `{'pass' if all(factor_checks.values()) else 'fail'}`. Intermediate-supervision gate: `{'pass' if all(intermediate_checks.values()) else 'fail'}`. Absolute selected system: `{selected or 'none'}`.", "",
        "## Full sealed evaluation", "", "| System | Predicate | Relation order | Truth | Exact fact | Exact scene | Absolute pass |", "|---|---:|---:|---:|---:|---:|:---:|",
        *[f"| {system} | {systems[system]['mean']['predicate_accuracy']:.3f} | {systems[system]['mean']['relation_argument_order_accuracy']:.3f} | {systems[system]['mean']['truth_status_accuracy']:.3f} | {systems[system]['mean']['exact_signed_fact_accuracy']:.3f} | {systems[system]['mean']['exact_scene_accuracy']:.3f} | {'yes' if systems[system]['passed'] else 'no'} |" for system in ("monolithic", "auxiliaryDirect", "factorizedCompiled")],
        "", "## Composition holdout", "", f"Factorized sign accuracy: {factor_mean['lexical_sign_accuracy']:.3f}.", f"Factorized operation accuracy: {factor_mean['outer_operation_accuracy']:.3f}.", f"Factorized compiled truth accuracy: {factor_mean['truth_status_accuracy']:.3f}.", f"Factorized minus auxiliary-direct exact-fact delta: {factor_vs_aux['mean_exact_signed_fact_delta']:+.3f} (family-bootstrap 95% interval [{factor_vs_aux['bootstrap_95_interval'][0]:+.3f}, {factor_vs_aux['bootstrap_95_interval'][1]:+.3f}]).", f"Auxiliary-direct minus monolithic exact-fact delta: {aux_vs_mono['mean_exact_signed_fact_delta']:+.3f} (family-bootstrap 95% interval [{aux_vs_mono['bootstrap_95_interval'][0]:+.3f}, {aux_vs_mono['bootstrap_95_interval'][1]:+.3f}]).", "",
        "## Decision integrity", "", f"One V28 replay authorized: `{str(selected is not None).lower()}`.", "Seed selection: `none`.", "Checkpoint or hyperparameter selection: `none`.", f"Post-result integrity audit: `{'pass' if audit['passed'] else 'fail'}`.", "",
    ]
    (PROJECT_ROOT / args.markdown).write_text("\n".join(lines))
    print(json.dumps(audit, indent=2, sort_keys=True))
    if not audit["passed"]: raise SystemExit(1)


if __name__ == "__main__": main()
