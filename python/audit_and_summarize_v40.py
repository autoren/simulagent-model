#!/usr/bin/env python3
"""Reproduce, audit, and summarize the V40 confirmation."""

from __future__ import annotations

import argparse
import json

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from evaluate_v40_confirmation import read, score_confirmation, score_safety


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", default="outputs/v40-independent-compiler-confirmation/evaluation/result.json")
    parser.add_argument("--audit", default="outputs/v40-independent-compiler-confirmation/post-result-audit.json")
    parser.add_argument("--markdown", default="docs/v40-results.md")
    args = parser.parse_args()
    result_path, audit_path, markdown_path = tuple((PROJECT_ROOT / value).resolve() for value in (args.result, args.audit, args.markdown))
    result = json.loads(result_path.read_text())
    seal_path = PROJECT_ROOT / result["corpus_seal"]
    seal = json.loads(seal_path.read_text())
    implementation_path = PROJECT_ROOT / seal["implementation_lock"]
    implementation = json.loads(implementation_path.read_text())
    core = read(PROJECT_ROOT / seal["corpora"]["independent_confirmation"]["path"])
    safety_rows = read(PROJECT_ROOT / seal["corpora"]["independent_safety"]["path"])
    confirmation, _ = score_confirmation(core, implementation["v32_config_payload"])
    safety, _ = score_safety(safety_rows)
    reproduction = {
        "confirmation": confirmation == result["confirmation"],
        "safety": safety == result["safety"],
        "predictions_hash": file_sha256(PROJECT_ROOT / result["predictions"]) == result["predictions_sha256"],
    }
    access = result["data_access"]
    integrity = {
        "one_confirmation": access["confirmation_evaluations"] == 1,
        "no_confirmation_selection": access["selection_on_confirmation"] == 0,
        "no_model_access": access["model_forward_passes"] == 0,
        "no_v32_evaluation": access["v32_evaluation_records_read"] == 0,
        "no_v28": access["v28_runs"] == 0,
        "no_adapter_training": access["adapter_training_runs"] == 0,
        "compiler_unchanged": file_sha256(PROJECT_ROOT / implementation["frozen_compiler"]) == implementation["frozen_compiler_sha256"],
        "implementation_bound": seal["implementation_lock_sha256"] == file_sha256(implementation_path),
        "seal_bound": result["corpus_seal_sha256"] == file_sha256(seal_path),
    }
    errors = []
    if not all(reproduction.values()):
        errors.append("V40 result does not reproduce")
    if not all(integrity.values()):
        errors.append("V40 firewall or lock chain failed")
    audit = {
        "schema_version": 40,
        "experiment": "v40_post_result_audit",
        "passed": not errors,
        "decision": "accept_v40_confirmation" if not errors else "reject_v40_confirmation",
        "errors": errors,
        "result": str(result_path.relative_to(PROJECT_ROOT)),
        "result_sha256": file_sha256(result_path),
        "reproduction_checks": reproduction,
        "integrity_checks": integrity,
        "scientific_decision": result["decision"],
    }
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    confirmation_result = result["confirmation"]
    safety_result = result["safety"]
    lines = [
        "# V40 results: independent declared-language confirmation",
        "",
        f"Decision: `{result['decision']}`.",
        "",
        "The unchanged V39 compiler was tested once on an independently generated, hash-sealed population with 12 new declared ontologies and new operator cues.",
        "",
        "| Metric | Result |",
        "|---|---:|",
        f"| Confirmation coverage | {confirmation_result['coverage']:.3f} |",
        f"| Confirmation exact parse | {confirmation_result['exact_parse']:.3f} |",
        f"| Confirmation compiled truth | {confirmation_result['compiled_truth']:.3f} |",
        f"| Worst ontology pack | {confirmation_result['minimum_ontology_pack_exact_parse']:.3f} |",
        f"| Worst operation/sign cell | {confirmation_result['minimum_operation_sign_exact_parse']:.3f} |",
        f"| Overall safety | {safety_result['safe_rate']:.3f} |",
        f"| Worst safety condition | {safety_result['minimum_condition_safe_rate']:.3f} |",
        "",
        f"All preregistered gates passed: `{str(result['qualification']['passed']).lower()}`.",
        "",
        "Interpretation: exact declared-language compilation is now supported as a stable interface component, including transfer to fresh declared lexicons. This does not establish open-language understanding. The next justified experiment is a preregistered end-to-end relational-mechanic confirmation using the frozen compiler, probabilistic graph/program inference, and the already declared relational scope.",
        "",
        f"Post-result integrity audit: `{'pass' if audit['passed'] else 'fail'}`.",
    ]
    markdown_path.write_text("\n".join(lines) + "\n")
    print(json.dumps(audit, indent=2, sort_keys=True))
    if not audit["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
