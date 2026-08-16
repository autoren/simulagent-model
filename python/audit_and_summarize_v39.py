#!/usr/bin/env python3
"""Reproduce, audit, and summarize the V39 result."""

from __future__ import annotations

import argparse
import json

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from evaluate_v39_compiler import read, score_paraphrases, score_safety, score_supported


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", default="outputs/v39-declared-language-compiler/evaluation/result.json")
    parser.add_argument("--audit", default="outputs/v39-declared-language-compiler/post-result-audit.json")
    parser.add_argument("--markdown", default="docs/v39-results.md")
    args = parser.parse_args()
    result_path, audit_path, markdown_path = tuple((PROJECT_ROOT / value).resolve() for value in (args.result, args.audit, args.markdown))
    result = json.loads(result_path.read_text())
    seal_path = PROJECT_ROOT / result["corpus_seal"]
    seal = json.loads(seal_path.read_text())
    implementation_path = PROJECT_ROOT / seal["implementation_lock"]
    implementation = json.loads(implementation_path.read_text())
    supported = read(PROJECT_ROOT / seal["corpora"]["supported_evaluation"]["path"])
    safety = read(PROJECT_ROOT / seal["corpora"]["compiler_safety"]["path"])
    paraphrases = read(PROJECT_ROOT / seal["corpora"]["novel_paraphrase_diagnostic"]["path"])
    supported_metrics, _ = score_supported(supported, implementation["v32_config_payload"])
    safety_metrics, _ = score_safety(safety)
    paraphrase_metrics, _ = score_paraphrases(paraphrases)
    reproduction = {
        "supported": supported_metrics == result["supported"],
        "safety": safety_metrics == result["safety"],
        "novel_paraphrase": paraphrase_metrics == result["novel_paraphrase_diagnostic"],
        "predictions_hash": file_sha256(PROJECT_ROOT / result["predictions"]) == result["predictions_sha256"],
    }
    access = result["data_access"]
    integrity = {
        "one_evaluation": access["evaluation_attempts"] == 1,
        "no_evaluation_selection": access["selection_on_evaluation"] == 0,
        "no_model_access": access["model_forward_passes"] == 0,
        "no_v32_evaluation": access["v32_evaluation_records_read"] == 0,
        "no_v28": access["v28_runs"] == 0,
        "no_adapter_training": access["adapter_training_runs"] == 0,
        "implementation_bound": seal["implementation_lock_sha256"] == file_sha256(implementation_path),
        "seal_bound": result["corpus_seal_sha256"] == file_sha256(seal_path),
    }
    errors = []
    if not all(reproduction.values()):
        errors.append("V39 result does not reproduce")
    if not all(integrity.values()):
        errors.append("V39 firewall or lock chain failed")
    audit = {
        "schema_version": 39,
        "experiment": "v39_post_result_audit",
        "passed": not errors,
        "decision": "accept_v39_result" if not errors else "reject_v39_result",
        "errors": errors,
        "result": str(result_path.relative_to(PROJECT_ROOT)),
        "result_sha256": file_sha256(result_path),
        "reproduction_checks": reproduction,
        "integrity_checks": integrity,
        "scientific_decision": result["decision"],
    }
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    supported_result = result["supported"]
    safety_result = result["safety"]
    novel = result["novel_paraphrase_diagnostic"]
    lines = [
        "# V39 results: declared-language compiler",
        "",
        f"Decision: `{result['decision']}`.",
        "",
        "V39 tests exact compilation within an explicitly declared controlled language. It does not test open-ended paraphrase understanding.",
        "",
        "| Primary metric | Result |",
        "|---|---:|",
        f"| Supported coverage | {supported_result['coverage']:.3f} |",
        f"| Supported exact parse | {supported_result['exact_parse']:.3f} |",
        f"| Supported compiled truth | {supported_result['compiled_truth']:.3f} |",
        f"| Worst held-out composition cell | {supported_result['minimum_composition_cell_exact_parse']:.3f} |",
        f"| Malformed-input abstention | {safety_result['malformed_abstention']:.3f} |",
        f"| Unknown-lexeme abstention | {safety_result['unknown_lexeme_abstention']:.3f} |",
        f"| Ambiguity safety | {safety_result['ambiguity_safety']:.3f} |",
        "",
        f"All preregistered gates passed: `{str(result['qualification']['passed']).lower()}`.",
        f"The non-gating novel-paraphrase exact parse rate was {novel['reference_exact_parse_non_gating']:.3f}; this records the declared scope boundary.",
        "",
        "Interpretation: the remaining V38 failure was an interface problem inside the tested controlled-language scope. V39 shows that once predicate and operator language are both declared, the existing symbolic representation can receive exact, safely compiled semantics. The next claim still requires a fresh preregistered confirmation population.",
        "",
        f"Post-result integrity audit: `{'pass' if audit['passed'] else 'fail'}`.",
    ]
    markdown_path.write_text("\n".join(lines) + "\n")
    print(json.dumps(audit, indent=2, sort_keys=True))
    if not audit["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
