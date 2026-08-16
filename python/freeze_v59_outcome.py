#!/usr/bin/env python3
"""Freeze the audited V59 result and its scientific claim boundary."""
from __future__ import annotations

import argparse
import hashlib
import json

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--audit",
        default="outputs/v59-budgeted-root-sampled-planning/post-result-audit.json",
    )
    parser.add_argument("--summary", default="docs/v59-results.md")
    parser.add_argument("--output", default="configs/v59-outcome-lock.json")
    args = parser.parse_args()
    audit_path, summary_path, output = tuple(
        (PROJECT_ROOT / value).resolve()
        for value in (args.audit, args.summary, args.output)
    )
    if output.exists():
        raise RuntimeError("V59 outcome is already frozen")
    audit = json.loads(audit_path.read_text())
    result_path = PROJECT_ROOT / audit["result"]
    attempt_path = PROJECT_ROOT / audit["attempt"]
    evaluation_lock_path = PROJECT_ROOT / audit["evaluation_implementation_lock"]
    if (
        not audit["passed"]
        or audit["result_sha256"] != file_sha256(result_path)
        or audit["attempt_sha256"] != file_sha256(attempt_path)
        or audit["evaluation_implementation_lock_sha256"]
        != file_sha256(evaluation_lock_path)
    ):
        raise RuntimeError("V59 post-result audit is not intact and bound")
    result = json.loads(result_path.read_text())
    outcome = {
        "schema_version": 59,
        "experiment": "v59_outcome_lock",
        "qualification_passed": result["qualification"]["passed"],
        "result": audit["result"],
        "result_sha256": file_sha256(result_path),
        "evaluation_attempt": audit["attempt"],
        "evaluation_attempt_sha256": file_sha256(attempt_path),
        "post_result_audit": str(audit_path.relative_to(PROJECT_ROOT)),
        "post_result_audit_sha256": file_sha256(audit_path),
        "results_summary": str(summary_path.relative_to(PROJECT_ROOT)),
        "results_summary_sha256": file_sha256(summary_path),
        "evaluation_implementation_lock": audit["evaluation_implementation_lock"],
        "evaluation_implementation_lock_sha256": file_sha256(evaluation_lock_path),
        "population_seal": audit["population_seal"],
        "population_seal_sha256": audit["population_seal_sha256"],
        "claim_boundary": {
            "budgeted_root_sampled_search_on_exact_belief": result["qualification"]["passed"],
            "exact_long_horizon_optimality": False,
            "approximate_belief_correctness": False,
            "formal_or_worst_case_safety": False,
            "human_authored_language_robustness": False,
            "model_or_adapter_performance": False,
        },
        "authorization": {
            "rerun_v59_candidate_evaluation": False,
            "modify_v59_population_or_evaluator": False,
            "treat_synthetic_v58_as_human": False,
            "continue_to_next_preregistered_stage": True,
        },
    }
    outcome["lock_payload_sha256"] = hashlib.sha256(
        json.dumps(outcome, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    output.write_text(json.dumps(outcome, indent=2, sort_keys=True) + "\n")
    print(json.dumps(outcome, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
