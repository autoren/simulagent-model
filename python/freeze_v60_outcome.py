#!/usr/bin/env python3
"""Freeze the audited V60 outcome."""
from __future__ import annotations

import argparse
import hashlib
import json

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--audit", default="outputs/v60-approximate-belief-decision-calibration/post-result-audit.json"
    )
    parser.add_argument("--summary", default="docs/v60-results.md")
    parser.add_argument("--output", default="configs/v60-outcome-lock.json")
    args = parser.parse_args()
    audit_path, summary_path, output = tuple(
        (PROJECT_ROOT / value).resolve()
        for value in (args.audit, args.summary, args.output)
    )
    if output.exists():
        raise RuntimeError("V60 outcome is already frozen")
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
        raise RuntimeError("V60 post-result audit is not intact and bound")
    result = json.loads(result_path.read_text())
    outcome = {
        "schema_version": 60,
        "experiment": "v60_outcome_lock",
        "qualification_passed": result["qualification"]["passed"],
        "result": audit["result"], "result_sha256": file_sha256(result_path),
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
            "frozen_smc2_to_budgeted_search_composition": result["qualification"]["passed"],
            "exact_long_horizon_optimality": False,
            "general_purpose_or_amortized_inference": False,
            "formal_or_worst_case_safety": False,
            "human_authored_language_robustness": False,
            "model_or_adapter_performance": False,
        },
        "authorization": {
            "rerun_v60_candidate_evaluation": False,
            "modify_v60_design_implementation_or_gates": False,
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
