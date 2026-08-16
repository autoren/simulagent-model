#!/usr/bin/env python3
"""Audit and summarize the conditional single V30/V28 integration replay."""

from __future__ import annotations

import argparse
import json

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", default="configs/v30-signed-fact-language-lock.json")
    parser.add_argument("--language-result", default="outputs/v30-signed-fact-language/evaluation/result.json")
    parser.add_argument("--result", default="outputs/v30-signed-fact-language/integration/result.json")
    parser.add_argument("--audit", default="outputs/v30-signed-fact-language/integration-post-result-audit.json")
    parser.add_argument("--markdown", default="docs/v30-integration-results.md")
    args = parser.parse_args()
    lock_path = (PROJECT_ROOT / args.lock).resolve()
    language_result_path = (PROJECT_ROOT / args.language_result).resolve()
    result_path = (PROJECT_ROOT / args.result).resolve()
    lock = json.loads(lock_path.read_text())
    result = json.loads(result_path.read_text())
    config = lock["config_payload"]
    attempt_path = PROJECT_ROOT / "outputs/v30-signed-fact-language/integration-attempt.json"
    attempt = json.loads(attempt_path.read_text())
    scores = PROJECT_ROOT / result["scores"]
    predictions = PROJECT_ROOT / result["grounding_predictions"]
    diagnostics = PROJECT_ROOT / result["episode_diagnostics"]
    current = result["current_metrics"]
    gates = config["gates"]["integration"]
    reproduced_checks = {
        "oracle_oracle_exact": result["integration"]["oracle_support_oracle_query"]["transition_set_exact_match"] >= gates["minimumOracleOracleExact"],
        "evaluation_support_exact_graph": current["evaluation_support_exact_graph"] >= gates["minimumEvaluationSupportExactGraph"],
        "frozen_support_oracle_query_exact": current["frozen_support_oracle_query_exact"] >= gates["minimumFrozenSupportOracleQueryExact"],
        "oracle_support_frozen_query_exact": current["oracle_support_frozen_query_exact"] >= gates["minimumOracleSupportFrozenQueryExact"],
        "frozen_frozen_exact": current["frozen_frozen_exact"] >= gates["minimumFrozenFrozenExact"],
        "target_program_top1": current["target_program_top1"] >= gates["minimumTargetProgramTop1"],
        "frozen_support_target_retention": current["target_retention"] >= gates["minimumFrozenSupportTargetRetention"],
        "frozen_support_empty_version_space": current["empty_version_space"] <= gates["maximumFrozenSupportEmptyVersionSpace"],
    }
    reproduced_deltas = {
        key: current[key] - result["v28_reference"][key] for key in current
    }
    expected_decision = (
        "signed_fact_v28_integration_pass_current_scope_complete"
        if all(reproduced_checks.values())
        else "signed_fact_language_pass_but_v28_integration_insufficient"
    )
    checks = {
        "lock_matches": result["protocol_lock_sha256"] == file_sha256(lock_path),
        "language_result_matches": result["language_result_sha256"] == file_sha256(language_result_path),
        "attempt_completed_once": (
            attempt["attempt_number"] == 1 and attempt["status"] == "completed"
            and attempt["result_sha256"] == file_sha256(result_path)
        ),
        "scores_intact": result["scores_sha256"] == file_sha256(scores),
        "predictions_intact": result["grounding_predictions_sha256"] == file_sha256(predictions),
        "diagnostics_intact": result["episode_diagnostics_sha256"] == file_sha256(diagnostics),
        "checks_reproduced": result["checks"] == reproduced_checks,
        "pass_reproduced": result["passed"] == all(reproduced_checks.values()),
        "deltas_reproduced": result["deltas"] == reproduced_deltas,
        "decision_reproduced": result["decision"] == expected_decision,
        "single_replay_zero_fit": (
            result["integration_replay_number"] == 1
            and result["data_access"]["model_forward_passes"] == lock["conditional_integration"]["planned_model_forward_passes"]
            and result["data_access"]["v28_integration_replays"] == 1
            and result["data_access"]["head_fits"] == 0
            and result["data_access"]["threshold_fits"] == 0
            and result["data_access"]["hyperparameter_selections"] == 0
            and result["data_access"]["adapter_training_runs"] == 0
        ),
    }
    audit = {
        "schema_version": 30,
        "experiment": "v30_integration_post_result_integrity_audit",
        "passed": all(checks.values()),
        "decision": "accept_v30_integration_result" if all(checks.values()) else "quarantine_v30_integration_result",
        "checks": checks, "reproduced_integration_checks": reproduced_checks,
        "result_sha256": file_sha256(result_path),
    }
    audit_path = (PROJECT_ROOT / args.audit).resolve()
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")

    ref = result["v28_reference"]
    lines = [
        "# V30 results: conditional signed-fact/V28 reintegration", "",
        f"Decision: `{result['decision']}`.", "",
        "The language interface passed its sealed gates, was frozen, and was replayed once on the",
        "exposed V22r2 evaluation population. Candidate alignment is deterministic; V28's",
        "episode-level marginal program selection, ontology, DSL, and executor remain unchanged.", "",
        "| Metric | V28 | V30 interface + V28 | Delta |", "|---|---:|---:|---:|",
        f"| Exact support graph | {ref['evaluation_support_exact_graph']:.3f} | {current['evaluation_support_exact_graph']:.3f} | {result['deltas']['evaluation_support_exact_graph']:+.3f} |",
        f"| Support / oracle-query execution | {ref['frozen_support_oracle_query_exact']:.3f} | {current['frozen_support_oracle_query_exact']:.3f} | {result['deltas']['frozen_support_oracle_query_exact']:+.3f} |",
        f"| Oracle-support / frozen-query execution | {ref['oracle_support_frozen_query_exact']:.3f} | {current['oracle_support_frozen_query_exact']:.3f} | {result['deltas']['oracle_support_frozen_query_exact']:+.3f} |",
        f"| Fully frozen execution | {ref['frozen_frozen_exact']:.3f} | {current['frozen_frozen_exact']:.3f} | {result['deltas']['frozen_frozen_exact']:+.3f} |",
        f"| Target-program top-1 | {ref['target_program_top1']:.3f} | {current['target_program_top1']:.3f} | {result['deltas']['target_program_top1']:+.3f} |",
        f"| Target retention | {ref['target_retention']:.3f} | {current['target_retention']:.3f} | {result['deltas']['target_retention']:+.3f} |",
        f"| Empty version spaces | {ref['empty_version_space']:.3f} | {current['empty_version_space']:.3f} | {result['deltas']['empty_version_space']:+.3f} |", "",
        f"Complete fully frozen episodes: {result['integration']['frozen_support_frozen_query']['complete_episodes']}/{result['integration']['frozen_support_frozen_query']['episodes']}.",
        f"All integration gates passed: `{str(result['passed']).lower()}`.",
        f"Post-result integrity audit: `{'pass' if audit['passed'] else 'fail'}`.", "",
    ]
    (PROJECT_ROOT / args.markdown).write_text("\n".join(lines))
    print(json.dumps(audit, indent=2, sort_keys=True))
    if not audit["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
