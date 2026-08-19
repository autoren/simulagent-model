#!/usr/bin/env python3
"""Independently reproduce and freeze the V83 model-free interface outcome."""
from __future__ import annotations

import hashlib
import json
from typing import Any

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT


CODES = ("ask_operation", "ask_recipient", "ask_full_details")
STYLES = (
    "concise", "neutral", "polite", "formal",
    "direct", "friendly", "explicit", "minimal",
)
OPERATION = "schedule the project review or send the project summary"
RECIPIENT = "Alex Chen or Alex Kim"
CANONICAL = {
    "ask_operation": "Should I schedule the project review or send the project summary?",
    "ask_recipient": "Should the recipient be Alex Chen or Alex Kim?",
    "ask_full_details": (
        "Should I schedule the project review or send the project summary, and should "
        "the recipient be Alex Chen or Alex Kim?"
    ),
}
WRAPPERS = {
    "concise": "{core}?",
    "neutral": "Please clarify: {core}?",
    "polite": "Could you please clarify: {core}?",
    "formal": "Please specify: {core}?",
    "direct": "Direct clarification: {core}?",
    "friendly": "Could you help me clarify: {core}?",
    "explicit": "For clarity: {core}?",
    "minimal": "{core}?",
}
CLAIMS = (
    "i have", "i've", "i will", "i'll", "already", "completed",
    "executed", "scheduled", "sent", "booked", "done",
)


def payload_hash(value: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def grammar(code: str, style: str) -> str:
    operation = "should I schedule the project review or send the project summary"
    recipient = "should the recipient be Alex Chen or Alex Kim"
    core = (
        operation if code == "ask_operation"
        else recipient if code == "ask_recipient"
        else f"{operation}, and {recipient}"
    )
    question = WRAPPERS[style].format(core=core)
    return question[0].upper() + question[1:]


def strict_content_valid(code: str, question: str) -> bool:
    required = {
        "ask_operation": (OPERATION,),
        "ask_recipient": (RECIPIENT,),
        "ask_full_details": (OPERATION, RECIPIENT),
    }[code]
    forbidden = tuple(value for value in (OPERATION, RECIPIENT) if value not in required)
    structural = bool(
        1 <= len(question) <= 180
        and question.isascii()
        and question.count("?") == 1
        and question.endswith("?")
        and "\n" not in question
        and "\r" not in question
        and "_" not in question
    )
    return bool(
        structural
        and all(question.count(fragment) == 1 for fragment in required)
        and not any(fragment in question for fragment in forbidden)
        and not any(fragment in question.lower() for fragment in CLAIMS)
    )


def close(left: Any, right: Any, tolerance: float = 1e-12) -> bool:
    if isinstance(left, dict) and isinstance(right, dict):
        return set(left) == set(right) and all(
            close(left[key], right[key], tolerance) for key in left
        )
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(
            close(a, b, tolerance) for a, b in zip(left, right)
        )
    if isinstance(left, float) or isinstance(right, float):
        return abs(float(left) - float(right)) <= tolerance
    return left == right


def main() -> None:
    implementation_path = (
        PROJECT_ROOT / "configs/v83-strict-clarification-interface-implementation-lock.json"
    )
    result_path = (
        PROJECT_ROOT / "outputs/v83-strict-clarification-interface/evaluation/result.json"
    )
    verifier_path = PROJECT_ROOT / "python/verify_and_freeze_v83_interface_outcome.py"
    results_doc_path = PROJECT_ROOT / "docs/v83-strict-clarification-interface-results.md"
    audit_path = PROJECT_ROOT / "outputs/v83-strict-clarification-interface/outcome-audit.json"
    lock_path = PROJECT_ROOT / "configs/v83-strict-clarification-interface-outcome-lock.json"
    if audit_path.exists() or lock_path.exists():
        raise RuntimeError("V83 interface outcome is already frozen")
    implementation = json.loads(implementation_path.read_text())
    implementation_payload = {
        key: value for key, value in implementation.items() if key != "lock_payload_sha256"
    }
    result = json.loads(result_path.read_text())
    parent_path = PROJECT_ROOT / implementation["parent_V79_result"]
    parent = json.loads(parent_path.read_text())
    v78 = json.loads(
        (PROJECT_ROOT / "configs/v78-clarification-benchmark-design.json").read_text()
    )

    expected_rendered: list[dict[str, Any]] = []
    expected_nonclarification: list[dict[str, Any]] = []
    certificate_violations = 0
    fixture_values: dict[str, float] = {}
    for fixture_name, fixture in sorted(parent["fixtures"].items()):
        exact = fixture["exact"]
        fixture_values[fixture_name] = exact["value"]
        certificate_violations += exact[
            "complete_belief_certificate_violation_count"
        ]
        for node_index, node in enumerate(exact["policy_nodes"]):
            action = node["action"]
            if action in CODES:
                cases = [("canonical", None, CANONICAL[action])]
                cases.extend(
                    ("finite_grammar", style, grammar(action, style))
                    for style in STYLES
                )
                for source, style, question in cases:
                    expected_rendered.append(
                        {
                            "fixture": fixture_name,
                            "node_index": node_index,
                            "history": node["history"],
                            "action": action,
                            "source": source,
                            "style": style,
                            "question": question,
                            "strict_surface_valid": strict_content_valid(action, question),
                            "source_authorized": source in ("canonical", "finite_grammar"),
                            "deployable": strict_content_valid(action, question)
                            and source in ("canonical", "finite_grammar"),
                            "action_code_preserved": True,
                            "policy_node_structurally_preserved": True,
                        }
                    )
            else:
                expected_nonclarification.append(
                    {
                        "fixture": fixture_name,
                        "node_index": node_index,
                        "action": action,
                        "structurally_identical": True,
                    }
                )
    mutations = result["unsafe_mutation_rows"]
    independently_rejected_mutations = [
        not strict_content_valid(row["action_code"], row["question"])
        for row in mutations
    ]
    expected_metrics = {
        "all_policy_node_count": 22,
        "reachable_clarification_node_count": 6,
        "rendered_reachable_case_count": 54,
        "nonclarification_node_count": 16,
        "strict_surface_validity_rate": 1.0,
        "action_code_preservation_rate": 1.0,
        "policy_node_structural_preservation_rate": 1.0,
        "unsafe_mutation_rejection_rate": 1.0,
        "disabled_untrusted_deployment_rate": 1.0,
        "maximum_policy_value_absolute_error": 0.0,
        "complete_belief_execution_certificate_violation_count": certificate_violations,
    }
    checks = {
        "implementation_lock_exact": bool(
            payload_hash(implementation_payload) == implementation["lock_payload_sha256"]
            and file_sha256(PROJECT_ROOT / implementation["module"])
            == implementation["module_sha256"]
            and file_sha256(PROJECT_ROOT / implementation["evaluator"])
            == implementation["evaluator_sha256"]
        ),
        "parent_V79_result_exact": file_sha256(parent_path)
        == implementation["parent_V79_result_sha256"],
        "rendered_rows_independently_reproduced": close(
            result["rendered_rows"], expected_rendered
        ),
        "nonclarification_rows_independently_reproduced": close(
            result["nonclarification_rows"], expected_nonclarification
        ),
        "metrics_independently_reproduced": close(result["metrics"], expected_metrics),
        "fixture_values_independently_reproduced": close(
            result["fixture_values"], fixture_values
        ),
        "all_unsafe_mutations_independently_rejected": all(
            independently_rejected_mutations
        ),
        "untrusted_valid_looking_surfaces_never_deployable": all(
            row["content_valid"]
            and not row["source_authorized"]
            and not row["deployable"]
            for row in result["untrusted_inspection_rows"]
        ),
        "none_hypothesis_preserved_at_index_four": bool(
            v78["hypothesesInTieBreakOrder"][4] == "none_of_the_above"
        ),
        "all_registered_checks_pass_and_access_is_zero": bool(
            result["passed"]
            and all(result["checks"].values())
            and all(value == 0 for value in result["access"].values())
        ),
    }
    passed = all(checks.values())
    audit = {
        "schema_version": "83-strict-clarification-interface-outcome-audit",
        "experiment": "v83_strict_clarification_interface_outcome_audit",
        "passed": passed,
        "decision": (
            "freeze_positive_model_free_shadow_interface_outcome"
            if passed else "reject_V83_outcome"
        ),
        "checks": checks,
        "independent_metrics": expected_metrics,
        "claim_boundary": result["claim_boundary"],
    }
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    if not passed:
        print(json.dumps(audit, indent=2, sort_keys=True))
        raise SystemExit(1)
    lock = {
        "schema_version": "83-strict-clarification-interface-outcome-lock",
        "experiment": "v83_strict_clarification_interface_outcome_lock",
        "implementation_lock": str(implementation_path.relative_to(PROJECT_ROOT)),
        "implementation_lock_sha256": file_sha256(implementation_path),
        "result": str(result_path.relative_to(PROJECT_ROOT)),
        "result_sha256": file_sha256(result_path),
        "verifier": str(verifier_path.relative_to(PROJECT_ROOT)),
        "verifier_sha256": file_sha256(verifier_path),
        "audit": str(audit_path.relative_to(PROJECT_ROOT)),
        "audit_sha256": file_sha256(audit_path),
        "results_document": str(results_doc_path.relative_to(PROJECT_ROOT)),
        "results_document_sha256": file_sha256(results_doc_path),
        "outcome": {
            "passed": True,
            "decision": result["decision"],
            "metrics": result["metrics"],
        },
        "authorization": {
            "modify_or_rerun_V83": False,
            "use_strict_model_free_interface_in_synthetic_shadow_mode": True,
            "deploy_local_API_adapter_or_untrusted_surface_renderer": False,
            "grant_surface_belief_action_or_execution_authority": False,
            "perform_real_tool_call_or_external_side_effect": False,
            "claim_open_world_language_or_safety_evidence": False,
            "preregister_fresh_schema_grounded_shadow_benchmark": True,
        },
    }
    lock["lock_payload_sha256"] = payload_hash(lock)
    lock_path.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n")
    print(json.dumps(audit, indent=2, sort_keys=True))
    print(json.dumps({"lock": str(lock_path), "sha256": file_sha256(lock_path)}, indent=2))


if __name__ == "__main__":
    main()
