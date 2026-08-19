#!/usr/bin/env python3
from __future__ import annotations

import json
from typing import Any

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from audit_and_freeze_v167_exact_evidence_gathering_planner import payload_hash
from v186_typed_contract_question_codebook_feasibility import audit_codebook, build_codebook


DEPENDENCY_KEYS = (
    "config", "parent_V185_outcome", "source_V183_outcome", "contract_catalog",
    "hidden_identifiability", "development_identities", "protected_identities", "roadmap",
    "plan", "protocol", "tests", "runner", "verifier", "auditor", "design_audit",
)


def write_json(path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def reconstruct(lock: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    codebook = build_codebook(
        json.loads((PROJECT_ROOT / lock["contract_catalog"]).read_text()),
        json.loads((PROJECT_ROOT / lock["hidden_identifiability"]).read_text()),
        json.loads((PROJECT_ROOT / lock["development_identities"]).read_text()),
        json.loads((PROJECT_ROOT / lock["protected_identities"]).read_text()),
    )
    return codebook, audit_codebook(codebook, lock["config_payload"])


def main() -> None:
    lock_path = PROJECT_ROOT / "configs/v186-typed-contract-question-codebook-feasibility-lock.json"
    output_root = PROJECT_ROOT / "outputs/v186-typed-contract-question-codebook-feasibility/codebook"
    if output_root.exists():
        raise RuntimeError("V186 codebook may be built only once")
    lock = json.loads(lock_path.read_text())
    if payload_hash({key: value for key, value in lock.items() if key != "lock_payload_sha256"}) != lock["lock_payload_sha256"]:
        raise RuntimeError("V186 lock mismatch")
    for key in DEPENDENCY_KEYS:
        if file_sha256(PROJECT_ROOT / lock[key]) != lock[f"{key}_sha256"]:
            raise RuntimeError(f"V186 dependency drifted: {key}")
    codebook, audit = reconstruct(lock)
    config = lock["config_payload"]
    decision = (
        config["decisionRule"]["ifEveryCodebookSeparabilityBindingAndSafetyGatePasses"]
        if audit["passed"]
        else config["decisionRule"]["otherwise"]
    )
    paths = {
        "question_codebook": output_root / "question-codebook.json",
        "contract_answer_vectors": output_root / "contract-answer-vectors.json",
        "equivalence_classes": output_root / "equivalence-classes.json",
        "pairwise_separation": output_root / "pairwise-separation.json",
        "development_bindings": output_root / "development-bindings.json",
        "protected_bindings": output_root / "protected-bindings.json",
        "codebook_summary": output_root / "codebook-summary.json",
    }
    payloads = {
        "question_codebook": {"questions": codebook["questions"], "source": "frozen semantic contract payload only"},
        "contract_answer_vectors": codebook["contract_answer_vectors"],
        "equivalence_classes": {"classes": codebook["equivalence_classes"]},
        "pairwise_separation": {"pairs": codebook["pairwise_separation"]},
        "development_bindings": {"bindings": codebook["bindings"]["development"], "contains_language": False},
        "protected_bindings": {"bindings": codebook["bindings"]["protected"], "contains_language": False},
        "codebook_summary": codebook["summary"],
    }
    for key, path in paths.items():
        write_json(path, payloads[key])
    integrity = {
        key: {"path": str(path.relative_to(PROJECT_ROOT)), "sha256": file_sha256(path)}
        for key, path in paths.items()
    }
    access = {
        "formal_codebook_build_count": 1,
        "contract_question_census_count": 1,
        "pairwise_separation_score_count": codebook["summary"]["contract_pair_count"],
        "utterance_or_dialogue_language_read_count": 0,
        "planner_policy_score_count": 0,
        "model_load_count": 0,
        "model_generation_count": 0,
        "API_call_count": 0,
        "training_run_count": 0,
        "ontology_registration_count": 0,
        "trusted_state_mutation_count": 0,
        "service_call_count": 0,
        "external_side_effect_count": 0,
        "actual_execution_count": 0,
    }
    result = {
        "schema_version": "186-typed-contract-question-codebook-feasibility-result",
        "experiment": config["experiment"],
        "passed": audit["passed"],
        "decision": decision,
        "summary": codebook["summary"],
        "feasibility_gates": audit["checks"],
        "access": access,
        "output_integrity": integrity,
        "claim_boundary": config["claimBoundary"],
    }
    write_json(output_root / "result.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
