#!/usr/bin/env python3
from __future__ import annotations

import json

from audit_and_freeze_v122_prequery_signal_inventory import valid_lock, write_json
from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from v156_model_free_explicit_metadata_question_retrieval import evaluate


def main() -> None:
    lock_path = PROJECT_ROOT / "configs/v156-model-free-explicit-metadata-question-retrieval-lock.json"
    output_dir = PROJECT_ROOT / "outputs/v156-model-free-explicit-metadata-question-retrieval/model-free-realization"
    result_path = output_dir / "result.json"
    access_path = output_dir / "access.json"
    if output_dir.exists():
        raise RuntimeError("V156 realization already exists")
    lock = json.loads(lock_path.read_text())
    dependencies = [key for key in lock if not key.endswith("_sha256") and f"{key}_sha256" in lock]
    if not valid_lock(lock) or not all(
        file_sha256(PROJECT_ROOT / lock[key]) == lock[f"{key}_sha256"] for key in dependencies
    ):
        raise RuntimeError("V156 lock or dependencies changed")
    if not lock["authorization"]["run_single_model_free_development_policy"]:
        raise RuntimeError("V156 policy run is not authorized")

    config = lock["config_payload"]
    public_requests = json.loads((PROJECT_ROOT / lock["development_public_projection"]).read_text())
    metadata = json.loads((PROJECT_ROOT / lock["development_metadata_projection"]).read_text())
    retrieval_catalog = json.loads((PROJECT_ROOT / lock["retrieval_catalog_projection"]).read_text())
    witness_catalog = json.loads((PROJECT_ROOT / lock["witness_catalog"]).read_text())
    witness_config = json.loads((PROJECT_ROOT / lock["witness_config"]).read_text())
    evaluated = evaluate(
        public_requests, metadata, retrieval_catalog, witness_catalog, witness_config, config
    )
    decision = (
        config["decisionRule"]["ifEveryRetrievalComparatorFirewallAndAccessGatePasses"]
        if evaluated["passed"] else config["decisionRule"]["otherwise"]
    )
    result = {
        "schema_version": "156-model-free-explicit-metadata-question-retrieval-result",
        "experiment": config["experiment"], "completed": True,
        "passed": evaluated["passed"], "checks": evaluated["checks"],
        "retrieval_request_metrics": evaluated["retrieval_request_metrics"],
        "comparator_metrics": evaluated["comparator_metrics"],
        "retrieval_records": evaluated["retrieval_records"],
        "episode_count": evaluated["episode_count"],
        "candidate_proposal_field_count": evaluated["candidate_proposal_field_count"],
        "decision": decision,
        "claim_boundary": config["claimBoundary"],
    }
    access = {
        "development_public_request_count": len(public_requests),
        "development_metadata_count": len(metadata),
        "policy_score_count": len(public_requests) * len(retrieval_catalog["queries"]),
        "evaluation_policy_read_count": 0, "model_load_count": 0,
        "model_generation_or_score_count": 0, "API_call_count": 0,
        "training_run_count": 0, "real_service_call_count": 0,
        "external_side_effect_count": 0, "actual_execution_count": 0,
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    write_json(result_path, result)
    write_json(access_path, access)
    print(json.dumps({
        "passed": result["passed"], "decision": decision,
        "retrieval_request_metrics": result["retrieval_request_metrics"],
        "comparator_metrics": result["comparator_metrics"], "access": access,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
