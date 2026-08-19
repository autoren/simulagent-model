#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from audit_and_freeze_v165_factored_ontology_identifiability_population import valid_lock
from audit_and_freeze_v167_exact_evidence_gathering_planner import payload_hash
from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main() -> None:
    paths = {
        "config": PROJECT_ROOT / "configs/v213-fresh-programmatic-concept-population.json",
        "plan": PROJECT_ROOT / "docs/v213-fresh-programmatic-concept-population-plan.md",
        "roadmap": PROJECT_ROOT / "docs/research-roadmap-after-v212.md",
        "protocol": PROJECT_ROOT / "python/v213_fresh_programmatic_concept_population.py",
        "tests": PROJECT_ROOT / "python/test_v213_fresh_programmatic_concept_population.py",
        "blueprint_worker": PROJECT_ROOT / "python/v213_blueprint_generator_worker.py",
        "public_projection_worker": PROJECT_ROOT / "python/v213_public_projection_worker.py",
        "runner": PROJECT_ROOT / "python/run_v213_fresh_programmatic_concept_population.py",
        "verifier": PROJECT_ROOT / "python/verify_and_freeze_v213_fresh_programmatic_concept_population_outcome.py",
        "auditor": PROJECT_ROOT / "python/audit_and_freeze_v213_fresh_programmatic_concept_population.py",
    }
    audit_path = PROJECT_ROOT / "outputs/v213-programmatic-concept-population/design-audit.json"
    lock_path = PROJECT_ROOT / "configs/v213-fresh-programmatic-concept-population-lock.json"
    output_root = PROJECT_ROOT / "outputs/v213-programmatic-concept-population/population"
    outcome_path = PROJECT_ROOT / "configs/v213-fresh-programmatic-concept-population-outcome-lock.json"
    if any(path.exists() for path in (audit_path, lock_path, output_root, outcome_path)):
        raise RuntimeError("V213 is already audited, frozen, materialized, or outcome-frozen")
    config = json.loads(paths["config"].read_text())
    parent_path = PROJECT_ROOT / config["parentV212OutcomeLock"]
    parent = json.loads(parent_path.read_text())
    design = config["populationDesign"]
    split = config["splitDesign"]
    roles = config["roleSeparation"]
    gates = config["populationGates"]
    exposure = config["preLockExposure"]
    public_worker = paths["public_projection_worker"].read_text()
    checks = {
        "V212_is_frozen_positive_and_authorizes_V213_design_only": bool(
            valid_lock(parent)
            and parent["outcome"]["passed"]
            and parent["outcome"]["branch"] == "V213_DESIGN_ELIGIBLE"
            and parent["authorization"]["design_V213_fresh_programmatic_concept_population"]
            and not parent["authorization"]["generate_V213_population_without_separate_lock"]
            and not parent["authorization"]["run_local_or_API_model_or_training"]
        ),
        "parent_public_semantics_is_exactly_frozen": bool(
            file_sha256(PROJECT_ROOT / parent["public_semantics"]) == parent["public_semantics_sha256"]
            and config["sourceAlgebra"]["reuseParentPublicSemanticsExactly"]
            and config["sourceAlgebra"]["worldCount"] == 8
            and config["sourceAlgebra"]["completeBehaviorCount"] == 256
        ),
        "role_separation_and_public_truth_schemas_are_explicit": bool(
            roles["blueprintGeneratorWritesPublicBlueprintAndSealedTruthSeparately"]
            and roles["publicProjectionWorkerReadsPublicBlueprintOnly"]
            and roles["splitAssignedBeforeDownstreamMethodDevelopment"]
            and roles["structuralTruthJoinOnlyAfterPublicProjectionFreeze"]
            and set(roles["publicFields"]) & set(roles["hiddenFields"])
            == {"case_id", "group_id", "split", "variant_code"}
        ),
        "balanced_grouped_population_counts_are_frozen": bool(
            len(design["families"]) == 10
            and design["groupsPerFamily"] == 12
            and design["variantsPerGroup"] == 4
            and design["groupCount"] == 120
            and design["recordCount"] == 480
            and split["developmentGroupsPerFamily"] == 8
            and split["protectedGroupsPerFamily"] == 4
            and split["developmentRecordCount"] == 320
            and split["protectedRecordCount"] == 160
            and split["allVariantsOfGroupRemainTogether"]
            and split["protectedDownstreamEvaluationCount"] == 0
        ),
        "public_projection_worker_has_no_config_or_hidden_truth_interface": bool(
            "--public-blueprints" in public_worker
            and "--public-records" in public_worker
            and "--config" not in public_worker
            and "sealed-truth" not in public_worker
            and all(token not in public_worker for token in roles["publicProjectionForbiddenTokens"])
        ),
        "all_population_nonleakage_reconstruction_and_witness_gates_are_exact": bool(
            gates["requiredExactCandidateSetAccuracy"] == 1.0
            and gates["requiredEvidenceStatusAccuracy"] == 1.0
            and gates["requiredExpressibilitySetAccuracy"] == 1.0
            and gates["requiredShadowActionAccuracy"] == 1.0
            and gates["requiredWithinGroupSemanticConsistency"] == 1.0
            and gates["requiredVariantResolutionInvariance"] == 1.0
            and gates["requiredDistinctPairBoundaryWitnessCoverage"] == 1.0
            and gates["maximumPublicHiddenFieldLeakageCount"] == 0
            and gates["maximumPublicForbiddenTokenCount"] == 0
            and gates["maximumCrossSplitGroupOverlapCount"] == 0
        ),
        "prelock_exposure_is_zero_and_pass_authorizes_only_V214_design": bool(
            all(value == 0 for value in exposure.values())
            and config["decisionRule"]["passAuthorizesV214DesignOnly"]
            and not config["decisionRule"]["passAuthorizesProtectedDownstreamEvaluationOrModelRun"]
            and not config["decisionRule"]["passAuthorizesExternalPayloadRegistrationMutationActionOrExecution"]
        ),
        "all_required_files_exist_and_formal_outputs_are_absent": bool(
            all(path.is_file() for path in (*paths.values(), parent_path))
            and not output_root.exists()
            and not outcome_path.exists()
        ),
    }
    passed = all(checks.values())
    audit = {
        "schema_version": "213-fresh-programmatic-concept-population-design-audit",
        "experiment": config["experiment"],
        "passed": passed,
        "decision": "freeze_and_authorize_one_V213_population_materialization" if passed else "reject_V213_design",
        "checks": checks,
    }
    write_json(audit_path, audit)
    if not passed:
        print(json.dumps(audit, indent=2, sort_keys=True))
        raise SystemExit(1)
    dependencies = {**paths, "parent_V212_outcome": parent_path, "parent_public_semantics": PROJECT_ROOT / parent["public_semantics"], "parent_public_cases": PROJECT_ROOT / parent["public_cases"], "design_audit": audit_path}
    lock: dict[str, Any] = {
        "schema_version": "213-fresh-programmatic-concept-population-lock",
        "experiment": config["experiment"],
        "config_payload": config,
        "authorization": {
            "materialize_one_role_separated_population": True,
            "structurally_verify_after_public_projection_freeze": True,
            "run_protected_downstream_method_or_model": False,
            "read_language_external_payload_or_mutate_act_execute": False,
        },
    }
    for key, path in dependencies.items():
        lock[key] = str(path.relative_to(PROJECT_ROOT))
        lock[f"{key}_sha256"] = file_sha256(path)
    lock["lock_payload_sha256"] = payload_hash(lock)
    write_json(lock_path, lock)
    print(json.dumps(audit, indent=2, sort_keys=True))
    print(json.dumps({"lock": str(lock_path.relative_to(PROJECT_ROOT)), "sha256": file_sha256(lock_path)}, indent=2))


if __name__ == "__main__":
    main()
