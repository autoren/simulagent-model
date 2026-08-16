#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="configs/v57-definition-augmented-ontology-transfer.json"
    )
    parser.add_argument(
        "--plan", default="docs/v57-definition-augmented-ontology-transfer-plan.md"
    )
    parser.add_argument(
        "--output",
        default="outputs/v57-definition-augmented-ontology-transfer/design-audit.json",
    )
    args = parser.parse_args()
    config_path, plan_path, output = tuple(
        (PROJECT_ROOT / value).resolve()
        for value in (args.config, args.plan, args.output)
    )
    config = json.loads(config_path.read_text())
    source_path = PROJECT_ROOT / config["sourceV56OutcomeLock"]
    source = json.loads(source_path.read_text())
    errors: list[str] = []

    source_ok = (
        source["qualification_passed"]
        and source["authorization"]["preregister_definition_transfer_track"]
        and not source["authorization"]["run_definition_transfer_evaluation"]
        and file_sha256(PROJECT_ROOT / source["result"]) == source["result_sha256"]
        and file_sha256(PROJECT_ROOT / source["post_result_audit"])
        == source["post_result_audit_sha256"]
        and file_sha256(PROJECT_ROOT / source["summary"])
        == source["summary_sha256"]
    )
    if not source_ok:
        errors.append("V56 does not authorize or bind the V57 preregistration")

    boundary = config["claimBoundary"]
    boundary_ok = (
        all(boundary[key] for key in (
            "newOpaqueConceptIdentifiers", "definitionConditionedCompilation",
            "unaryRelationAndActionConcepts", "typedSignAndArgumentRoleTransfer",
            "familiarDeclaredEvidenceGrammar", "controlledDefinitionLanguage",
            "exactExecutableAst", "definitionDependenceControls",
        ))
        and not any(boundary[key] for key in (
            "humanAuthoredLanguage", "openNaturalLanguageDefinitions",
            "jointNewConceptAndNewSurfaceClaim",
            "probabilisticInferenceOrPlanningEvaluation", "modelAccess",
            "adapterTraining",
        ))
    )
    if not boundary_ok:
        errors.append("V57 controlled-definition claim boundary is invalid")

    schema = config["schemaContract"]
    schema_ok = (
        schema["conceptKinds"]
        == ["unary_predicate", "binary_relation", "bound_action"]
        and schema["definitionTemplateFamiliesPerKind"] == 3
        and schema["opaqueIdentifiersCarryNoLexicalSemantics"]
        and schema["schemaSuppliedAtEvaluation"]
        and len(schema["requiredFields"]) == 5
    )
    if not schema_ok:
        errors.append("V57 typed schema contract is incomplete")

    population = config["population"]
    population_ok = (
        population["ontologyPacks"] == 16
        and sum(population["conceptsPerPack"].values()) == 6
        and population["coreRecordsPerPack"] == 120
        and population["coreRecords"] == 1920
        and population["statisticalUnit"] == "ontology_pack"
        and population["independentGenerator"]
        and all(population[key] for key in (
            "newEntityNamespaces", "newConceptLexemes", "newOpaqueIds"
        ))
        and config["safetyPopulation"]["records"] == 224
    )
    if not population_ok:
        errors.append("V57 independent population census or factors are invalid")

    controls = config["controls"]
    controls_ok = (
        set(("noDefinition", "opaqueNameOnly", "shuffledSchema")) <= set(controls)
        and len(controls["mutationControls"]) == 6
    )
    if not controls_ok:
        errors.append("V57 definition-dependence or mutation controls are incomplete")

    gates = config["gates"]
    gates_ok = (
        len(gates) == 15
        and all(
            value == 1.0 for key, value in gates.items()
            if key.startswith("minimum")
        )
        and gates["maximumShuffledSchemaExactAst"] == 0.0
        and gates["maximumTruthAccessCount"] == 0
        and gates["maximumUnexpectedEvaluationAttemptCount"] == 0
    )
    if not gates_ok:
        errors.append("V57 noncompensatory gates are invalid")

    stage = config["stageAuthorization"]
    firewall_ok = (
        set(config["firewall"].values()) == {"forbidden"}
        and stage == {
            "writeAndAuditDefinitionCompiler": True,
            "writeAndAuditIndependentGenerator": True,
            "constructPopulation": False,
            "runCandidateEvaluation": False,
            "collectHumanLanguage": False,
            "modelAccess": False,
        }
    )
    if not firewall_ok:
        errors.append("V57 firewall or stage authorization is invalid")

    downstream_absent = not any(
        (PROJECT_ROOT / path).exists()
        for path in (
            "configs/v57-design-lock.json",
            "configs/v57-implementation-lock.json",
            "configs/v57-population-seal.json",
            "configs/v57-outcome-lock.json",
            "data/v57-definition-augmented-ontology-transfer",
            "outputs/v57-definition-augmented-ontology-transfer/evaluation",
        )
    )
    if not downstream_absent:
        errors.append("V57 downstream artifact exists before design lock")

    checks = {
        "source_v56_authorization_and_binding": source_ok,
        "controlled_definition_claim_boundary": boundary_ok,
        "typed_schema_contract": schema_ok,
        "independent_population_census_and_factors": population_ok,
        "definition_dependence_and_mutation_controls": controls_ok,
        "fifteen_noncompensatory_gates": gates_ok,
        "firewall_and_stage_authorization": firewall_ok,
        "downstream_absent": downstream_absent,
    }
    audit = {
        "schema_version": 57,
        "experiment": "v57_design_audit",
        "passed": not errors,
        "decision": "authorize_v57_design_lock" if not errors else "repair_v57_design",
        "errors": errors,
        "config": str(config_path.relative_to(PROJECT_ROOT)),
        "config_sha256": file_sha256(config_path),
        "preregistration": str(plan_path.relative_to(PROJECT_ROOT)),
        "preregistration_sha256": file_sha256(plan_path),
        "source_outcome_lock": str(source_path.relative_to(PROJECT_ROOT)),
        "source_outcome_lock_sha256": file_sha256(source_path),
        "checks": checks,
        "data_access": {
            "v57_candidate_records_accessed": 0,
            "v57_evaluation_runs": 0,
            "human_authored_records_collected": 0,
            "model_forward_passes": 0,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    print(json.dumps(audit, indent=2, sort_keys=True))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
