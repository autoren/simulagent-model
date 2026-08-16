#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/v58-human-authored-known-ontology-language.json")
    parser.add_argument("--plan", default="docs/v58-human-authored-known-ontology-language-plan.md")
    parser.add_argument("--output", default="outputs/v58-human-authored-known-ontology-language/design-audit.json")
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
        and source["authorization"]["preregister_human_authored_language_track"]
        and not source["authorization"]["run_human_authored_language_evaluation"]
        and file_sha256(PROJECT_ROOT / source["result"]) == source["result_sha256"]
        and file_sha256(PROJECT_ROOT / source["post_result_audit"])
        == source["post_result_audit_sha256"]
    )
    if not source_ok:
        errors.append("V56 does not authorize or bind the V58 preregistration")

    boundary = config["claimBoundary"]
    boundary_ok = (
        all(boundary[key] for key in (
            "humanAuthoredEvaluationLanguage", "knownFrozenOntology",
            "authorHeldOutEvaluation", "constructionFamilyHeldOutEvaluation",
            "exactAstPrimaryMetric", "ambiguityAndUnknownRequireAbstention",
        ))
        and not any(boundary[key] for key in (
            "newConcepts", "definitionAugmentedTransfer",
            "jointNewConceptAndNewSurfaceClaim", "syntheticTextCountedAsHuman",
            "probabilisticInferenceOrPlanningEvaluation",
        ))
    )
    if not boundary_ok:
        errors.append("V58 human-language known-ontology boundary is invalid")

    collection = config["collection"]
    collection_ok = (
        collection["pilotAuthorsExcludedFromEvaluation"] == 2
        and collection["minimumEvaluationAuthors"] == 10
        and collection["acceptedUtterancesPerEvaluationAuthor"] == 60
        and collection["minimumEvaluationUtterances"] == 600
        and len(collection["constructionFamilies"]) == 10
        and set(collection["splitUnit"]) == {"author", "construction_family"}
        and not collection["randomUtteranceSplit"]
        and collection["independentSemanticValidation"]
        == "two_blinded_validators_with_adjudication"
        and collection["syntheticOrModelGeneratedUtterances"] == "excluded"
        and len(collection["requiredProvenance"]) == 5
    )
    if not collection_ok:
        errors.append("V58 author census, split, validation, or provenance is invalid")

    candidate = config["candidateProtocol"]
    candidate_ok = (
        candidate["freezeBeforeEvaluationAuthorTextIsVisible"]
        and candidate["developmentTextMayUsePilotAuthorsOnly"]
        and candidate["evaluationAuthorTextMayNotTrainTunePromptOrSelectCandidate"]
        and candidate["canonicalCompilerAndExecutorRemainFrozen"]
        and candidate["abstentionIsAnExplicitOutput"]
    )
    if not candidate_ok:
        errors.append("V58 candidate freeze or evaluation-text firewall is invalid")

    controls_ok = len(config["controls"]) == 6 and all(config["controls"].values())
    if not controls_ok:
        errors.append("V58 human-transfer controls are incomplete")

    gates = config["gates"]
    gates_ok = (
        len(gates) == 16
        and gates["minimumEvaluationAuthors"] == 10
        and gates["minimumAcceptedEvaluationUtterances"] == 600
        and gates["minimumExactAst"] == 0.9
        and gates["minimumWorstAuthorExactAst"] == 0.75
        and gates["minimumWorstConstructionFamilyExactAst"] == 0.75
        and gates["minimumProvenanceCompleteness"] == 1.0
        and gates["maximumTargetLeakCount"] == 0
        and gates["maximumUnexpectedEvaluationAttemptCount"] == 0
    )
    if not gates_ok:
        errors.append("V58 noncompensatory author/structure gates are invalid")

    stage = config["stageAuthorization"]
    firewall_ok = (
        set(config["firewall"].values()) == {"forbidden"}
        and stage == {
            "writeAndAuditCollectionProtocol": True,
            "generateBlindedAuthorPackets": False,
            "collectPilotLanguage": False,
            "collectEvaluationLanguage": False,
            "writeCandidateParser": False,
            "runCandidateEvaluation": False,
        }
    )
    if not firewall_ok:
        errors.append("V58 firewall or stage authorization is invalid")

    downstream_absent = not any(
        (PROJECT_ROOT / path).exists()
        for path in (
            "configs/v58-design-lock.json", "configs/v58-collection-protocol-lock.json",
            "configs/v58-population-seal.json", "configs/v58-outcome-lock.json",
            "data/v58-human-authored-known-ontology-language",
            "outputs/v58-human-authored-known-ontology-language/evaluation",
        )
    )
    if not downstream_absent:
        errors.append("V58 downstream artifact exists before design lock")

    checks = {
        "source_v56_authorization_and_binding": source_ok,
        "human_language_known_ontology_boundary": boundary_ok,
        "author_census_split_validation_and_provenance": collection_ok,
        "candidate_freeze_and_evaluation_text_firewall": candidate_ok,
        "human_transfer_controls": controls_ok,
        "sixteen_noncompensatory_gates": gates_ok,
        "firewall_and_stage_authorization": firewall_ok,
        "downstream_absent": downstream_absent,
    }
    audit = {
        "schema_version": 58,
        "experiment": "v58_design_audit",
        "passed": not errors,
        "decision": "authorize_v58_design_lock" if not errors else "repair_v58_design",
        "errors": errors,
        "config": str(config_path.relative_to(PROJECT_ROOT)),
        "config_sha256": file_sha256(config_path),
        "preregistration": str(plan_path.relative_to(PROJECT_ROOT)),
        "preregistration_sha256": file_sha256(plan_path),
        "source_outcome_lock": str(source_path.relative_to(PROJECT_ROOT)),
        "source_outcome_lock_sha256": file_sha256(source_path),
        "checks": checks,
        "data_access": {
            "human_authored_records_collected": 0,
            "evaluation_author_text_accessed": 0,
            "v58_candidate_evaluation_runs": 0,
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
