#!/usr/bin/env python3
"""Adversarial pre-collection audit for the V58 human workflow."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from typing import Any

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from v58_collection_protocol import (
    audit_collection_metadata,
    submission_errors,
)


PROTOCOL_FILES = (
    "configs/v58-collection-and-adjudication-protocol.json",
    "docs/v58-collection-and-adjudication-protocol.md",
    "python/v58_collection_protocol.py",
    "python/audit_v58_collection_protocol.py",
)


def _prompt(
    writer: str,
    cohort: str,
    family: str,
    ordinal: int,
    stratum: str,
) -> dict[str, Any]:
    token = hashlib.sha256(
        f"fixture|{writer}|{family}|{ordinal}|{stratum}".encode()
    ).hexdigest()[:16]
    semantics = None if stratum == "abstention" else {
        "predicate": "fixture_relation",
        "arguments": ["fixture_source", "fixture_target"],
        "lexical_sign": "positive",
        "outer_operation": "assert",
    }
    condition = None
    if stratum == "abstention":
        condition = (
            "genuinely_ambiguous_scope_or_referent"
            if ordinal % 2 == 0 else "unsupported_or_unknown_reference"
        )
    return {
        "packet_id": f"packet_{writer}",
        "prompt_id": f"prompt_{token}",
        "anonymous_writer_id": writer,
        "writer_cohort": cohort,
        "collection_round": "synthetic_protocol_fixture",
        "construction_family": family,
        "stratum": stratum,
        "abstention_condition": condition,
        "source_record_id": f"fixture_source_{token}",
        "entity_legend": [
            {"id": "fixture_source", "entity_type": "fixture_type"},
            {"id": "fixture_target", "entity_type": "fixture_type"},
        ],
        "known_ontology_glossary": {
            "predicate": "fixture_relation",
            "source_role": "fixture_source",
            "target_role": "fixture_target",
        },
        "intended_semantics": semantics,
        "writing_instructions": ["synthetic_structure_fixture_only"],
    }


def _synthetic_fixture(protocol: dict[str, Any]) -> tuple[list, list, list, list]:
    split = protocol["constructionSplit"]
    families = split["pilotExposedFamilies"] + split["evaluationOnlyFamilies"]
    prompts = []
    for writer_index in range(protocol["quotas"]["pilotAuthors"]):
        writer = f"fixture_pilot_{writer_index:02d}"
        for family in split["pilotExposedFamilies"]:
            for ordinal in range(
                protocol["quotas"][
                    "acceptedPrimaryPerPilotExposedFamilyPerPilotAuthor"
                ]
            ):
                prompts.append(_prompt(writer, "pilot", family, ordinal, "primary"))
    for writer_index in range(protocol["quotas"]["minimumEvaluationAuthors"]):
        writer = f"fixture_evaluation_{writer_index:02d}"
        for family in families:
            for ordinal in range(
                protocol["quotas"]["acceptedPrimaryPerFamilyPerEvaluationAuthor"]
            ):
                prompts.append(
                    _prompt(writer, "evaluation", family, ordinal, "primary")
                )
            prompts.append(_prompt(writer, "evaluation", family, 0, "abstention"))

    submissions = []
    validations = []
    for index, prompt in enumerate(prompts):
        submission_id = f"fixture_submission_{index:04d}"
        submissions.append({
            "submission_id": submission_id,
            "packet_id": prompt["packet_id"],
            "prompt_id": prompt["prompt_id"],
            "anonymous_writer_id": prompt["anonymous_writer_id"],
            "collection_round": prompt["collection_round"],
            "submitted_text": "Synthetic structural fixture utterance.",
            "timestamp": "2026-08-15T12:00:00Z",
            "consent_and_license_attestation": protocol[
                "submissionSchema"
            ]["attestation"],
        })
        if prompt["stratum"] == "primary":
            verdict = "equivalent_unique"
            ast = prompt["intended_semantics"]
        else:
            verdict = (
                "ambiguous"
                if prompt["abstention_condition"].startswith("genuinely")
                else "unsupported"
            )
            ast = None
        for validator in ("fixture_validator_a", "fixture_validator_b"):
            validations.append({
                "validation_id": f"{validator}_{index:04d}",
                "submission_id": submission_id,
                "anonymous_validator_id": validator,
                "verdict": verdict,
                "inferred_canonical_ast_or_null": ast,
                "construction_family_realized": True,
                "source_target_order_preserved_or_null": True,
                "notes_without_writer_identity": "synthetic fixture",
                "timestamp": "2026-08-15T13:00:00Z",
            })
    return prompts, submissions, validations, []


def _error_names(audit: dict[str, Any]) -> set[str]:
    return {name for name, _count in audit["errors"]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--protocol",
        default="configs/v58-collection-and-adjudication-protocol.json",
    )
    parser.add_argument(
        "--output",
        default=(
            "outputs/v58-human-authored-known-ontology-language/"
            "collection-protocol-audit.json"
        ),
    )
    args = parser.parse_args()
    protocol_path = (PROJECT_ROOT / args.protocol).resolve()
    output = (PROJECT_ROOT / args.output).resolve()
    protocol = json.loads(protocol_path.read_text())
    errors: list[str] = []

    v57_path = PROJECT_ROOT / protocol["sourceV57OutcomeLock"]
    v58_path = PROJECT_ROOT / protocol["sourceV58DesignLock"]
    v57 = json.loads(v57_path.read_text())
    v58 = json.loads(v58_path.read_text())
    source_ok = (
        v57["qualification_passed"]
        and v57["authorization"]["write_and_audit_v58_collection_protocol"]
        and v57["authorization"]["freeze_v58_collection_protocol"]
        and not v57["authorization"]["collect_v58_pilot_language"]
        and not v57["authorization"]["collect_v58_evaluation_language"]
        and v58["authorization"]["write_and_audit_collection_protocol"]
        and not v58["authorization"]["generate_blinded_author_packets"]
        and file_sha256(PROJECT_ROOT / v57["result"]) == v57["result_sha256"]
        and file_sha256(PROJECT_ROOT / v58["config"]) == v58["config_sha256"]
    )
    if not source_ok:
        errors.append("V57/V58 locks do not authorize the collection protocol")

    families = v58["config_payload"]["collection"]["constructionFamilies"]
    seed = v58["lock_payload_sha256"]
    ranked = sorted(
        families,
        key=lambda family: hashlib.sha256(
            f"{seed}|pilot-family|{family}".encode()
        ).hexdigest(),
    )
    split = protocol["constructionSplit"]
    construction_ok = (
        split["pilotExposedFamilies"] == ranked[:5]
        and split["evaluationOnlyFamilies"] == ranked[5:]
        and set(ranked) == set(families)
        and split["pilotTextForEvaluationOnlyFamilies"] == 0
        and split["evaluationAuthorsCoverAllFamilies"]
        and not split["randomUtteranceSplit"]
    )
    if not construction_ok:
        errors.append("V58 deterministic construction holdout is invalid")

    quotas = protocol["quotas"]
    quota_ok = (
        quotas["pilotAuthors"] == 2
        and quotas["acceptedPrimaryPerPilotAuthor"] == 60
        and quotas["acceptedPrimaryPerPilotExposedFamilyPerPilotAuthor"] * 5
        == quotas["acceptedPrimaryPerPilotAuthor"]
        and quotas["minimumEvaluationAuthors"] == 10
        and quotas["acceptedPrimaryPerEvaluationAuthor"] == 60
        and quotas["acceptedPrimaryPerFamilyPerEvaluationAuthor"] * 10
        == quotas["acceptedPrimaryPerEvaluationAuthor"]
        and quotas["minimumAcceptedEvaluationPrimary"] == 600
        and quotas["acceptedAbstentionPerEvaluationAuthor"] == 10
        and quotas["acceptedAbstentionPerFamilyPerEvaluationAuthor"] == 1
        and quotas["minimumAcceptedEvaluationAbstention"] == 100
        and not quotas["rejectedOrWithdrawnItemsCountTowardQuota"]
    )
    if not quota_ok:
        errors.append("V58 author/family/abstention quotas are invalid")

    v40_seal_path = PROJECT_ROOT / protocol["knownOntologySource"]["corpusSeal"]
    v40_seal = json.loads(v40_seal_path.read_text())
    v40_core = protocol["knownOntologySource"]["corePopulation"]
    ontology_ok = (
        v40_core == v40_seal["corpora"]["independent_confirmation"]["path"]
        and file_sha256(PROJECT_ROOT / v40_core)
        == v40_seal["corpora"]["independent_confirmation"]["sha256"]
        and file_sha256(PROJECT_ROOT / v40_seal["manifest"])
        == v40_seal["manifest_sha256"]
        and set(protocol["knownOntologySource"]["authorPacketExclusions"])
        == {"original_evidence_text", "oracle_metadata", "reference_surface_realization"}
        and "target_ast" in protocol["knownOntologySource"][
            "candidateInputExclusions"
        ]
    )
    if not ontology_ok:
        errors.append("V58 frozen V40 source or prompt/candidate boundary is invalid")

    role_ok = (
        len(protocol["roles"]) == 6
        and not any(protocol["roleSeparation"].values())
        and protocol["validation"]["validatorsPerSubmission"] == 2
        and protocol["validation"]["minimumRawAgreementBeforePopulationSeal"]
        == v58["config_payload"]["gates"]["minimumValidatorAgreement"]
        and not protocol["validation"]["unadjudicatedDisagreementAccepted"]
        and protocol["validation"]["validatorAndAdjudicatorRecordsImmutable"]
    )
    if not role_ok:
        errors.append("V58 role separation or blinded validation is invalid")

    sequence = protocol["freezeAndReleaseSequence"]
    sequence_ok = (
        sequence.index("freeze_collection_protocol")
        < sequence.index("release_pilot_packets_only")
        < sequence.index(
            "freeze_candidate_parser_all_baselines_prompts_and_scoring_implementation"
        )
        < sequence.index("release_evaluation_packets")
        < sequence.index("seal_evaluation_primary_and_abstention_populations")
        < sequence.index("run_exactly_one_candidate_evaluation")
        and set(protocol["firewall"].values()) == {"forbidden"}
        and protocol["protocolAuthorization"] == {
            "generateAndAuditBlindedAuthorPackets": True,
            "releasePilotPackets": False,
            "collectPilotLanguage": False,
            "releaseEvaluationPackets": False,
            "collectEvaluationLanguage": False,
            "writeCandidateParser": False,
            "runCandidateEvaluation": False,
            "modelGeneratedWritingAssistance": False,
        }
    )
    if not sequence_ok:
        errors.append("V58 freeze/release sequence or authorization is invalid")

    prompts, submissions, validations, adjudications = _synthetic_fixture(protocol)
    fixture = audit_collection_metadata(
        prompts, submissions, validations, adjudications, protocol
    )
    fixture_ok = (
        fixture["passed"]
        and fixture["metrics"] == {
            "raw_validator_verdict_agreement": 1.0,
            "pilot_authors": 2,
            "evaluation_authors": 10,
            "accepted_primary": 720,
            "accepted_abstention": 100,
            "adjudications": 0,
        }
    )
    if not fixture_ok:
        errors.append("V58 complete synthetic metadata fixture failed")

    missing_attestation = copy.deepcopy(submissions[0])
    missing_attestation["consent_and_license_attestation"] = {
        **missing_attestation["consent_and_license_attestation"],
        "humanAuthoredWithoutGenerativeAssistance": False,
    }
    leaked_target = copy.deepcopy(submissions[0])
    leaked_target["target_ast"] = prompts[0]["intended_semantics"]
    submission_attacks_ok = (
        "attestation" in submission_errors(missing_attestation, prompts[0], protocol)
        and "submission_fields" in submission_errors(
            leaked_target, prompts[0], protocol
        )
    )

    duplicate_validators = copy.deepcopy(validations)
    duplicate_validators[1]["anonymous_validator_id"] = duplicate_validators[0][
        "anonymous_validator_id"
    ]
    duplicate_audit = audit_collection_metadata(
        prompts, submissions, duplicate_validators, adjudications, protocol
    )

    writer_validator_overlap = copy.deepcopy(validations)
    writer_validator_overlap[0]["anonymous_validator_id"] = submissions[0][
        "anonymous_writer_id"
    ]
    writer_validator_audit = audit_collection_metadata(
        prompts,
        submissions,
        writer_validator_overlap,
        adjudications,
        protocol,
    )

    low_agreement = copy.deepcopy(validations)
    second_indices = list(range(1, len(low_agreement), 2))[:83]
    for index in second_indices:
        low_agreement[index]["verdict"] = "not_equivalent"
        low_agreement[index]["inferred_canonical_ast_or_null"] = None
    agreement_audit = audit_collection_metadata(
        prompts, submissions, low_agreement, adjudications, protocol
    )

    under_quota_audit = audit_collection_metadata(
        prompts[:-1], submissions[:-1], validations[:-2], adjudications, protocol
    )

    family_leak_prompts = copy.deepcopy(prompts)
    family_leak_prompts[0]["construction_family"] = split[
        "evaluationOnlyFamilies"
    ][0]
    family_leak_audit = audit_collection_metadata(
        family_leak_prompts, submissions, validations, adjudications, protocol
    )
    adversarial_ok = (
        submission_attacks_ok
        and "duplicate_validator" in _error_names(duplicate_audit)
        and "validator_writer_overlap" in _error_names(writer_validator_audit)
        and "validator_agreement" in _error_names(agreement_audit)
        and "missing_or_duplicate_adjudication" in _error_names(agreement_audit)
        and (
            "evaluation_abstention_quota" in _error_names(under_quota_audit)
            or "total_abstention_quota" in _error_names(under_quota_audit)
        )
        and "evaluation_only_family_leak_into_pilot"
        in _error_names(family_leak_audit)
        and not any(
            audit["passed"] for audit in (
                duplicate_audit,
                writer_validator_audit,
                agreement_audit,
                under_quota_audit,
                family_leak_audit,
            )
        )
    )
    if not adversarial_ok:
        errors.append("V58 adversarial protocol fixtures were not rejected")

    downstream_absent = not any(
        (PROJECT_ROOT / path).exists()
        for path in (
            "configs/v58-collection-protocol-lock.json",
            "configs/v58-author-packet-seal.json",
            "configs/v58-pilot-population-seal.json",
            "configs/v58-candidate-lock.json",
            "configs/v58-evaluation-population-seal.json",
            "configs/v58-outcome-lock.json",
            "data/v58-human-authored-known-ontology-language/author-packets",
            "data/v58-human-authored-known-ontology-language/pilot-submissions",
            "data/v58-human-authored-known-ontology-language/evaluation-submissions",
        )
    )
    if not downstream_absent:
        errors.append("V58 packet, human text, or downstream artifact already exists")

    audit = {
        "schema_version": 58,
        "experiment": "v58_collection_protocol_audit",
        "passed": not errors,
        "decision": (
            "authorize_v58_collection_protocol_lock"
            if not errors else "repair_v58_collection_protocol"
        ),
        "errors": errors,
        "protocol": str(protocol_path.relative_to(PROJECT_ROOT)),
        "protocol_sha256": file_sha256(protocol_path),
        "protocol_files_sha256": {
            path: file_sha256(PROJECT_ROOT / path) for path in PROTOCOL_FILES
        },
        "v57_outcome_lock": str(v57_path.relative_to(PROJECT_ROOT)),
        "v57_outcome_lock_sha256": file_sha256(v57_path),
        "v58_design_lock": str(v58_path.relative_to(PROJECT_ROOT)),
        "v58_design_lock_sha256": file_sha256(v58_path),
        "v40_corpus_seal": str(v40_seal_path.relative_to(PROJECT_ROOT)),
        "v40_corpus_seal_sha256": file_sha256(v40_seal_path),
        "checks": {
            "v57_outcome_and_v58_design_authorization": source_ok,
            "deterministic_construction_family_holdout": construction_ok,
            "balanced_author_family_and_abstention_quotas": quota_ok,
            "frozen_v40_source_and_prompt_candidate_boundary": ontology_ok,
            "role_separation_blinding_and_adjudication": role_ok,
            "freeze_release_sequence_and_narrow_authorization": sequence_ok,
            "complete_synthetic_metadata_fixture": fixture_ok,
            "adversarial_protocol_fixtures_rejected": adversarial_ok,
            "human_text_packets_and_downstream_absent": downstream_absent,
        },
        "fixture_metrics": fixture["metrics"],
        "data_access": {
            "human_authored_records_collected": 0,
            "human_authored_text_accessed": 0,
            "evaluation_author_text_accessed": 0,
            "synthetic_metadata_fixture_prompts": len(prompts),
            "synthetic_metadata_fixture_submissions": len(submissions),
            "candidate_evaluation_runs": 0,
            "model_forward_passes": 0,
        },
    }
    output.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    print(json.dumps(audit, indent=2, sort_keys=True))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
