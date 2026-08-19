from __future__ import annotations

import hashlib
import itertools
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable
import xml.etree.ElementTree as ET

from v10_protocol import file_sha256


RDF_ABOUT = "{http://www.w3.org/1999/02/22-rdf-syntax-ns#}about"
RDF_NODE_ID = "{http://www.w3.org/1999/02/22-rdf-syntax-ns#}nodeID"
SOURCE_ID_RE = re.compile(
    r"(?i)(?:https?://purl\.obolibrary\.org/obo/)?UBERON(?::|_)[0-9]+"
)


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def stable_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _finite(value: Any) -> bool:
    if isinstance(value, dict):
        return all(_finite(item) for item in value.values())
    if isinstance(value, list):
        return all(_finite(item) for item in value)
    if isinstance(value, float):
        return math.isfinite(value)
    return True


def _rate(values: Iterable[bool], *, empty: float = 1.0) -> float:
    materialized = list(values)
    return sum(materialized) / len(materialized) if materialized else empty


def normalize_space(value: str) -> str:
    return " ".join(value.strip().split())


def strip_obo_comment(value: str) -> str:
    return normalize_space(value.split(" ! ", 1)[0])


def extract_quoted(value: str) -> str:
    match = re.match(r'^"((?:\\.|[^"\\])*)"', value.strip())
    if not match:
        return normalize_space(value)
    return normalize_space(match.group(1).replace(r'\"', '"').replace(r"\\", "\\"))


def redact_source_ids(value: str) -> str:
    return SOURCE_ID_RE.sub("[SOURCE_ID]", value)


def parse_obo_text(text: str) -> dict[str, dict[str, list[str]]]:
    terms: dict[str, dict[str, list[str]]] = {}
    stanza_type: str | None = None
    fields: dict[str, list[str]] = defaultdict(list)

    def flush() -> None:
        nonlocal fields
        if stanza_type == "Term" and fields.get("id"):
            term_id = normalize_space(fields["id"][0])
            if term_id in terms:
                raise ValueError(f"duplicate OBO term id: {term_id}")
            terms[term_id] = {key: list(values) for key, values in fields.items()}
        fields = defaultdict(list)

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("!"):
            continue
        if line.startswith("[") and line.endswith("]"):
            flush()
            stanza_type = line[1:-1]
            continue
        if stanza_type is None or ":" not in line:
            continue
        key, value = line.split(":", 1)
        fields[key.strip()].append(value.strip())
    flush()
    return terms


def load_obo(path: Path, encoding: str = "utf-8") -> dict[str, dict[str, list[str]]]:
    return parse_obo_text(path.read_text(encoding=encoding))


def _first(fields: dict[str, list[str]] | None, key: str) -> str:
    if not fields or not fields.get(key):
        return ""
    return normalize_space(fields[key][0])


def text_state(fields: dict[str, list[str]] | None) -> dict[str, Any]:
    if not fields:
        return {"name": "", "definition": "", "synonyms": []}
    return {
        "name": redact_source_ids(_first(fields, "name")),
        "definition": redact_source_ids(extract_quoted(_first(fields, "def"))),
        "synonyms": sorted(
            {
                redact_source_ids(extract_quoted(value))
                for value in fields.get("synonym", [])
                if extract_quoted(value)
            }
        ),
    }


def status_state(fields: dict[str, list[str]] | None) -> dict[str, Any]:
    if not fields:
        return {"is_obsolete": False, "replaced_by": [], "consider": []}
    return {
        "is_obsolete": _first(fields, "is_obsolete").lower() == "true",
        "replaced_by": sorted(strip_obo_comment(value) for value in fields.get("replaced_by", [])),
        "consider": sorted(strip_obo_comment(value) for value in fields.get("consider", [])),
    }


def asserted_axioms(
    fields: dict[str, list[str]] | None, logical_fields: list[str]
) -> list[str]:
    if not fields:
        return []
    values = []
    for key in logical_fields:
        for value in fields.get(key, []):
            normalized = strip_obo_comment(value)
            if normalized:
                values.append(f"{key}={normalized}")
    return sorted(set(values))


def change_types(
    older: dict[str, list[str]] | None,
    newer: dict[str, list[str]] | None,
    logical_fields: list[str],
) -> list[str]:
    if older is None and newer is not None:
        return ["ADDED"]
    if newer is None:
        return ["REMOVED"]
    old_text = text_state(older)
    new_text = text_state(newer)
    old_status = status_state(older)
    new_status = status_state(newer)
    changes = []
    if old_text["name"] != new_text["name"]:
        changes.append("NAME_CHANGED")
    if old_text["definition"] != new_text["definition"]:
        changes.append("DEFINITION_CHANGED")
    if old_text["synonyms"] != new_text["synonyms"]:
        changes.append("SYNONYM_CHANGED")
    if asserted_axioms(older, logical_fields) != asserted_axioms(newer, logical_fields):
        changes.append("LOGICAL_AXIOM_CHANGED")
    if old_status["is_obsolete"] != new_status["is_obsolete"]:
        changes.append("OBSOLETION_CHANGED")
    if (
        old_status["replaced_by"] != new_status["replaced_by"]
        or old_status["consider"] != new_status["consider"]
    ):
        changes.append("REPLACEMENT_CHANGED")
    return changes


def _primary(changes: list[str], precedence: list[str]) -> str:
    for candidate in precedence:
        if candidate in changes:
            return candidate
    raise ValueError(f"no primary change for {changes}")


def build_population_records(
    older_terms: dict[str, dict[str, list[str]]],
    newer_terms: dict[str, dict[str, list[str]]],
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    design = config["populationDesign"]
    logical_fields = config["parserDesign"]["logicalFields"]
    eligible_changes = set(design["eligibleChangeTypes"])
    provisional: list[tuple[dict[str, Any], dict[str, Any], str]] = []
    exclusion_counts: Counter[str] = Counter()
    all_change_counts: Counter[str] = Counter()

    for term_id in sorted(set(older_terms) | set(newer_terms)):
        older = older_terms.get(term_id)
        newer = newer_terms.get(term_id)
        changes = change_types(older, newer, logical_fields)
        all_change_counts.update(changes)
        if newer is None:
            exclusion_counts["removed"] += 1
            continue
        selected = [value for value in changes if value in eligible_changes]
        if not selected:
            exclusion_counts["unchanged_or_ineligible_change"] += 1
            continue
        current_text = text_state(newer)
        current_axioms = asserted_axioms(newer, logical_fields)
        if not current_text["name"]:
            exclusion_counts["missing_current_name"] += 1
            continue
        if not current_text["definition"]:
            exclusion_counts["missing_current_definition"] += 1
            continue
        if not current_axioms:
            exclusion_counts["missing_current_asserted_axiom"] += 1
            continue
        prior_text = text_state(older)
        prior_axioms = asserted_axioms(older, logical_fields)
        observation = {
            "prior_name": prior_text["name"],
            "prior_definition": prior_text["definition"],
            "prior_synonyms": prior_text["synonyms"],
            "current_name": current_text["name"],
            "current_definition": current_text["definition"],
            "current_synonyms": current_text["synonyms"],
        }
        observation_hash = stable_hash(observation)
        class_id = "C_" + stable_hash(current_axioms)[:20]
        primary = _primary(selected, design["primaryChangePrecedence"])
        case_id = "E_" + stable_hash(
            [config["experiment"], term_id, config["populationDesign"]["newerPayloadId"]]
        )[:20]
        public = {
            "case_id": case_id,
            "group_id": "G_" + observation_hash[:20],
            "split": "UNASSIGNED",
            "task": "RETROSPECTIVE_ASSERTED_AXIOM_RECONSTRUCTION",
            "observation": observation,
        }
        truth = {
            "case_id": case_id,
            "group_id": public["group_id"],
            "split": "UNASSIGNED",
            "source_term_id": term_id,
            "change_types": sorted(selected),
            "primary_change_type": primary,
            "prior_asserted_axioms": prior_axioms,
            "current_asserted_axioms": current_axioms,
            "added_axioms": sorted(set(current_axioms) - set(prior_axioms)),
            "removed_axioms": sorted(set(prior_axioms) - set(current_axioms)),
            "oracle_class_id": class_id,
            "candidate_class_ids": [],
            "evidence_status": "UNASSIGNED",
        }
        provisional.append((public, truth, observation_hash))

    candidates_by_observation: dict[str, set[str]] = defaultdict(set)
    for _, truth, observation_hash in provisional:
        candidates_by_observation[observation_hash].add(truth["oracle_class_id"])
    for _, truth, observation_hash in provisional:
        candidates = sorted(candidates_by_observation[observation_hash])
        truth["candidate_class_ids"] = candidates
        truth["evidence_status"] = "SINGLETON" if len(candidates) == 1 else "AMBIGUOUS"

    group_records: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(list)
    for public, truth, _ in provisional:
        group_records[public["group_id"]].append((public, truth))
    strata: dict[str, list[str]] = defaultdict(list)
    for group_id, records in group_records.items():
        primary_signature = "+".join(sorted({truth["primary_change_type"] for _, truth in records}))
        strata[primary_signature].append(group_id)
    protected_groups: set[str] = set()
    for stratum, groups in sorted(strata.items()):
        ranked = sorted(groups, key=lambda group_id: stable_hash([stratum, group_id]))
        for index, group_id in enumerate(ranked):
            if index % 4 == 3:
                protected_groups.add(group_id)
    public_records: list[dict[str, Any]] = []
    truth_records: list[dict[str, Any]] = []
    for public, truth, _ in provisional:
        split = "PROTECTED" if public["group_id"] in protected_groups else "DEVELOPMENT"
        public["split"] = split
        truth["split"] = split
        public_records.append(public)
        truth_records.append(truth)
    public_records.sort(key=lambda record: record["case_id"])
    truth_records.sort(key=lambda record: record["case_id"])
    build_manifest = {
        "schema_version": "216-population-build-manifest",
        "older_term_count": len(older_terms),
        "newer_term_count": len(newer_terms),
        "all_change_counts": dict(sorted(all_change_counts.items())),
        "exclusion_counts": dict(sorted(exclusion_counts.items())),
        "eligible_record_count": len(public_records),
        "eligible_group_count": len(group_records),
        "development_record_count": sum(record["split"] == "DEVELOPMENT" for record in public_records),
        "protected_record_count": sum(record["split"] == "PROTECTED" for record in public_records),
        "development_group_ids": sorted(set(group_records) - protected_groups),
        "protected_group_ids": sorted(protected_groups),
        "primary_change_type_counts": dict(
            sorted(Counter(record["primary_change_type"] for record in truth_records).items())
        ),
        "ambiguous_record_count": sum(record["evidence_status"] == "AMBIGUOUS" for record in truth_records),
        "ambiguous_group_count": len(
            {record["group_id"] for record in truth_records if record["evidence_status"] == "AMBIGUOUS"}
        ),
    }
    return public_records, truth_records, build_manifest


def w3c_rdfxml_control(path: Path) -> dict[str, Any]:
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError as error:
        return {
            "parse_success": False,
            "root_tag": None,
            "rdf_subject_count": 0,
            "error": f"{type(error).__name__}: {error}",
        }
    subjects = {
        element.attrib.get(RDF_ABOUT) or f"_:{element.attrib[RDF_NODE_ID]}"
        for element in root.iter()
        if RDF_ABOUT in element.attrib or RDF_NODE_ID in element.attrib
    }
    return {
        "parse_success": True,
        "root_tag": root.tag,
        "rdf_subject_count": len(subjects),
        "error": None,
    }


def score_population(
    retrieval_manifest: dict[str, Any],
    parser_control: dict[str, Any],
    public_records: list[dict[str, Any]],
    truth_records: list[dict[str, Any]],
    split: dict[str, Any],
    population_manifest: dict[str, Any],
    config: dict[str, Any],
    project_root: Path,
) -> dict[str, Any]:
    payloads = {payload["payloadId"]: payload for payload in config["payloads"]}
    attempts = retrieval_manifest["attempts"]
    attempt_ids = [row["payload_id"] for row in attempts]
    successful = [row for row in attempts if row["success"]]
    raw_hash_checks = []
    byte_checks = []
    digest_checks = []
    for row in successful:
        frozen = payloads[row["payload_id"]]
        path = project_root / frozen["rawPath"]
        raw_hash_checks.append(path.is_file() and file_sha256(path) == row["sha256"])
        byte_checks.append(path.is_file() and path.stat().st_size == frozen["expectedByteCount"] == row["byte_count"])
        digest_checks.append(
            frozen["declaredSha256"] is None or row["sha256"] == frozen["declaredSha256"]
        )
    public_by_id = {record["case_id"]: record for record in public_records}
    truth_by_id = {record["case_id"]: record for record in truth_records}
    duplicate_case_ids = len(public_records) - len(public_by_id) + len(truth_records) - len(truth_by_id)
    aligned_ids = set(public_by_id) == set(truth_by_id)
    development_groups = {record["group_id"] for record in public_records if record["split"] == "DEVELOPMENT"}
    protected_groups = {record["group_id"] for record in public_records if record["split"] == "PROTECTED"}
    classes_by_observation: dict[str, set[str]] = defaultdict(set)
    observation_hash_by_case: dict[str, str] = {}
    for case_id, public in public_by_id.items():
        observation_hash = stable_hash(public["observation"])
        observation_hash_by_case[case_id] = observation_hash
        if case_id in truth_by_id:
            classes_by_observation[observation_hash].add(truth_by_id[case_id]["oracle_class_id"])
    version_checks = []
    for case_id, truth in truth_by_id.items():
        expected = sorted(classes_by_observation[observation_hash_by_case[case_id]])
        version_checks.append(truth["candidate_class_ids"] == expected)
    class_axioms: dict[str, tuple[str, ...]] = {}
    for truth in truth_records:
        signature = tuple(truth["current_asserted_axioms"])
        existing = class_axioms.setdefault(truth["oracle_class_id"], signature)
        if existing != signature:
            raise ValueError("oracle class collision with different axiom signatures")
    witness_checks = []
    for candidates in classes_by_observation.values():
        for left, right in itertools.combinations(sorted(candidates), 2):
            witness_checks.append(bool(set(class_axioms[left]) ^ set(class_axioms[right])))
    public_source_leakage = sum(
        len(SOURCE_ID_RE.findall(canonical_json(record))) for record in public_records
    )
    current_definition_checks = [
        bool(public_by_id[truth["case_id"]]["observation"]["current_definition"])
        for truth in truth_records
    ]
    current_axiom_checks = [bool(truth["current_asserted_axioms"]) for truth in truth_records]
    metrics = {
        "payload_count": len(payloads),
        "attempt_count": len(attempts),
        "exact_payload_id_accounting": sorted(attempt_ids) == sorted(payloads),
        "successful_payload_retrieval_rate": len(successful) / len(payloads),
        "raw_hash_coverage": _rate(raw_hash_checks),
        "expected_byte_count_accuracy": _rate(byte_checks),
        "declared_digest_accuracy": _rate(digest_checks),
        "total_payload_bytes": sum(row["byte_count"] for row in successful),
        "older_term_count": parser_control["older_term_count"],
        "newer_term_count": parser_control["newer_term_count"],
        "w3c_rdfxml_parse_success": parser_control["w3c"]["parse_success"],
        "w3c_rdf_subject_count": parser_control["w3c"]["rdf_subject_count"],
        "eligible_record_count": len(public_records),
        "eligible_group_count": len({record["group_id"] for record in public_records}),
        "development_record_count": sum(record["split"] == "DEVELOPMENT" for record in public_records),
        "protected_record_count": sum(record["split"] == "PROTECTED" for record in public_records),
        "development_group_count": len(development_groups),
        "protected_group_count": len(protected_groups),
        "distinct_primary_change_type_count": len({record["primary_change_type"] for record in truth_records}),
        "primary_change_type_counts": population_manifest["primary_change_type_counts"],
        "ambiguous_record_count": population_manifest["ambiguous_record_count"],
        "ambiguous_group_count": population_manifest["ambiguous_group_count"],
        "current_definition_coverage": _rate(current_definition_checks),
        "current_asserted_axiom_coverage": _rate(current_axiom_checks),
        "version_space_reconstruction_accuracy": _rate(version_checks),
        "boundary_witness_coverage": _rate(witness_checks),
        "cross_split_group_overlap_count": len(development_groups & protected_groups),
        "duplicate_case_id_count": duplicate_case_ids,
        "public_source_identifier_leakage_count": public_source_leakage,
        "public_truth_case_alignment": aligned_ids,
        "split_manifest_exact": bool(
            sorted(development_groups) == sorted(split["development_group_ids"])
            and sorted(protected_groups) == sorted(split["protected_group_ids"])
        ),
        "finite_metrics": True,
    }
    metrics["finite_metrics"] = _finite(metrics)
    return metrics


def audit_population(metrics: dict[str, Any], access: dict[str, int], config: dict[str, Any]) -> dict[str, Any]:
    gates = config["populationGates"]
    checks = {
        "payloads_exactly_accounted_hashed_and_within_budget": bool(
            metrics["payload_count"] == gates["requiredPayloadCount"]
            and metrics["attempt_count"] == gates["requiredPayloadCount"]
            and metrics["exact_payload_id_accounting"]
            and metrics["successful_payload_retrieval_rate"] == gates["requiredSuccessfulPayloadRetrievalRate"]
            and metrics["raw_hash_coverage"] == gates["requiredRawHashCoverage"]
            and metrics["expected_byte_count_accuracy"] == gates["requiredExpectedByteCountAccuracy"]
            and metrics["declared_digest_accuracy"] == gates["requiredDeclaredDigestAccuracy"]
            and metrics["total_payload_bytes"] <= gates["maximumTotalPayloadBytes"]
        ),
        "uberon_and_W3C_parser_controls_pass": bool(
            metrics["older_term_count"] >= gates["minimumTermsPerUberonRelease"]
            and metrics["newer_term_count"] >= gates["minimumTermsPerUberonRelease"]
            and metrics["w3c_rdfxml_parse_success"] == gates["requiredW3CRDFXMLParseSuccess"]
            and metrics["w3c_rdf_subject_count"] >= gates["minimumW3CRDFSubjectCount"]
        ),
        "population_has_group_disjoint_change_diversity_and_scale": bool(
            metrics["eligible_record_count"] >= gates["minimumEligibleRecordCount"]
            and metrics["eligible_group_count"] >= gates["minimumEligibleGroupCount"]
            and metrics["distinct_primary_change_type_count"] >= gates["minimumDistinctPrimaryChangeTypeCount"]
            and metrics["development_group_count"] >= gates["minimumDevelopmentGroupCount"]
            and metrics["protected_group_count"] >= gates["minimumProtectedGroupCount"]
            and metrics["cross_split_group_overlap_count"] <= gates["maximumCrossSplitGroupOverlapCount"]
            and metrics["split_manifest_exact"]
        ),
        "text_logic_version_space_and_boundary_evidence_are_exact": bool(
            metrics["current_definition_coverage"] == gates["requiredCurrentDefinitionCoverage"]
            and metrics["current_asserted_axiom_coverage"] == gates["requiredCurrentAssertedAxiomCoverage"]
            and metrics["version_space_reconstruction_accuracy"] == gates["requiredVersionSpaceReconstructionAccuracy"]
            and metrics["boundary_witness_coverage"] == gates["requiredBoundaryWitnessCoverage"]
        ),
        "public_truth_identity_and_nonleakage_hold": bool(
            metrics["duplicate_case_id_count"] <= gates["maximumDuplicateCaseIdCount"]
            and metrics["public_source_identifier_leakage_count"] <= gates["maximumPublicSourceIdentifierLeakageCount"]
            and metrics["public_truth_case_alignment"]
        ),
        "metrics_are_finite": metrics["finite_metrics"] == gates["requiredFiniteMetrics"],
    }
    limits = config["accessGates"]
    access_checks = {
        "one_bounded_retrieval_and_population_build": bool(
            access["bounded_retrieval_run_count"] == limits["requiredBoundedRetrievalRunCount"]
            and access["population_build_run_count"] == limits["requiredPopulationBuildRunCount"]
            and access["payload_count"] <= limits["maximumPayloadCount"]
            and access["unlisted_network_request_count"] <= limits["maximumUnlistedNetworkRequestCount"]
        ),
        "protected_model_and_effect_boundaries_zero": bool(
            access["v213_protected_access_count"] <= limits["maximumV213ProtectedAccessCount"]
            and access["protected_downstream_method_evaluation_count"] <= limits["maximumProtectedDownstreamMethodEvaluationCount"]
            and access["protected_manual_semantic_inspection_count"] <= limits["maximumProtectedManualSemanticInspectionCount"]
            and access["model_load_count"] <= limits["maximumModelLoadCount"]
            and access["model_generation_count"] <= limits["maximumModelGenerationCount"]
            and access["model_api_call_count"] <= limits["maximumModelAPICallCount"]
            and access["training_run_count"] <= limits["maximumTrainingRunCount"]
            and access["ontology_registration_count"] <= limits["maximumOntologyRegistrationCount"]
            and access["trusted_state_mutation_count"] <= limits["maximumTrustedStateMutationCount"]
            and access["service_action_count"] <= limits["maximumServiceActionCount"]
            and access["external_side_effect_count_beyond_read_only_retrieval"] <= limits["maximumExternalSideEffectCountBeyondReadOnlyRetrieval"]
            and access["actual_execution_count"] <= limits["maximumActualExecutionCount"]
        ),
    }
    passed = all(checks.values()) and all(access_checks.values())
    return {
        "passed": passed,
        "branch": "V217_DETERMINISTIC_EXTERNAL_RECONSTRUCTION_CONTROLS_ELIGIBLE" if passed else "NEGATIVE_EXTERNAL_PAYLOAD_OR_POPULATION_FEASIBILITY",
        "decision": config["decisionRule"]["ifEveryPayloadPopulationIntegrityAndAccessGatePasses"] if passed else config["decisionRule"]["otherwise"],
        "checks": checks,
        "access_checks": access_checks,
    }

