from __future__ import annotations

import csv
import hashlib
import itertools
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from v10_protocol import file_sha256


MONDO_ID_RE = re.compile(
    r"(?i)(?:https?://purl\.obolibrary\.org/obo/)?MONDO(?::|_)([0-9]+)"
)


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def stable_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _rate(values: Iterable[bool], *, empty: float = 1.0) -> float:
    materialized = list(values)
    return sum(materialized) / len(materialized) if materialized else empty


def _finite(value: Any) -> bool:
    if isinstance(value, dict):
        return all(_finite(item) for item in value.values())
    if isinstance(value, list):
        return all(_finite(item) for item in value)
    if isinstance(value, float):
        return math.isfinite(value)
    return True


def normalize_space(value: str) -> str:
    return " ".join(value.strip().split())


def normalize_mondo_id(value: str) -> str | None:
    match = MONDO_ID_RE.search(value)
    return f"MONDO:{match.group(1)}" if match else None


def redact_source_ids(value: str) -> str:
    return MONDO_ID_RE.sub("[SOURCE_ID]", value)


def strip_obo_comment(value: str) -> str:
    return normalize_space(value.split(" ! ", 1)[0])


def extract_quoted(value: str) -> str:
    match = re.match(r'^"((?:\\.|[^"\\])*)"', value.strip())
    if not match:
        return normalize_space(value)
    return normalize_space(match.group(1).replace(r'\"', '"').replace(r"\\", "\\"))


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
        "name": _first(fields, "name"),
        "definition": extract_quoted(_first(fields, "def")),
        "synonyms": sorted(
            {extract_quoted(value) for value in fields.get("synonym", []) if extract_quoted(value)}
        ),
    }


def status_state(fields: dict[str, list[str]] | None) -> dict[str, Any]:
    if not fields:
        return {"is_obsolete": False, "replaced_by": [], "consider": []}
    return {
        "is_obsolete": _first(fields, "is_obsolete").lower() == "true",
        "replaced_by": sorted(
            {target for value in fields.get("replaced_by", []) if (target := normalize_mondo_id(value))}
        ),
        "consider": sorted(
            {target for value in fields.get("consider", []) if (target := normalize_mondo_id(value))}
        ),
    }


def asserted_values(fields: dict[str, list[str]] | None, keys: list[str]) -> list[str]:
    if not fields:
        return []
    values = []
    for key in keys:
        for value in fields.get(key, []):
            normalized = strip_obo_comment(value)
            if normalized:
                values.append(f"{key}={normalized}")
    return sorted(set(values))


def semantic_state(fields: dict[str, list[str]], config: dict[str, Any]) -> dict[str, Any]:
    parser = config["parserDesign"]
    return {
        **text_state(fields),
        "asserted_axioms": asserted_values(fields, parser["logicalFields"]),
        "asserted_mappings": asserted_values(fields, parser["mappingFields"]),
        "lifecycle": status_state(fields),
    }


def state_class(signature: dict[str, Any]) -> str:
    return "C_" + stable_hash(signature)[:24]


def absence_signature(release_role: str) -> dict[str, Any]:
    return {"presence": "ABSENT", "release_role": release_role}


def event_types(
    older: dict[str, list[str]] | None,
    newer: dict[str, list[str]] | None,
    older_candidate: bool,
    newer_candidate: bool,
    config: dict[str, Any],
) -> list[str]:
    if older is None and newer is not None:
        changes = ["ADDED"]
    elif older is not None and newer is None:
        changes = ["REMOVED"]
    elif older is None and newer is None:
        changes = []
    else:
        assert older is not None and newer is not None
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
        parser = config["parserDesign"]
        if asserted_values(older, parser["logicalFields"]) != asserted_values(newer, parser["logicalFields"]):
            changes.append("LOGICAL_AXIOM_CHANGED")
        if asserted_values(older, parser["mappingFields"]) != asserted_values(newer, parser["mappingFields"]):
            changes.append("MAPPING_CHANGED")
        if old_status["is_obsolete"] != new_status["is_obsolete"]:
            changes.append("OBSOLETION_CHANGED")
        if old_status["replaced_by"] != new_status["replaced_by"] or old_status["consider"] != new_status["consider"]:
            changes.append("REPLACEMENT_CHANGED")
    if older_candidate != newer_candidate:
        changes.append("OBSOLETION_CANDIDATE_STATUS_CHANGED")
    return sorted(set(changes))


def parse_tsv(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8", newline="") as stream:
            rows = list(csv.reader(stream, delimiter="\t"))
    except (OSError, UnicodeError, csv.Error) as error:
        return {"parse_success": False, "header": [], "row_count": 0, "mondo_ids": [], "error": f"{type(error).__name__}: {error}"}
    header = rows[0] if rows else []
    body = [row for row in rows[1:] if any(cell.strip() for cell in row)]
    mondo_ids = []
    for row in body:
        found = next((normalize_mondo_id(cell) for cell in row if normalize_mondo_id(cell)), None)
        if found:
            mondo_ids.append(found)
    return {
        "parse_success": bool(header),
        "header": header,
        "row_count": len(body),
        "mondo_ids": sorted(set(mondo_ids)),
        "mondo_id_row_count": len(mondo_ids),
        "mondo_id_unique": len(mondo_ids) == len(set(mondo_ids)),
        "error": None if header else "missing header",
    }


def release_summary_control(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    patterns = {
        "NEW_TERMS": r"(?i)(new terms|classes added)",
        "RENAMED_TERMS": r"(?i)terms renamed",
        "TEXT_DEFINITIONS": r"(?i)text definitions? (added|changed)",
        "OBSOLETIONS_WITH_REPLACEMENT": r"(?i)terms obsoleted with replacement",
    }
    categories = sorted(key for key, pattern in patterns.items() if re.search(pattern, text))
    return {
        "parse_success": bool(text.strip()),
        "byte_count": len(text.encode("utf-8")),
        "categories": categories,
    }


class UnionFind:
    def __init__(self, values: Iterable[str]) -> None:
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        self.parent.setdefault(value, value)
        if self.parent[value] != value:
            self.parent[value] = self.find(self.parent[value])
        return self.parent[value]

    def union(self, left: str, right: str) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        smaller, larger = sorted([left_root, right_root])
        self.parent[larger] = smaller


def _primary(events: set[str], precedence: list[str]) -> str:
    for event in precedence:
        if event in events:
            return event
    raise ValueError(f"no primary event for {sorted(events)}")


def _surface_text(member_ids: list[str], older_terms: dict[str, Any], newer_terms: dict[str, Any]) -> str:
    candidates: list[str] = []
    for term_id in member_ids:
        old_text = text_state(older_terms.get(term_id))
        new_text = text_state(newer_terms.get(term_id))
        candidates.extend([old_text["name"], new_text["name"]])
        candidates.extend(old_text["synonyms"])
        candidates.extend(new_text["synonyms"])
    return next((redact_source_ids(value) for value in candidates if value), "")


def _witnesses(signatures: dict[str, dict[str, Any]], candidates: list[str]) -> list[dict[str, str]]:
    witnesses = []
    for left, right in itertools.combinations(sorted(candidates), 2):
        left_signature = signatures[left]
        right_signature = signatures[right]
        witness = ""
        for field in sorted(set(left_signature) | set(right_signature)):
            left_value = left_signature.get(field)
            right_value = right_signature.get(field)
            if canonical_json(left_value) != canonical_json(right_value):
                witness = f"{field}:{canonical_json(left_value)} != {canonical_json(right_value)}"
                break
        witnesses.append({"left": left, "right": right, "witness": witness})
    return witnesses


def build_population_records(
    older_terms: dict[str, dict[str, list[str]]],
    newer_terms: dict[str, dict[str, list[str]]],
    older_candidates: set[str],
    newer_candidates: set[str],
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    all_ids = set(older_terms) | set(newer_terms) | older_candidates | newer_candidates
    union_find = UnionFind(all_ids)
    for fields in list(older_terms.values()) + list(newer_terms.values()):
        if not fields.get("id"):
            continue
        source = normalize_space(fields["id"][0])
        lifecycle = status_state(fields)
        for target in lifecycle["replaced_by"] + lifecycle["consider"]:
            union_find.union(source, target)
    families: dict[str, list[str]] = defaultdict(list)
    for term_id in sorted(union_find.parent):
        families[union_find.find(term_id)].append(term_id)

    event_by_id: dict[str, list[str]] = {}
    for term_id in sorted(all_ids):
        event_by_id[term_id] = event_types(
            older_terms.get(term_id),
            newer_terms.get(term_id),
            term_id in older_candidates,
            term_id in newer_candidates,
            config,
        )

    provisional: list[tuple[dict[str, Any], dict[str, Any]]] = []
    family_summaries: dict[str, dict[str, Any]] = {}
    exclusions: Counter[str] = Counter()
    precedence = config["eventDesign"]["primaryEventPrecedence"]
    for root, member_ids in sorted(families.items()):
        events = {event for term_id in member_ids for event in event_by_id.get(term_id, [])}
        if not events:
            exclusions["unchanged_family"] += 1
            continue
        surface = _surface_text(member_ids, older_terms, newer_terms)
        if not surface:
            exclusions["no_surface_text"] += 1
            continue
        signatures: dict[str, dict[str, Any]] = {}
        old_classes: list[str] = []
        current_classes: list[str] = []
        for term_id in member_ids:
            if term_id in older_terms:
                signature = semantic_state(older_terms[term_id], config)
                class_id = state_class(signature)
                signatures.setdefault(class_id, signature)
                old_classes.append(class_id)
            if term_id in newer_terms:
                signature = semantic_state(newer_terms[term_id], config)
                class_id = state_class(signature)
                signatures.setdefault(class_id, signature)
                current_classes.append(class_id)
        if not old_classes:
            signature = absence_signature("OLDER")
            class_id = state_class(signature)
            signatures[class_id] = signature
            old_classes.append(class_id)
        if not current_classes:
            signature = absence_signature("CURRENT")
            class_id = state_class(signature)
            signatures[class_id] = signature
            current_classes.append(class_id)
        replacement_classes: list[str] = []
        replacement_ids: set[str] = set()
        for term_id in member_ids:
            for fields in (older_terms.get(term_id), newer_terms.get(term_id)):
                if fields:
                    replacement_ids.update(status_state(fields)["replaced_by"])
        for target in sorted(replacement_ids):
            if target in newer_terms:
                signature = semantic_state(newer_terms[target], config)
                class_id = state_class(signature)
                signatures.setdefault(class_id, signature)
                replacement_classes.append(class_id)
        all_version_classes = sorted(set(old_classes + current_classes + replacement_classes))
        current_present = any(term_id in newer_terms for term_id in member_ids)
        current_obsolete = any(status_state(newer_terms.get(term_id))["is_obsolete"] for term_id in member_ids if term_id in newer_terms)
        if not current_present:
            expressibility = "OLD_ONLY"
            declared_classes: list[str] = []
            declared_decision = "ABSTAIN_NOT_EXPRESSIBLE"
        elif current_obsolete and replacement_classes:
            expressibility = "CURRENT_OBSOLETE_WITH_REPLACEMENT"
            declared_classes = sorted(set(replacement_classes))
            declared_decision = "FOLLOW_ASSERTED_REPLACEMENT"
        elif current_obsolete:
            expressibility = "CURRENT_OBSOLETE_WITHOUT_REPLACEMENT"
            declared_classes = []
            declared_decision = "ABSTAIN_NOT_EXPRESSIBLE"
        elif not any(term_id in older_terms for term_id in member_ids):
            expressibility = "NEW_ONLY"
            declared_classes = sorted(set(current_classes))
            declared_decision = "RESOLVE_CURRENT_STATE"
        else:
            expressibility = "OLD_AND_CURRENT"
            declared_classes = sorted(set(current_classes))
            declared_decision = "RESOLVE_CURRENT_STATE"
        family_id = "F_" + stable_hash(member_ids)[:24]
        group_id = "G_" + stable_hash([config["experiment"], family_id])[:24]
        primary = _primary(events, precedence)
        family_summaries[family_id] = {
            "family_id": family_id,
            "group_id": group_id,
            "member_ids": member_ids,
            "event_types": sorted(events),
            "primary_event_type": primary,
            "all_version_class_ids": all_version_classes,
            "declared_current_class_ids": declared_classes,
            "expressibility_state": expressibility,
        }
        for mode in config["populationDesign"]["evidenceModes"]:
            if mode == "VERSION_UNSPECIFIED":
                candidates = all_version_classes
                correct_decision = "PRESERVE_VERSION_SPACE_OR_CLARIFY"
                release_evidence = None
                evidence_state = "AMBIGUOUS" if len(candidates) > 1 else "SINGLETON"
            else:
                candidates = declared_classes
                correct_decision = declared_decision
                release_evidence = "v2026-08-04"
                evidence_state = "NOT_EXPRESSIBLE" if not candidates else ("SINGLETON" if len(candidates) == 1 else "SET_VALUED")
            case_id = "E_" + stable_hash([family_id, mode])[:24]
            public = {
                "case_id": case_id,
                "group_id": group_id,
                "split": "UNASSIGNED",
                "task": "RETROSPECTIVE_MONDO_REPRESENTATIONAL_DIAGNOSIS",
                "surface_text": surface,
                "release_evidence": release_evidence,
                "requested_semantic_target": "CURRENT_PUBLISHED_STATE",
                "evidence_mode": mode,
            }
            losses = {correct_decision: 0.0}
            if mode == "VERSION_UNSPECIFIED" and len(candidates) > 1:
                losses["FORCE_SINGLETON"] = 1.0
            elif mode == "CURRENT_RELEASE_DECLARED":
                losses["PRESERVE_VERSION_SPACE_OR_CLARIFY"] = 0.1
                losses["MISMATCHED_LIFECYCLE_ACTION"] = 1.0
            truth = {
                "case_id": case_id,
                "group_id": group_id,
                "split": "UNASSIGNED",
                "source_concept_ids": member_ids,
                "event_types": sorted(events),
                "primary_event_type": primary,
                "old_state_class_ids": sorted(set(old_classes)),
                "current_state_class_ids": sorted(set(current_classes)),
                "replacement_target_class_ids": sorted(set(replacement_classes)),
                "all_version_class_ids": all_version_classes,
                "declared_current_class_ids": declared_classes,
                "candidate_class_ids": candidates,
                "equivalence_classes": signatures,
                "expressibility_state": expressibility,
                "evidence_state": evidence_state,
                "boundary_witness": _witnesses(signatures, candidates),
                "correct_decision": correct_decision,
                "decision_consequence": {"action_loss": losses},
            }
            provisional.append((public, truth))

    family_ids = sorted(family_summaries, key=lambda value: stable_hash([config["experiment"], value]))
    protected = {family_id for index, family_id in enumerate(family_ids) if index % 4 == 3}
    group_split = {
        summary["group_id"]: ("PROTECTED" if family_id in protected else "DEVELOPMENT")
        for family_id, summary in family_summaries.items()
    }
    public_records: list[dict[str, Any]] = []
    truth_records: list[dict[str, Any]] = []
    for public, truth in provisional:
        split = group_split[public["group_id"]]
        public["split"] = split
        truth["split"] = split
        public_records.append(public)
        truth_records.append(truth)
    public_records.sort(key=lambda value: value["case_id"])
    truth_records.sort(key=lambda value: value["case_id"])
    truth_by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for truth in truth_records:
        family_key = stable_hash(truth["source_concept_ids"])
        truth_by_family[family_key].append(truth)
    text_events = {"NAME_CHANGED", "DEFINITION_CHANGED", "SYNONYM_CHANGED"}
    lifecycle_events = {"OBSOLETION_CHANGED", "REPLACEMENT_CHANGED", "OBSOLETION_CANDIDATE_STATUS_CHANGED"}
    ambiguous_families = {
        stable_hash(truth["source_concept_ids"])
        for truth in truth_records
        if truth["evidence_state"] == "AMBIGUOUS"
    }
    decision_contrast = 0
    for records in truth_by_family.values():
        if len({record["correct_decision"] for record in records}) > 1:
            decision_contrast += 1
    manifest = {
        "schema_version": "218-mondo-population-build-manifest",
        "older_term_count": len(older_terms),
        "newer_term_count": len(newer_terms),
        "raw_event_type_counts": dict(sorted(Counter(event for events in event_by_id.values() for event in events).items())),
        "exclusion_counts": dict(sorted(exclusions.items())),
        "eligible_record_count": len(public_records),
        "eligible_concept_family_count": len(family_summaries),
        "development_group_ids": sorted(summary["group_id"] for family_id, summary in family_summaries.items() if family_id not in protected),
        "protected_group_ids": sorted(summary["group_id"] for family_id, summary in family_summaries.items() if family_id in protected),
        "primary_event_type_counts": dict(sorted(Counter(summary["primary_event_type"] for summary in family_summaries.values()).items())),
        "distinct_primary_event_type_count": len({summary["primary_event_type"] for summary in family_summaries.values()}),
        "text_change_family_count": sum(bool(set(summary["event_types"]) & text_events) for summary in family_summaries.values()),
        "lifecycle_event_family_count": sum(bool(set(summary["event_types"]) & lifecycle_events) for summary in family_summaries.values()),
        "ambiguous_unspecified_family_count": len(ambiguous_families),
        "decision_contrast_family_count": decision_contrast,
    }
    return public_records, truth_records, manifest


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
    successful = [row for row in attempts if row["success"]]
    raw_checks = []
    byte_checks = []
    digest_checks = []
    for row in successful:
        payload = payloads[row["payload_id"]]
        path = project_root / payload["rawPath"]
        raw_checks.append(path.is_file() and file_sha256(path) == row["sha256"])
        byte_checks.append(path.is_file() and path.stat().st_size == payload["expectedByteCount"] == row["byte_count"])
        digest_checks.append(row["sha256"] == payload["declaredSha256"])
    public_by_id = {record["case_id"]: record for record in public_records}
    truth_by_id = {record["case_id"]: record for record in truth_records}
    duplicate_case_ids = len(public_records) - len(public_by_id) + len(truth_records) - len(truth_by_id)
    development_groups = {record["group_id"] for record in public_records if record["split"] == "DEVELOPMENT"}
    protected_groups = {record["group_id"] for record in public_records if record["split"] == "PROTECTED"}
    state_checks = []
    version_checks = []
    witness_checks = []
    decision_checks = []
    for public in public_records:
        truth = truth_by_id[public["case_id"]]
        state_checks.extend(class_id == state_class(signature) for class_id, signature in truth["equivalence_classes"].items())
        expected = truth["all_version_class_ids"] if public["evidence_mode"] == "VERSION_UNSPECIFIED" else truth["declared_current_class_ids"]
        version_checks.append(truth["candidate_class_ids"] == expected)
        if len(expected) > 1:
            expected_pairs = len(list(itertools.combinations(expected, 2)))
            witness_checks.append(len(truth["boundary_witness"]) == expected_pairs and all(row["witness"] for row in truth["boundary_witness"]))
        losses = truth["decision_consequence"]["action_loss"]
        decision_checks.append(losses.get(truth["correct_decision"]) == 0.0 and all(value >= 0.0 for value in losses.values()) and (len(losses) == 1 or any(value > 0.0 for action, value in losses.items() if action != truth["correct_decision"])))
    source_leakage = sum(len(MONDO_ID_RE.findall(canonical_json(record))) for record in public_records)
    metrics = {
        "payload_count": len(payloads),
        "attempt_count": len(attempts),
        "exact_payload_id_accounting": sorted(row["payload_id"] for row in attempts) == sorted(payloads),
        "successful_payload_retrieval_rate": len(successful) / len(payloads),
        "raw_hash_coverage": _rate(raw_checks),
        "expected_byte_count_accuracy": _rate(byte_checks),
        "declared_digest_accuracy": _rate(digest_checks),
        "total_payload_bytes": sum(row["byte_count"] for row in successful),
        "older_term_count": parser_control["older_term_count"],
        "newer_term_count": parser_control["newer_term_count"],
        "unique_term_id_rate": parser_control["unique_term_id_rate"],
        "remote_import_resolution_count": parser_control["remote_import_resolution_count"],
        "tabular_control_parse_rate": parser_control["tabular_control_parse_rate"],
        "new_term_control_agreement": parser_control["new_term_control_agreement"],
        "changed_term_control_precision": parser_control["changed_term_control_precision"],
        "release_summary_category_coverage": parser_control["release_summary_category_coverage"],
        "eligible_concept_family_count": population_manifest["eligible_concept_family_count"],
        "eligible_record_count": len(public_records),
        "development_group_count": len(development_groups),
        "protected_group_count": len(protected_groups),
        "distinct_primary_event_type_count": population_manifest["distinct_primary_event_type_count"],
        "primary_event_type_counts": population_manifest["primary_event_type_counts"],
        "text_change_family_count": population_manifest["text_change_family_count"],
        "lifecycle_event_family_count": population_manifest["lifecycle_event_family_count"],
        "ambiguous_unspecified_family_count": population_manifest["ambiguous_unspecified_family_count"],
        "decision_contrast_family_count": population_manifest["decision_contrast_family_count"],
        "semantic_state_reconstruction_accuracy": _rate(state_checks),
        "version_space_reconstruction_accuracy": _rate(version_checks),
        "boundary_witness_coverage": _rate(witness_checks),
        "decision_consequence_coverage": _rate(decision_checks),
        "cross_split_group_overlap_count": len(development_groups & protected_groups),
        "duplicate_case_id_count": duplicate_case_ids,
        "public_source_identifier_leakage_count": source_leakage,
        "public_truth_case_alignment": set(public_by_id) == set(truth_by_id),
        "split_manifest_exact": sorted(development_groups) == sorted(split["development_group_ids"]) and sorted(protected_groups) == sorted(split["protected_group_ids"]),
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
        "parsers_and_published_controls_agree": bool(
            metrics["older_term_count"] >= gates["minimumParsedTermCountPerRelease"]
            and metrics["newer_term_count"] >= gates["minimumParsedTermCountPerRelease"]
            and metrics["unique_term_id_rate"] == gates["requiredUniqueTermIdRate"]
            and metrics["remote_import_resolution_count"] == gates["requiredRemoteImportResolutionCount"]
            and metrics["tabular_control_parse_rate"] == gates["requiredTabularControlParseRate"]
            and metrics["new_term_control_agreement"] == gates["requiredNewTermControlAgreement"]
            and metrics["changed_term_control_precision"] == gates["requiredChangedTermControlPrecision"]
            and metrics["release_summary_category_coverage"] == gates["requiredReleaseSummaryCategoryCoverage"]
        ),
        "population_has_direct_event_lifecycle_ambiguity_and_split_scale": bool(
            metrics["eligible_concept_family_count"] >= gates["minimumEligibleConceptFamilyCount"]
            and metrics["eligible_record_count"] >= gates["minimumEligibleRecordCount"]
            and metrics["distinct_primary_event_type_count"] >= gates["minimumDistinctPrimaryEventTypeCount"]
            and metrics["text_change_family_count"] >= gates["minimumTextChangeFamilyCount"]
            and metrics["lifecycle_event_family_count"] >= gates["minimumLifecycleEventFamilyCount"]
            and metrics["ambiguous_unspecified_family_count"] >= gates["minimumAmbiguousUnspecifiedFamilyCount"]
            and metrics["decision_contrast_family_count"] >= gates["minimumDecisionContrastFamilyCount"]
            and metrics["development_group_count"] >= gates["minimumDevelopmentGroupCount"]
            and metrics["protected_group_count"] >= gates["minimumProtectedGroupCount"]
        ),
        "representational_diagnosis_and_decision_oracles_are_exact": bool(
            metrics["semantic_state_reconstruction_accuracy"] == gates["requiredSemanticStateReconstructionAccuracy"]
            and metrics["version_space_reconstruction_accuracy"] == gates["requiredVersionSpaceReconstructionAccuracy"]
            and metrics["boundary_witness_coverage"] == gates["requiredBoundaryWitnessCoverage"]
            and metrics["decision_consequence_coverage"] == gates["requiredDecisionConsequenceCoverage"]
        ),
        "public_truth_split_and_finite_integrity_hold": bool(
            metrics["cross_split_group_overlap_count"] <= gates["maximumCrossSplitGroupOverlapCount"]
            and metrics["duplicate_case_id_count"] <= gates["maximumDuplicateCaseIdCount"]
            and metrics["public_source_identifier_leakage_count"] <= gates["maximumPublicSourceIdentifierLeakageCount"]
            and metrics["public_truth_case_alignment"] == gates["requiredPublicTruthCaseAlignment"]
            and metrics["split_manifest_exact"] == gates["requiredSplitManifestExact"]
            and metrics["finite_metrics"] == gates["requiredFiniteMetrics"]
        ),
    }
    limits = config["accessGates"]
    access_checks = {
        "one_bounded_retrieval_and_population_build": bool(
            access["bounded_retrieval_run_count"] == limits["requiredBoundedRetrievalRunCount"]
            and access["population_build_run_count"] == limits["requiredPopulationBuildRunCount"]
            and access["unlisted_network_request_count"] <= limits["maximumUnlistedNetworkRequestCount"]
            and access["payload_count"] <= limits["maximumPayloadCount"]
            and access["remote_import_resolution_count"] <= limits["maximumRemoteImportResolutionCount"]
        ),
        "protected_model_and_effect_boundaries_zero": bool(
            access["v216_protected_access_count"] <= limits["maximumV216ProtectedAccessCount"]
            and access["v213_protected_access_count"] <= limits["maximumV213ProtectedAccessCount"]
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
        "branch": "MONDO_REPRESENTATIONAL_POPULATION_ELIGIBLE" if passed else "NEGATIVE_MONDO_PAYLOAD_CONTROL_OR_POPULATION_FEASIBILITY",
        "decision": config["decisionRule"]["ifEveryPayloadControlPopulationIntegrityAndAccessGatePasses"] if passed else config["decisionRule"]["otherwise"],
        "checks": checks,
        "access_checks": access_checks,
    }
