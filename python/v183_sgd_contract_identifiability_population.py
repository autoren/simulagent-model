from __future__ import annotations

from collections import Counter, defaultdict
from io import BytesIO
import hashlib
import json
import re
import tarfile
from typing import Any

from v93_open_set_source import canonical_sha256, hash_order
from v124_sgd_source_feasibility import service_domain
from v133_sgd_capability_label_identifiability import (
    normalize_description,
    normalize_name,
)


ROOT = "dstc8-schema-guided-dialogue-e852981ae34990f4358979625854259302feaa78"


def _sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _definition_id(service: str, intent: str) -> str:
    return f"{service}::{intent}"


def _parse_candidate_id(candidate_id: str) -> tuple[str, str, int, str, str]:
    pieces = candidate_id.split("::")
    if len(pieces) != 6 or pieces[0] != "sgd":
        raise ValueError(f"invalid SGD candidate identifier: {candidate_id}")
    partition, dialogue_id, turn_text, service, intent = pieces[1:]
    return partition, dialogue_id, int(turn_text), service, intent


def _schema_contract(service: dict[str, Any], intent: dict[str, Any]) -> dict[str, Any]:
    slot_rows = []
    for slot in sorted(service["slots"], key=lambda row: row["name"]):
        normalized_values = sorted(
            normalize_description(str(value)) for value in slot.get("possible_values", [])
        )
        slot_rows.append(
            {
                "name": slot["name"],
                "description_sha256": hashlib.sha256(
                    normalize_description(slot["description"]).encode()
                ).hexdigest(),
                "is_categorical": bool(slot["is_categorical"]),
                "possible_values_sha256": _sha(normalized_values),
                "possible_value_count": len(normalized_values),
            }
        )
    semantic_payload = {
        "domain": service_domain(service["service_name"]),
        "service_description_sha256": hashlib.sha256(
            normalize_description(service["description"]).encode()
        ).hexdigest(),
        "normalized_intent_name": normalize_name(intent["name"]),
        "intent_description_sha256": hashlib.sha256(
            normalize_description(intent["description"]).encode()
        ).hexdigest(),
        "is_transactional": bool(intent["is_transactional"]),
        "required_slots": sorted(intent.get("required_slots", [])),
        "optional_slots": sorted(intent.get("optional_slots", [])),
        "result_slots": sorted(intent.get("result_slots", [])),
        "slots": slot_rows,
    }
    digest = _sha(semantic_payload)
    return {
        "capability_contract_id": f"C_{digest[:24]}",
        "capability_contract_sha256": digest,
        "semantic_payload": semantic_payload,
    }


def _frame_signature(frame: dict[str, Any]) -> dict[str, Any]:
    state = frame.get("state", {})
    action_slots = sorted(
        {
            str(action.get("slot"))
            for action in frame.get("actions", [])
            if action.get("slot") not in (None, "", "intent")
        }
    )
    state_slots = sorted(str(value) for value in state.get("slot_values", {}).keys())
    requested_slots = sorted(str(value) for value in state.get("requested_slots", []))
    span_slots = sorted(
        {
            str(value.get("slot"))
            for value in frame.get("slots", [])
            if value.get("slot") not in (None, "")
        }
    )
    observed = sorted(set(action_slots + state_slots + requested_slots + span_slots))
    return {
        "normalized_active_intent_name": normalize_name(state["active_intent"]),
        "current_action_slot_names": action_slots,
        "current_state_slot_names": state_slots,
        "current_requested_slot_names": requested_slots,
        "current_span_slot_names": span_slots,
        "all_observed_slot_names": observed,
    }


def read_structured_source(
    archive_bytes: bytes, selected_candidate_ids: set[str]
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    selected_parts = {_parse_candidate_id(value)[0] for value in selected_candidate_ids}
    if selected_parts != {"dev"}:
        raise ValueError("V183 expects only the frozen SGD dev population")
    definitions: dict[str, dict[str, Any]] = {}
    frames: dict[str, dict[str, Any]] = {}
    wanted_by_dialogue: dict[str, list[tuple[str, int, str, str]]] = defaultdict(list)
    for candidate_id in selected_candidate_ids:
        _, dialogue_id, turn_index, service, intent = _parse_candidate_id(candidate_id)
        wanted_by_dialogue[dialogue_id].append(
            (candidate_id, turn_index, service, intent)
        )

    with tarfile.open(fileobj=BytesIO(archive_bytes), mode="r:gz") as archive:
        members = [member for member in archive.getmembers() if member.isfile()]
        if any(member.name.startswith("/") or ".." in member.name.split("/") for member in members):
            raise ValueError("unsafe archive member")
        schema_matches = [
            member for member in members if member.name == f"{ROOT}/dev/schema.json"
        ]
        if len(schema_matches) != 1:
            raise ValueError("expected one frozen dev schema")
        schema_handle = archive.extractfile(schema_matches[0])
        if schema_handle is None:
            raise ValueError("unreadable dev schema")
        for service in json.loads(schema_handle.read()):
            for intent in service["intents"]:
                source_id = _definition_id(service["service_name"], intent["name"])
                row = _schema_contract(service, intent)
                definitions[source_id] = {
                    "source_definition_id": source_id,
                    "service": service["service_name"],
                    "intent": intent["name"],
                    **row,
                }

        dialogue_members = sorted(
            (
                member
                for member in members
                if re.fullmatch(rf"{re.escape(ROOT)}/dev/dialogues_\d+\.json", member.name)
            ),
            key=lambda member: member.name,
        )
        for member in dialogue_members:
            handle = archive.extractfile(member)
            if handle is None:
                raise ValueError("unreadable dialogue member")
            for dialogue in json.loads(handle.read()):
                dialogue_id = str(dialogue["dialogue_id"])
                wanted = wanted_by_dialogue.get(dialogue_id)
                if not wanted:
                    continue
                for candidate_id, turn_index, service, intent in wanted:
                    turn = dialogue["turns"][turn_index]
                    matches = [
                        frame
                        for frame in turn.get("frames", [])
                        if frame.get("service") == service
                        and frame.get("state", {}).get("active_intent") == intent
                    ]
                    if len(matches) != 1:
                        raise ValueError(f"could not reconstruct exact frame: {candidate_id}")
                    frames[candidate_id] = {
                        "source_definition_id": _definition_id(service, intent),
                        "frame_signature": _frame_signature(matches[0]),
                    }

    if set(frames) != selected_candidate_ids:
        missing = sorted(selected_candidate_ids - set(frames))[:3]
        raise ValueError(f"selected source reconstruction incomplete: {missing}")
    return definitions, frames


def _truth_kind(fixture: dict[str, Any]) -> str:
    mapping = {
        "known": "KNOWN",
        "novel_valid": "PROVISIONAL",
        "unsupported": "UNSUPPORTED",
    }
    value = fixture["derived_class_label"]
    if value not in mapping:
        raise ValueError(f"unsupported observed truth kind: {value}")
    return mapping[value]


def _build_contract_catalog(
    definitions: dict[str, dict[str, Any]], fixtures: list[dict[str, Any]]
) -> tuple[dict[str, Any], dict[str, str]]:
    kinds_by_definition: dict[str, set[str]] = defaultdict(set)
    for fixture in fixtures:
        if fixture["observation_available"]:
            kinds_by_definition[_definition_id(fixture["service"], fixture["intent"])].add(
                _truth_kind(fixture)
            )
    rows_by_contract: dict[str, list[dict[str, Any]]] = defaultdict(list)
    definition_to_contract: dict[str, str] = {}
    for source_definition_id, truth_kinds in sorted(kinds_by_definition.items()):
        definition = definitions[source_definition_id]
        contract_id = definition["capability_contract_id"]
        rows_by_contract[contract_id].append(definition)
        definition_to_contract[source_definition_id] = contract_id
    contracts = []
    for contract_id, rows in sorted(rows_by_contract.items()):
        truth_kinds = sorted(
            {
                kind
                for row in rows
                for kind in kinds_by_definition[row["source_definition_id"]]
            }
        )
        payloads = {canonical_sha256(row["semantic_payload"]) for row in rows}
        if len(payloads) != 1:
            raise AssertionError("contract identifier collision")
        contracts.append(
            {
                "capability_contract_id": contract_id,
                "capability_contract_sha256": rows[0]["capability_contract_sha256"],
                "source_definition_ids": sorted(
                    row["source_definition_id"] for row in rows
                ),
                "truth_kinds": truth_kinds,
                "mixed_truth_kind": len(truth_kinds) != 1,
                "normalized_intent_name": rows[0]["semantic_payload"][
                    "normalized_intent_name"
                ],
                "slot_names": sorted(
                    slot["name"] for slot in rows[0]["semantic_payload"]["slots"]
                ),
                "semantic_payload": rows[0]["semantic_payload"],
            }
        )
    return {
        "contract_count": len(contracts),
        "contracts": contracts,
        "contract_catalog_sha256": canonical_sha256(contracts),
        "contains_utterance_dialogue_or_slot_values": False,
    }, definition_to_contract


def _assign_roles(fixtures: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, str]:
    split = config["roleSplit"]
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for fixture in fixtures:
        groups[(fixture["truth_choice_id"], fixture["presented_candidate_choice_id"])].append(fixture)
    roles: dict[str, str] = {}
    for (truth, candidate), rows in sorted(groups.items()):
        ordered = sorted(
            rows,
            key=lambda row: hash_order(
                split["baseSalt"], truth, candidate, row["fixture_id"]
            ),
        )
        if len(ordered) != split["recordsPerTruthCandidateCell"]:
            raise ValueError("unexpected frozen V134 cell size")
        for row in ordered[: split["developmentRecordsPerCell"]]:
            roles[row["fixture_id"]] = "development"
        for row in ordered[split["developmentRecordsPerCell"] :]:
            roles[row["fixture_id"]] = "protected"
    return roles


def build_population(
    archive_bytes: bytes,
    source_catalog: dict[str, Any],
    source_population: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    del source_catalog  # V134 catalog identity is checked by the outer lock; full contracts supersede composites.
    fixtures = source_population["fixtures"]
    selected_ids = {
        fixture["candidate_id"]
        for fixture in fixtures
        if fixture["observation_available"]
    }
    definitions, frames = read_structured_source(archive_bytes, selected_ids)
    catalog, definition_to_contract = _build_contract_catalog(definitions, fixtures)
    contract_rows = {
        row["capability_contract_id"]: row for row in catalog["contracts"]
    }
    roles = _assign_roles(fixtures, config)
    records = []
    public = {"development": [], "protected": []}
    for fixture in sorted(fixtures, key=lambda row: row["fixture_id"]):
        role = roles[fixture["fixture_id"]]
        opaque = hashlib.sha256(
            f"{config['roleSplit']['baseSalt']}::{fixture['fixture_id']}".encode()
        ).hexdigest()[:24]
        record_id = f"v183::{opaque}"
        if fixture["observation_available"]:
            source = frames[fixture["candidate_id"]]
            target_contract = definition_to_contract[source["source_definition_id"]]
            signature = source["frame_signature"]
            observed_slots = set(signature["all_observed_slot_names"])
            compatible = sorted(
                row["capability_contract_id"]
                for row in catalog["contracts"]
                if row["normalized_intent_name"]
                == signature["normalized_active_intent_name"]
                and observed_slots.issubset(row["slot_names"])
            )
            compatible_kinds = sorted(
                {kind for cid in compatible for kind in contract_rows[cid]["truth_kinds"]}
            )
            has_mixed = any(contract_rows[cid]["mixed_truth_kind"] for cid in compatible)
            if len(compatible) == 0:
                status = "INVALID_SOURCE_RECORD"
            elif len(compatible) == 1 and not has_mixed:
                status = "IDENTIFIABLE"
            else:
                status = "INSUFFICIENT"
            truth_kind = _truth_kind(fixture)
            target_retained = target_contract in compatible
            evaluation_choice = target_contract if status == "IDENTIFIABLE" else "A00"
            hidden = {
                "record_id": record_id,
                "role": role,
                "source_candidate_id": fixture["candidate_id"],
                "source_definition_id": source["source_definition_id"],
                "truth_contract_id": target_contract,
                "truth_kind": truth_kind,
                "source_truth_choice_id": fixture["truth_choice_id"],
                "frame_signature": signature,
                "compatible_contract_ids": compatible,
                "compatible_truth_kinds": compatible_kinds,
                "compatible_contract_count": len(compatible),
                "target_contract_retained": target_retained,
                "identifiability_status": status,
                "evaluation_choice": evaluation_choice,
                "presented_candidate_choice_id": fixture[
                    "presented_candidate_choice_id"
                ],
                "observation_available": True,
            }
        else:
            hidden = {
                "record_id": record_id,
                "role": role,
                "source_candidate_id": None,
                "source_definition_id": None,
                "truth_contract_id": None,
                "truth_kind": "INSUFFICIENT_CONTROL",
                "source_truth_choice_id": fixture["truth_choice_id"],
                "frame_signature": None,
                "compatible_contract_ids": sorted(contract_rows),
                "compatible_truth_kinds": sorted(
                    {kind for row in catalog["contracts"] for kind in row["truth_kinds"]}
                ),
                "compatible_contract_count": len(contract_rows),
                "target_contract_retained": True,
                "identifiability_status": "INSUFFICIENT",
                "evaluation_choice": "A00",
                "presented_candidate_choice_id": fixture[
                    "presented_candidate_choice_id"
                ],
                "observation_available": False,
            }
        records.append(hidden)
        public[role].append(
            {
                "record_id": record_id,
                "role": role,
                "observation_available": hidden["observation_available"],
                "presented_candidate_choice_id": hidden[
                    "presented_candidate_choice_id"
                ],
            }
        )

    role_summary: dict[str, Any] = {}
    for role in ("development", "protected"):
        subset = [row for row in records if row["role"] == role]
        source = [row for row in subset if row["observation_available"]]
        identifiable = [row for row in source if row["identifiability_status"] == "IDENTIFIABLE"]
        cells = Counter(
            (
                row["source_truth_choice_id"],
                row["presented_candidate_choice_id"],
            )
            for row in subset
        )
        role_summary[role] = {
            "fixture_count": len(subset),
            "source_record_count": len(source),
            "missing_control_count": len(subset) - len(source),
            "cell_count": len(cells),
            "records_per_cell_values": sorted(set(cells.values())),
            "identifiable_source_record_count": len(identifiable),
            "insufficient_source_record_count": sum(
                row["identifiability_status"] == "INSUFFICIENT" for row in source
            ),
            "invalid_source_record_count": sum(
                row["identifiability_status"] == "INVALID_SOURCE_RECORD"
                for row in source
            ),
            "identifiable_by_truth_kind": dict(
                sorted(Counter(row["truth_kind"] for row in identifiable).items())
            ),
            "compatibility_size_counts": dict(
                sorted(
                    Counter(str(row["compatible_contract_count"]) for row in source).items()
                )
            ),
        }

    observed = [row for row in records if row["observation_available"]]
    missing = [row for row in records if not row["observation_available"]]
    summary = {
        "source_fixture_count": len(records),
        "source_record_count": len(observed),
        "missing_control_count": len(missing),
        "selected_source_reconstruction_rate": len(frames) / len(selected_ids),
        "capability_contract_count": catalog["contract_count"],
        "contract_truth_kind_counts": dict(
            sorted(
                Counter(
                    row["truth_kinds"][0]
                    for row in catalog["contracts"]
                    if len(row["truth_kinds"]) == 1
                ).items()
            )
        ),
        "cross_truth_kind_contract_collision_count": sum(
            row["mixed_truth_kind"] for row in catalog["contracts"]
        ),
        "target_contract_retention_rate": sum(
            row["target_contract_retained"] for row in observed
        )
        / len(observed),
        "missing_control_insufficient_rate": sum(
            row["identifiability_status"] == "INSUFFICIENT" for row in missing
        )
        / len(missing),
        "invalid_source_record_count": sum(
            row["identifiability_status"] == "INVALID_SOURCE_RECORD"
            for row in observed
        ),
        "role_identifier_overlap": len(
            {row["record_id"] for row in public["development"]}
            & {row["record_id"] for row in public["protected"]}
        ),
        "role_source_identifier_overlap": len(
            {row["source_candidate_id"] for row in records if row["role"] == "development" and row["source_candidate_id"]}
            & {row["source_candidate_id"] for row in records if row["role"] == "protected" and row["source_candidate_id"]}
        ),
        "roles": role_summary,
        "persisted_utterance_or_dialogue_text_count": 0,
        "persisted_slot_value_or_span_count": 0,
        "manual_language_inspection_count": 0,
        "model_load_count": 0,
        "model_generation_count": 0,
        "API_call_count": 0,
        "training_run_count": 0,
        "ontology_registration_count": 0,
        "trusted_state_mutation_count": 0,
        "service_or_sensor_call_count": 0,
        "external_side_effect_count": 0,
        "actual_execution_count": 0,
    }
    return {
        "contract_catalog": catalog,
        "hidden_records": records,
        "public_development": sorted(public["development"], key=lambda row: row["record_id"]),
        "public_protected": sorted(public["protected"], key=lambda row: row["record_id"]),
        "summary": summary,
    }


def audit_population(population: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    summary = population["summary"]
    gates = config["populationGates"]
    roles = summary["roles"]
    checks = {
        "source_population_counts_exact": bool(
            summary["source_fixture_count"] == gates["requiredSourceFixtureCount"]
            and summary["source_record_count"] == gates["requiredSourceRecordCount"]
            and summary["missing_control_count"] == gates["requiredMissingControlCount"]
        ),
        "role_split_is_exact_balanced_and_disjoint": bool(
            len(roles) == gates["requiredRoleCount"]
            and all(
                row["fixture_count"] == gates["requiredFixtureCountPerRole"]
                and row["source_record_count"] == gates["requiredSourceRecordCountPerRole"]
                and row["missing_control_count"] == gates["requiredMissingControlCountPerRole"]
                and row["records_per_cell_values"]
                == [gates["requiredRecordsPerTruthCandidateCellPerRole"]]
                for row in roles.values()
            )
            and summary["role_identifier_overlap"] == gates["requiredRoleIdentifierOverlap"]
            and summary["role_source_identifier_overlap"] == 0
        ),
        "source_reconstruction_target_retention_and_missing_controls_are_exact": bool(
            summary["selected_source_reconstruction_rate"]
            == gates["requiredSelectedSourceReconstructionRate"]
            and summary["target_contract_retention_rate"]
            == gates["requiredTargetContractRetentionRate"]
            and summary["missing_control_insufficient_rate"]
            == gates["requiredMissingControlInsufficientRate"]
            and summary["invalid_source_record_count"]
            <= gates["maximumInvalidSourceRecordCount"]
        ),
        "capability_contracts_do_not_cross_truth_kinds": bool(
            summary["cross_truth_kind_contract_collision_count"]
            == gates["requiredCrossTruthKindContractCollisionCount"]
        ),
        "development_identifiable_population_is_meaningful": _role_gate(
            roles["development"], gates
        ),
        "protected_identifiable_population_is_meaningful": _role_gate(
            roles["protected"], gates
        ),
        "text_value_model_authority_and_execution_boundaries_hold": bool(
            summary["persisted_utterance_or_dialogue_text_count"]
            <= gates["maximumPersistedUtteranceOrDialogueTextCount"]
            and summary["persisted_slot_value_or_span_count"]
            <= gates["maximumPersistedSlotValueOrSpanCount"]
            and all(
                summary[key] == gates[gate]
                for key, gate in (
                    ("manual_language_inspection_count", "maximumManualLanguageInspectionCount"),
                    ("model_load_count", "maximumModelLoadCount"),
                    ("model_generation_count", "maximumModelGenerationCount"),
                    ("API_call_count", "maximumAPICallCount"),
                    ("training_run_count", "maximumTrainingRunCount"),
                    ("ontology_registration_count", "maximumOntologyRegistrationCount"),
                    ("trusted_state_mutation_count", "maximumTrustedStateMutationCount"),
                    ("service_or_sensor_call_count", "maximumServiceOrSensorCallCount"),
                    ("external_side_effect_count", "maximumExternalSideEffectCount"),
                    ("actual_execution_count", "maximumActualExecutionCount"),
                )
            )
        ),
    }
    return {"passed": all(checks.values()), "checks": checks, "summary": summary}


def _role_gate(role: dict[str, Any], gates: dict[str, Any]) -> bool:
    counts = role["identifiable_by_truth_kind"]
    return bool(
        role["identifiable_source_record_count"]
        >= gates["minimumIdentifiableSourceRecordCountPerRole"]
        and counts.get("KNOWN", 0)
        >= gates["minimumIdentifiableKnownRecordCountPerRole"]
        and counts.get("PROVISIONAL", 0)
        >= gates["minimumIdentifiableProvisionalRecordCountPerRole"]
        and counts.get("UNSUPPORTED", 0)
        >= gates["minimumIdentifiableUnsupportedRecordCountPerRole"]
        and role["invalid_source_record_count"]
        <= gates["maximumInvalidSourceRecordCount"]
    )


__all__ = ["audit_population", "build_population", "read_structured_source"]
