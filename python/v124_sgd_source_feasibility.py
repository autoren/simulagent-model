from __future__ import annotations

from collections import Counter, defaultdict
from io import BytesIO
import hashlib
import json
import re
import tarfile
from typing import Any

from v93_open_set_source import canonical_sha256


DOMAIN_PATTERN = re.compile(r"^(.*)_\d+$")


def service_domain(service: str) -> str:
    match = DOMAIN_PATTERN.match(service)
    if not match:
        raise ValueError(f"service lacks numeric suffix: {service}")
    return match.group(1).lower()


def _json_member(archive: tarfile.TarFile, member: tarfile.TarInfo) -> Any:
    handle = archive.extractfile(member)
    if handle is None:
        raise ValueError(f"unreadable archive member: {member.name}")
    return json.loads(handle.read())


def build_inventory(archive_bytes: bytes, config: dict[str, Any]) -> dict[str, Any]:
    root_expected = f"dstc8-schema-guided-dialogue-{config['revision']}"
    partitions = set(config["candidateDefinition"]["partitions"])
    with tarfile.open(fileobj=BytesIO(archive_bytes), mode="r:gz") as archive:
        files = [member for member in archive.getmembers() if member.isfile()]
        if any(member.name.startswith("/") or ".." in member.name.split("/") for member in files):
            raise ValueError("unsafe archive member")
        roots = {member.name.split("/", 1)[0] for member in files}
        if roots != {root_expected}:
            raise ValueError("archive revision root mismatch")
        license_members = [member for member in files if member.name == f"{root_expected}/LICENSE.txt"]
        if len(license_members) != 1:
            raise ValueError("expected one license")
        license_handle = archive.extractfile(license_members[0])
        if license_handle is None:
            raise ValueError("license unreadable")
        license_text = license_handle.read().decode("utf-8")

        schema_members: dict[str, tarfile.TarInfo] = {}
        dialogue_members: dict[str, list[tarfile.TarInfo]] = defaultdict(list)
        for member in files:
            relative = member.name[len(root_expected) + 1 :]
            pieces = relative.split("/")
            if len(pieces) == 2 and pieces[0] in partitions and pieces[1] == "schema.json":
                schema_members[pieces[0]] = member
            if len(pieces) == 2 and pieces[0] in partitions and re.fullmatch(r"dialogues_\d+\.json", pieces[1]):
                dialogue_members[pieces[0]].append(member)
        if set(schema_members) != partitions or set(dialogue_members) != partitions:
            raise ValueError("missing SGD partition schemas or dialogues")

        schemas: dict[str, dict[str, set[str]]] = {}
        schema_counts: dict[str, Any] = {}
        all_domains: set[str] = set()
        all_services: set[str] = set()
        all_intents: set[str] = set()
        for partition in sorted(partitions):
            values = _json_member(archive, schema_members[partition])
            service_map: dict[str, set[str]] = {}
            for service in values:
                name = service["service_name"]
                intents = {intent["name"] for intent in service["intents"]}
                if not name or not intents or name in service_map:
                    raise ValueError("invalid or duplicate service schema")
                service_map[name] = intents
                all_domains.add(service_domain(name))
                all_services.add(name)
                all_intents.update(f"{name}::{intent}" for intent in intents)
            schemas[partition] = service_map
            schema_counts[partition] = {
                "domain_count": len({service_domain(name) for name in service_map}),
                "service_count": len(service_map),
                "intent_count": sum(len(intents) for intents in service_map.values()),
            }

        train_pairs = {(service, intent) for service, intents in schemas["train"].items() for intent in intents}
        train_domains = {service_domain(service) for service in schemas["train"]}
        candidates: list[dict[str, Any]] = []
        dialogue_counts: Counter[str] = Counter()
        candidate_counts: Counter[str] = Counter()
        class_counts: Counter[str] = Counter()
        class_domains: dict[str, set[str]] = defaultdict(set)
        for partition in sorted(partitions):
            for member in sorted(dialogue_members[partition], key=lambda item: item.name):
                dialogues = _json_member(archive, member)
                for dialogue in dialogues:
                    dialogue_id = str(dialogue["dialogue_id"])
                    dialogue_counts[partition] += 1
                    for turn_index, turn in enumerate(dialogue["turns"]):
                        if turn.get("speaker") != "USER" or not isinstance(turn.get("utterance"), str) or not turn["utterance"].strip():
                            continue
                        matches: list[tuple[str, str]] = []
                        for frame in turn.get("frames", []):
                            service = frame.get("service")
                            state = frame.get("state", {})
                            active = state.get("active_intent")
                            intent_actions = [
                                action for action in frame.get("actions", [])
                                if action.get("act") == "INFORM_INTENT" and action.get("slot") == "intent"
                            ]
                            action_values = [value for action in intent_actions for value in action.get("values", [])]
                            if len(intent_actions) == 1 and len(action_values) == 1 and active == action_values[0] and active != "NONE":
                                matches.append((service, active))
                        if len(matches) != 1:
                            continue
                        service, intent = matches[0]
                        if service not in schemas[partition] or intent not in schemas[partition][service]:
                            raise ValueError("candidate intent absent from partition schema")
                        domain = service_domain(service)
                        class_label = None
                        if partition == "test":
                            if (service, intent) in train_pairs:
                                class_label = "known"
                            elif domain in train_domains:
                                class_label = "novel_valid"
                            else:
                                class_label = "unsupported"
                            class_counts[class_label] += 1
                            class_domains[class_label].add(domain)
                        identifier = f"sgd::{partition}::{dialogue_id}::{turn_index:03d}::{service}::{intent}"
                        candidates.append({
                            "candidate_id": identifier,
                            "partition": partition,
                            "domain": domain,
                            "service": service,
                            "intent": intent,
                            "class_label": class_label,
                        })
                        candidate_counts[partition] += 1

    identifiers = [row["candidate_id"] for row in candidates]
    forbidden = {"utterance", "text", "slot_values", "values", "tokens", "dialogue"}
    keys = set().union(*(row.keys() for row in candidates)) if candidates else set()
    if keys & forbidden:
        raise AssertionError("language leaked into SGD inventory")
    source_gates = config["sourceGates"]
    checks = {
        "dialogue_count": sum(dialogue_counts.values()) >= source_gates["minimumDialogueCount"],
        "domain_count": len(all_domains) >= source_gates["minimumDomainCount"],
        "service_count": len(all_services) >= source_gates["minimumServiceCount"],
        "intent_count": len(all_intents) >= source_gates["minimumIntentCount"],
        "candidate_count": len(candidates) >= source_gates["minimumIntentIntroductionCandidateCount"],
        "train_candidate_count": candidate_counts["train"] >= source_gates["minimumTrainCandidateCount"],
        "test_class_counts": all(class_counts[label] >= source_gates["minimumTestCandidateCountPerOpenSetClass"] for label in ("known", "novel_valid", "unsupported")),
        "known_test_domain_coverage": len(class_domains["known"]) >= source_gates["minimumKnownTestDomainCoverage"],
        "novel_valid_test_domain_coverage": len(class_domains["novel_valid"]) >= source_gates["minimumNovelValidTestDomainCoverage"],
        "unsupported_test_domain_coverage": len(class_domains["unsupported"]) >= source_gates["minimumUnsupportedTestDomainCoverage"],
        "partitions_exact": set(schema_counts) == set(source_gates["requiredPartitions"]),
        "license_exact_family": source_gates["requiredLicenseSubstring"] in license_text,
        "schema_intent_membership": source_gates["requireSchemaIntentMembership"],
        "unique_candidate_identifiers": source_gates["requireUniqueCandidateIdentifiers"] and len(identifiers) == len(set(identifiers)),
        "text_free_inventory": not bool(keys & forbidden),
        "zero_manual_language": source_gates["maximumManualLanguageInspectionCount"] == 0,
        "zero_model": source_gates["maximumModelLoadCount"] == 0 and source_gates["maximumModelGenerationCount"] == 0,
        "zero_execution": source_gates["maximumActualExecutionCount"] == 0,
    }
    passed = all(checks.values())
    candidates.sort(key=lambda row: row["candidate_id"])
    return {
        "archive_sha256": hashlib.sha256(archive_bytes).hexdigest(),
        "archive_root": root_expected,
        "license_sha256": hashlib.sha256(license_text.encode()).hexdigest(),
        "schema_counts": schema_counts,
        "dialogue_counts": dict(sorted(dialogue_counts.items())),
        "dialogue_count": sum(dialogue_counts.values()),
        "domain_count": len(all_domains),
        "service_count": len(all_services),
        "intent_count": len(all_intents),
        "candidate_counts": dict(sorted(candidate_counts.items())),
        "candidate_count": len(candidates),
        "test_open_set_class_counts": dict(sorted(class_counts.items())),
        "test_open_set_domain_coverage": {key: len(value) for key, value in sorted(class_domains.items())},
        "candidate_index_sha256": canonical_sha256(candidates),
        "candidate_index": candidates,
        "contains_language_or_slot_values": False,
        "source_gates": checks,
        "source_pass": passed,
        "decision": config["decisionRule"]["ifEverySourceAndAccessGatePasses"] if passed else config["decisionRule"]["otherwise"],
        "emitted_language_record_count": 0,
        "manual_language_inspection_count": 0,
        "model_load_count": 0,
        "model_generation_count": 0,
        "actual_execution_count": 0,
    }


__all__ = ["build_inventory", "service_domain"]
