from __future__ import annotations

from collections import Counter
from copy import deepcopy
import hashlib
import json
from typing import Any


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def initial_state(index: int) -> dict[str, Any]:
    parity = index % 2
    return {
        "schema_version": 1,
        "environment": "local_reversible_sandbox",
        "devices": {
            "D1": {"mode": "active", "quota": 3 + parity, "owner_team": "red", "revision": 10 + index},
            "D2": {"mode": "active", "quota": 4 - parity, "owner_team": "blue", "revision": 20 + index},
            "D3": {"mode": "offline", "quota": 0, "owner_team": "green", "revision": 30 + index},
        },
    }


def invariant_errors(state: dict[str, Any], config: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    ontology = config["fixedOntology"]
    devices = state.get("devices", {})
    if set(devices) != set(ontology["entityIds"]):
        errors.append("entity_set")
        return errors
    allowed_keys = set(ontology["mutableFields"]) | set(ontology["systemManagedFields"])
    for entity, row in devices.items():
        if set(row) != allowed_keys:
            errors.append(f"field_set:{entity}")
            continue
        mode, quota, revision = row["mode"], row["quota"], row["revision"]
        if type(quota) is not int or not 0 <= quota <= 8:
            errors.append(f"quota:{entity}")
        if type(revision) is not int or revision < 0:
            errors.append(f"revision:{entity}")
        if mode == "offline" and quota != 0:
            errors.append(f"offline_quota:{entity}")
        if mode == "maintenance" and quota > 2:
            errors.append(f"maintenance_quota:{entity}")
        if mode == "active" and quota < 1:
            errors.append(f"active_quota:{entity}")
    if sum(row["quota"] for row in devices.values() if type(row.get("quota")) is int) > 12:
        errors.append("total_quota")
    if not any(row.get("mode") == "active" for row in devices.values()):
        errors.append("active_required")
    if state.get("schema_version") != 1 or state.get("environment") != "local_reversible_sandbox":
        errors.append("immutable_root")
    return sorted(errors)


def proposal_for(scenario: str, index: int, state: dict[str, Any]) -> dict[str, Any]:
    revisions = {entity: row["revision"] for entity, row in state["devices"].items()}
    proposal = {
        "transaction_id": f"TX-{scenario}-{index:02d}",
        "expected_revisions": {"D1": revisions["D1"]},
        "operations": [{"entity_id": "D1", "field": "quota", "value": state["devices"]["D1"]["quota"] + 1}],
        "request_explicit_rollback": False,
        "fault": None,
    }
    if scenario == "valid_explicit_rollback":
        proposal["operations"] = [{"entity_id": "D2", "field": "owner_team", "value": "green"}]
        proposal["expected_revisions"] = {"D2": revisions["D2"]}
        proposal["request_explicit_rollback"] = True
    elif scenario == "valid_multi_entity_atomic":
        proposal["operations"] = [
            {"entity_id": "D1", "field": "mode", "value": "maintenance"},
            {"entity_id": "D1", "field": "quota", "value": 2},
            {"entity_id": "D3", "field": "mode", "value": "active"},
            {"entity_id": "D3", "field": "quota", "value": 1},
        ]
        proposal["expected_revisions"] = {"D1": revisions["D1"], "D3": revisions["D3"]}
    elif scenario == "invariant_violation":
        proposal["operations"] = [{"entity_id": "D3", "field": "quota", "value": 3}]
        proposal["expected_revisions"] = {"D3": revisions["D3"]}
    elif scenario == "unauthorized_field":
        proposal["operations"] = [{"entity_id": "D1", "field": "firmware_version", "value": "9.9"}]
    elif scenario == "stale_revision":
        proposal["expected_revisions"] = {"D1": revisions["D1"] - 1}
    elif scenario == "malformed_type":
        proposal["operations"] = [{"entity_id": "D1", "field": "quota", "value": "high"}]
    elif scenario == "contradictory_duplicate_patch":
        proposal["operations"] = [
            {"entity_id": "D1", "field": "quota", "value": 4},
            {"entity_id": "D1", "field": "quota", "value": 5},
        ]
    elif scenario == "unknown_entity":
        proposal["operations"] = [{"entity_id": "D9", "field": "quota", "value": 1}]
        proposal["expected_revisions"] = {"D9": 0}
    elif scenario == "preview_token_tamper":
        proposal["fault"] = "preview_token_tamper"
    elif scenario == "post_commit_corruption":
        proposal["fault"] = "post_commit_corruption"
    return proposal


def validate_proposal(state: dict[str, Any], proposal: dict[str, Any], config: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    ontology = config["fixedOntology"]
    mutable = ontology["mutableFields"]
    seen: set[tuple[str, str]] = set()
    touched: set[str] = set()
    for operation in proposal.get("operations", []):
        entity = operation.get("entity_id")
        field = operation.get("field")
        key = (entity, field)
        if key in seen:
            errors.append("duplicate_entity_field")
        seen.add(key)
        touched.add(entity)
        if entity not in state["devices"]:
            errors.append("unknown_entity")
            continue
        if field not in mutable:
            errors.append("unauthorized_field")
            continue
        spec = mutable[field]
        value = operation.get("value")
        if spec["type"] == "integer" and (type(value) is not int or not spec["minimum"] <= value <= spec["maximum"]):
            errors.append("malformed_or_out_of_range_integer")
        if spec["type"] == "enum" and value not in spec["values"]:
            errors.append("malformed_enum")
    expected = proposal.get("expected_revisions", {})
    if set(expected) != touched:
        errors.append("expected_revision_scope")
    for entity in touched & set(state["devices"]):
        if expected.get(entity) != state["devices"][entity]["revision"]:
            errors.append("stale_revision")
    if not proposal.get("operations"):
        errors.append("empty_patch")
    return sorted(set(errors))


def apply_operations(state: dict[str, Any], operations: list[dict[str, Any]]) -> dict[str, Any]:
    after = deepcopy(state)
    touched = set()
    for operation in operations:
        entity = operation["entity_id"]
        after["devices"][entity][operation["field"]] = deepcopy(operation["value"])
        touched.add(entity)
    for entity in touched:
        after["devices"][entity]["revision"] += 1
    return after


def state_diff(before: dict[str, Any], after: dict[str, Any]) -> list[dict[str, Any]]:
    changes = []
    for entity in sorted(before["devices"]):
        for field in sorted(before["devices"][entity]):
            if before["devices"][entity][field] != after["devices"][entity][field]:
                changes.append({"entity_id": entity, "field": field, "before": before["devices"][entity][field], "after": after["devices"][entity][field]})
    return changes


class SandboxStore:
    def __init__(self, state: dict[str, Any], config: dict[str, Any]) -> None:
        self.state = deepcopy(state)
        self.config = config
        self.log: list[dict[str, Any]] = []
        self.snapshots: dict[str, dict[str, Any]] = {}
        self.invariant_history = [not invariant_errors(self.state, config)]

    def _log(self, event: str, details: dict[str, Any]) -> None:
        entry = {
            "sequence": len(self.log),
            "previous_hash": self.log[-1]["entry_hash"] if self.log else "GENESIS",
            "event": event,
            "details": deepcopy(details),
        }
        entry["entry_hash"] = canonical_hash(entry)
        self.log.append(entry)

    def preview(self, proposal: dict[str, Any]) -> dict[str, Any]:
        base = deepcopy(self.state)
        errors = validate_proposal(base, proposal, self.config)
        after = None if errors else apply_operations(base, proposal["operations"])
        invariant = [] if after is None else invariant_errors(after, self.config)
        accepted = not errors and not invariant
        preview = {
            "transaction_id": proposal["transaction_id"],
            "accepted": accepted,
            "errors": errors,
            "invariant_errors": invariant,
            "base_state_hash": canonical_hash(base),
            "patch_hash": canonical_hash(proposal["operations"]),
            "expected_post_state_hash": canonical_hash(after) if after is not None else None,
            "expected_post_state": after if accepted else None,
            "diff": state_diff(base, after) if accepted else [],
        }
        preview["preview_token"] = canonical_hash({key: preview[key] for key in ("transaction_id", "base_state_hash", "patch_hash", "expected_post_state_hash")}) if accepted else None
        self._log("preview", {"transaction_id": proposal["transaction_id"], "accepted": accepted, "errors": errors + invariant})
        self.invariant_history.append(not invariant_errors(self.state, self.config))
        return preview

    def commit(self, proposal: dict[str, Any], supplied_preview: dict[str, Any]) -> dict[str, Any]:
        before = deepcopy(self.state)
        if supplied_preview.get("base_state_hash") != canonical_hash(before):
            self._log("commit_rejected", {"transaction_id": proposal["transaction_id"], "reason": "stale_preview"})
            return {"committed": False, "reason": "stale_preview", "preview_commit_parity": False}
        expected = self.preview(proposal)
        if not expected["accepted"]:
            self._log("commit_rejected", {"transaction_id": proposal["transaction_id"], "reason": "invalid_preview"})
            return {"committed": False, "reason": "invalid_preview", "preview_commit_parity": False}
        if supplied_preview != expected:
            self._log("commit_rejected", {"transaction_id": proposal["transaction_id"], "reason": "preview_token_or_payload_mismatch"})
            return {"committed": False, "reason": "preview_token_or_payload_mismatch", "preview_commit_parity": False}
        self.snapshots[proposal["transaction_id"]] = before
        self.state = apply_operations(before, proposal["operations"])
        parity = self.state == supplied_preview["expected_post_state"]
        self._log("commit", {"transaction_id": proposal["transaction_id"], "before_hash": canonical_hash(before), "after_hash": canonical_hash(self.state), "parity": parity})
        self.invariant_history.append(not invariant_errors(self.state, self.config))
        return {"committed": True, "reason": None, "preview_commit_parity": parity}

    def inject_post_commit_corruption(self, transaction_id: str) -> None:
        self.state["devices"]["D3"]["owner_team"] = "red"
        self._log("injected_fault", {"transaction_id": transaction_id, "kind": "post_commit_corruption"})
        self.invariant_history.append(not invariant_errors(self.state, self.config))

    def rollback(self, transaction_id: str, reason: str) -> bool:
        if transaction_id not in self.snapshots:
            return False
        target = deepcopy(self.snapshots[transaction_id])
        self.state = target
        self._log("rollback", {"transaction_id": transaction_id, "reason": reason, "restored_hash": canonical_hash(target)})
        self.invariant_history.append(not invariant_errors(self.state, self.config))
        return True

    def verify_or_rollback(self, proposal: dict[str, Any], preview: dict[str, Any]) -> dict[str, Any]:
        matches = canonical_hash(self.state) == preview["expected_post_state_hash"]
        self._log("verification", {"transaction_id": proposal["transaction_id"], "matches_preview": matches})
        if matches:
            return {"verified": True, "rollback_recovered": None}
        recovered = self.rollback(proposal["transaction_id"], "independent_verification_failure")
        return {"verified": False, "rollback_recovered": recovered}

    def provenance_valid(self) -> bool:
        previous = "GENESIS"
        for index, entry in enumerate(self.log):
            payload = {key: value for key, value in entry.items() if key != "entry_hash"}
            if entry["sequence"] != index or entry["previous_hash"] != previous or canonical_hash(payload) != entry["entry_hash"]:
                return False
            previous = entry["entry_hash"]
        return True


def oracle_final_state(initial: dict[str, Any], proposal: dict[str, Any], scenario: str) -> dict[str, Any]:
    if scenario in {"valid_retain", "valid_multi_entity_atomic"}:
        return apply_operations(initial, proposal["operations"])
    return deepcopy(initial)


def expected_disposition(scenario: str) -> str:
    if scenario in {"valid_retain", "valid_multi_entity_atomic"}:
        return "retained"
    if scenario == "valid_explicit_rollback":
        return "rolled_back"
    if scenario == "post_commit_corruption":
        return "rolled_back_after_verification_failure"
    return "rejected"


def build_fixtures(config: dict[str, Any]) -> list[dict[str, Any]]:
    fixtures = []
    for scenario in config["population"]["scenarios"]:
        for index in range(config["population"]["recordsPerScenario"]):
            before = initial_state(index)
            proposal = proposal_for(scenario, index, before)
            fixtures.append({
                "record_id": f"v168-{canonical_hash([scenario, index])[:16]}",
                "split": "development_only",
                "scenario": scenario,
                "initial_state": before,
                "proposal": proposal,
                "expected_disposition": expected_disposition(scenario),
                "expected_final_state": oracle_final_state(before, proposal, scenario),
            })
    return sorted(fixtures, key=lambda row: row["record_id"])


def _authorized_commit_paths(proposal: dict[str, Any]) -> set[tuple[str, str]]:
    paths = {(row["entity_id"], row["field"]) for row in proposal["operations"]}
    paths |= {(row["entity_id"], "revision") for row in proposal["operations"]}
    return paths


def run_fixture(fixture: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    initial = deepcopy(fixture["initial_state"])
    proposal = deepcopy(fixture["proposal"])
    store = SandboxStore(initial, config)
    before_preview_hash = canonical_hash(store.state)
    preview = store.preview(proposal)
    preview_nonmutation = canonical_hash(store.state) == before_preview_hash
    supplied = deepcopy(preview)
    if proposal["fault"] == "preview_token_tamper" and supplied["accepted"]:
        supplied["preview_token"] = "tampered"
    commit = store.commit(proposal, supplied) if preview["accepted"] else {"committed": False, "reason": "preview_rejected", "preview_commit_parity": False}
    commit_state = deepcopy(store.state)
    unauthorized_commit_paths: list[tuple[str, str]] = []
    if commit["committed"]:
        unauthorized_commit_paths = [
            (row["entity_id"], row["field"])
            for row in state_diff(initial, commit_state)
            if (row["entity_id"], row["field"]) not in _authorized_commit_paths(proposal)
        ]
    fault_detected = proposal["fault"] is None
    verification = None
    disposition = "rejected"
    if commit["committed"]:
        if proposal["fault"] == "post_commit_corruption":
            store.inject_post_commit_corruption(proposal["transaction_id"])
        verification = store.verify_or_rollback(proposal, preview)
        fault_detected = proposal["fault"] != "post_commit_corruption" or not verification["verified"]
        if not verification["verified"]:
            disposition = "rolled_back_after_verification_failure"
        elif proposal["request_explicit_rollback"]:
            recovered = store.rollback(proposal["transaction_id"], "explicit_request")
            disposition = "rolled_back" if recovered else "rollback_failed"
        else:
            disposition = "retained"
    elif proposal["fault"] == "preview_token_tamper":
        fault_detected = commit["reason"] == "preview_token_or_payload_mismatch"

    scenario = fixture["scenario"]
    rollback_required = scenario in {"valid_explicit_rollback", "post_commit_corruption"}
    rollback_recovered = store.state == initial if rollback_required else True
    rejected = disposition == "rejected"
    result = {
        "record_id": fixture["record_id"],
        "scenario": scenario,
        "disposition": disposition,
        "expected_disposition": fixture["expected_disposition"],
        "expected_disposition_exact": disposition == fixture["expected_disposition"],
        "exact_final_target_state": store.state == fixture["expected_final_state"],
        "rejected_state_unchanged": not rejected or store.state == initial,
        "preview_nonmutation": preview_nonmutation,
        "preview_accepted": preview["accepted"],
        "committed": commit["committed"],
        "preview_commit_parity": not commit["committed"] or commit["preview_commit_parity"],
        "atomic_multi_entity_commit": scenario != "valid_multi_entity_atomic" or (commit["committed"] and commit["preview_commit_parity"]),
        "rollback_recovered": rollback_recovered,
        "invariants_preserved": all(store.invariant_history) and not invariant_errors(store.state, config),
        "zero_unauthorized_commit_mutation": not unauthorized_commit_paths,
        "fault_detected": fault_detected,
        "provenance_chain_valid": store.provenance_valid(),
        "final_state_hash": canonical_hash(store.state),
        "expected_final_state_hash": canonical_hash(fixture["expected_final_state"]),
        "commit_reason": commit["reason"],
        "log_event_types": [entry["event"] for entry in store.log],
    }
    return result


def evaluate_census(fixtures: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    results = [run_fixture(fixture, config) for fixture in fixtures]
    by_scenario = Counter(row["scenario"] for row in results)
    rate = lambda key, rows=results: sum(bool(row[key]) for row in rows) / len(rows)
    committed = [row for row in results if row["committed"]]
    rejected = [row for row in results if row["disposition"] == "rejected"]
    explicit = [row for row in results if row["scenario"] == "valid_explicit_rollback"]
    corruption = [row for row in results if row["scenario"] == "post_commit_corruption"]
    multi = [row for row in results if row["scenario"] == "valid_multi_entity_atomic"]
    faults = [row for row in results if row["scenario"] in {"preview_token_tamper", "post_commit_corruption"}]
    summary = {
        "record_count": len(results),
        "scenario_count": len(by_scenario),
        "scenario_counts": dict(sorted(by_scenario.items())),
        "expected_disposition_accuracy": rate("expected_disposition_exact"),
        "exact_final_target_state": rate("exact_final_target_state"),
        "rejected_state_immutability": rate("rejected_state_unchanged", rejected),
        "preview_nonmutation": rate("preview_nonmutation"),
        "preview_commit_parity": rate("preview_commit_parity", committed),
        "atomic_multi_entity_commit": rate("atomic_multi_entity_commit", multi),
        "explicit_rollback_recovery": rate("rollback_recovered", explicit),
        "verification_failure_rollback_recovery": rate("rollback_recovered", corruption),
        "invariant_preservation": rate("invariants_preserved"),
        "zero_unauthorized_commit_mutation": rate("zero_unauthorized_commit_mutation"),
        "fault_detection": rate("fault_detected", faults),
        "provenance_chain_validity": rate("provenance_chain_valid"),
        "commit_count": len(committed),
        "rejection_count": len(rejected),
        "explicit_rollback_count": len(explicit),
        "automatic_rollback_count": len(corruption),
    }
    return {"results": results, "summary": summary}


def evaluate_gates(evaluation: dict[str, Any], access: dict[str, int], config: dict[str, Any]) -> dict[str, bool]:
    summary, gates = evaluation["summary"], config["sandboxGates"]
    return {
        "record_count": summary["record_count"] == gates["requiredRecordCount"],
        "scenario_count": summary["scenario_count"] == gates["requiredScenarioCount"],
        "records_per_scenario": set(summary["scenario_counts"].values()) == {gates["requiredRecordsPerScenario"]},
        "expected_disposition": summary["expected_disposition_accuracy"] == gates["requiredExpectedDispositionAccuracy"],
        "final_target_state": summary["exact_final_target_state"] == gates["requiredExactFinalTargetState"],
        "rejected_state_immutability": summary["rejected_state_immutability"] == gates["requiredRejectedStateImmutability"],
        "preview_nonmutation": summary["preview_nonmutation"] == gates["requiredPreviewNonmutation"],
        "preview_commit_parity": summary["preview_commit_parity"] == gates["requiredPreviewCommitParity"],
        "atomic_multi_entity_commit": summary["atomic_multi_entity_commit"] == gates["requiredAtomicMultiEntityCommit"],
        "explicit_rollback_recovery": summary["explicit_rollback_recovery"] == gates["requiredExplicitRollbackRecovery"],
        "verification_failure_rollback_recovery": summary["verification_failure_rollback_recovery"] == gates["requiredVerificationFailureRollbackRecovery"],
        "invariant_preservation": summary["invariant_preservation"] == gates["requiredInvariantPreservation"],
        "zero_unauthorized_commit_mutation": summary["zero_unauthorized_commit_mutation"] == gates["requiredZeroUnauthorizedCommitMutation"],
        "fault_detection": summary["fault_detection"] == gates["requiredFaultDetection"],
        "provenance_chain_validity": summary["provenance_chain_validity"] == gates["requiredProvenanceChainValidity"],
        "zero_disallowed_access": all(access[key] <= gates[maximum] for key, maximum in {
            "evaluation_record_count": "maximumEvaluationRecordCount", "manual_judgment_count": "maximumManualJudgmentCount",
            "model_load_count": "maximumModelLoadCount", "model_generation_count": "maximumModelGenerationCount",
            "API_call_count": "maximumAPICallCount", "training_run_count": "maximumTrainingRunCount",
            "provisional_ontology_use_count": "maximumProvisionalOntologyUseCount", "real_service_call_count": "maximumRealServiceCallCount",
            "external_side_effect_count": "maximumExternalSideEffectCount", "real_execution_count": "maximumRealExecutionCount",
        }.items()),
    }


__all__ = [
    "SandboxStore", "apply_operations", "build_fixtures", "canonical_hash", "evaluate_census",
    "evaluate_gates", "expected_disposition", "initial_state", "invariant_errors", "proposal_for",
    "run_fixture", "validate_proposal",
]
