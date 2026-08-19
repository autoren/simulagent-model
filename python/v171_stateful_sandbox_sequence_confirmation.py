from __future__ import annotations

from collections import Counter
from copy import deepcopy
import json
from typing import Any

from v168_fixed_ontology_reversible_sandbox import (
    SandboxStore,
    apply_operations,
    canonical_hash,
    initial_state,
    invariant_errors,
    state_diff,
)


class ProvenanceError(RuntimeError):
    pass


def compose_config(design: dict[str, Any], v168_config: dict[str, Any]) -> dict[str, Any]:
    combined = deepcopy(design)
    for key in ("fixedOntology", "invariants", "transactionProtocol"):
        combined[key] = deepcopy(v168_config[key])
    return combined


def proposal(
    transaction_id: str,
    state: dict[str, Any],
    operations: list[dict[str, Any]],
) -> dict[str, Any]:
    touched = sorted({row["entity_id"] for row in operations})
    return {
        "transaction_id": transaction_id,
        "expected_revisions": {
            entity: state["devices"][entity]["revision"] for entity in touched
        },
        "operations": deepcopy(operations),
        "request_explicit_rollback": False,
        "fault": None,
    }


def owner_operation(state: dict[str, Any], entity: str, offset: int = 1) -> dict[str, Any]:
    values = ["red", "blue", "green"]
    current = state["devices"][entity]["owner_team"]
    return {
        "entity_id": entity,
        "field": "owner_team",
        "value": values[(values.index(current) + offset) % len(values)],
    }


def atomic_operations(state: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        owner_operation(state, "D1", 1),
        owner_operation(state, "D2", 2),
        {"entity_id": "D3", "field": "mode", "value": "active"},
        {"entity_id": "D3", "field": "quota", "value": 1},
    ]


def authorized_paths(operations: list[dict[str, Any]]) -> set[tuple[str, str]]:
    paths = {(row["entity_id"], row["field"]) for row in operations}
    return paths | {(row["entity_id"], "revision") for row in operations}


class DurableSandboxHarness:
    """Simulation-only durability adapter around the unchanged V168 transaction engine."""

    def __init__(self, state: dict[str, Any], config: dict[str, Any]) -> None:
        self.config = config
        self.store = SandboxStore(state, config)
        self.lifecycle: dict[str, dict[str, Any]] = {}
        self.pending: dict[str, Any] | None = None
        self.visible_boundaries: list[dict[str, Any]] = [deepcopy(state)]
        self.retained_authorization_checks: list[bool] = []

    def _boundary(self) -> None:
        self.visible_boundaries.append(deepcopy(self.store.state))

    def _mark_lifecycle(self, transaction_id: str, status: str, **extra: Any) -> None:
        self.lifecycle[transaction_id] = {"status": status, **deepcopy(extra)}

    def execute(self, item: dict[str, Any], explicit_rollback: bool = False) -> dict[str, Any]:
        transaction_id = item["transaction_id"]
        if transaction_id in self.lifecycle:
            self.store._log(
                "terminal_replay_rejected",
                {"transaction_id": transaction_id, "status": self.lifecycle[transaction_id]["status"]},
            )
            self._boundary()
            return {"committed": False, "reason": "terminal_replay", "replayed": True}

        before = deepcopy(self.store.state)
        preview = self.store.preview(item)
        if not preview["accepted"]:
            self._mark_lifecycle(transaction_id, "rejected", reason="preview_rejected")
            self._boundary()
            return {"committed": False, "reason": "preview_rejected", "replayed": False}
        commit = self.store.commit(item, preview)
        if not commit["committed"]:
            self._mark_lifecycle(transaction_id, "rejected", reason=commit["reason"])
            self._boundary()
            return {**commit, "replayed": False}
        verification = self.store.verify_or_rollback(item, preview)
        if not verification["verified"]:
            self._mark_lifecycle(transaction_id, "rolled_back", reason="verification_failure")
            self._boundary()
            return {"committed": True, "verified": False, "rolled_back": True, "replayed": False}

        changed = {
            (row["entity_id"], row["field"])
            for row in state_diff(before, self.store.state)
        }
        self.retained_authorization_checks.append(changed <= authorized_paths(item["operations"]))
        self._mark_lifecycle(
            transaction_id,
            "retained",
            before_state=before,
            expected_post_state=deepcopy(preview["expected_post_state"]),
            expected_post_state_hash=preview["expected_post_state_hash"],
        )
        if explicit_rollback:
            rollback = self.request_rollback(transaction_id)
            return {
                "committed": True,
                "verified": True,
                "rolled_back": rollback["rolled_back"],
                "replayed": False,
            }
        self._boundary()
        return {"committed": True, "verified": True, "rolled_back": False, "replayed": False}

    def request_rollback(self, transaction_id: str) -> dict[str, Any]:
        lifecycle = self.lifecycle.get(transaction_id)
        if lifecycle is None:
            self.store._log("rollback_rejected", {"transaction_id": transaction_id, "reason": "unknown_transaction"})
            self._boundary()
            return {"rolled_back": False, "reason": "unknown_transaction"}
        if lifecycle["status"] in {"rolled_back", "recovered_rollback"}:
            self.store._log("rollback_idempotent", {"transaction_id": transaction_id, "status": lifecycle["status"]})
            self._boundary()
            return {"rolled_back": False, "reason": "already_rolled_back", "idempotent": True}
        if lifecycle["status"] != "retained":
            self.store._log("rollback_rejected", {"transaction_id": transaction_id, "reason": lifecycle["status"]})
            self._boundary()
            return {"rolled_back": False, "reason": "not_retained"}
        if canonical_hash(self.store.state) != lifecycle["expected_post_state_hash"]:
            self.store._log("rollback_rejected", {"transaction_id": transaction_id, "reason": "later_state_exists"})
            self._boundary()
            return {"rolled_back": False, "reason": "later_state_exists"}
        restored = self.store.rollback(transaction_id, "explicit_stateful_request")
        if restored:
            self._mark_lifecycle(transaction_id, "rolled_back")
        self._boundary()
        return {"rolled_back": restored, "reason": None if restored else "snapshot_missing"}

    def crash_image(
        self,
        item: dict[str, Any],
        crash_point: str,
        partial_write: bool = False,
    ) -> dict[str, Any]:
        if self.pending is not None:
            raise RuntimeError("only one pending transaction is supported")
        before = deepcopy(self.store.state)
        preview = self.store.preview(item)
        if not preview["accepted"]:
            raise RuntimeError("crash transaction must have an accepted preview")
        self.pending = {
            "transaction_id": item["transaction_id"],
            "before_state": before,
            "proposal": deepcopy(item),
            "preview": deepcopy(preview),
            "phase": "previewed",
            "partial_write": partial_write,
        }
        self.store._log("journal_prepared", {"transaction_id": item["transaction_id"]})
        if crash_point == "after_preview":
            return self.durable_image()

        self.store.snapshots[item["transaction_id"]] = deepcopy(before)
        self.pending["phase"] = "snapshotted"
        self.store._log("snapshot_durable", {"transaction_id": item["transaction_id"]})
        if crash_point == "after_snapshot_before_apply":
            return self.durable_image()

        applied_operations = item["operations"][:1] if partial_write else item["operations"]
        self.store.state = apply_operations(before, applied_operations)
        self.pending["phase"] = "applied"
        self.store._log(
            "durable_state_applied",
            {
                "transaction_id": item["transaction_id"],
                "partial_write": partial_write,
                "state_hash": canonical_hash(self.store.state),
            },
        )
        if crash_point == "after_apply_before_verify" or partial_write:
            return self.durable_image()

        matches = canonical_hash(self.store.state) == preview["expected_post_state_hash"]
        self.store._log("durable_verification", {"transaction_id": item["transaction_id"], "matches": matches})
        if not matches:
            raise RuntimeError("full durable apply must match its preview")
        self.pending["phase"] = "verified"
        if crash_point != "after_verify_before_finalize":
            raise ValueError(f"unknown crash point: {crash_point}")
        return self.durable_image()

    def durable_image(self) -> dict[str, Any]:
        return {
            "state": deepcopy(self.store.state),
            "log": deepcopy(self.store.log),
            "snapshots": deepcopy(self.store.snapshots),
            "lifecycle": deepcopy(self.lifecycle),
            "pending": deepcopy(self.pending),
            "visible_boundaries": deepcopy(self.visible_boundaries),
            "retained_authorization_checks": deepcopy(self.retained_authorization_checks),
        }

    @classmethod
    def restart(cls, image: dict[str, Any], config: dict[str, Any]) -> tuple["DurableSandboxHarness", dict[str, Any]]:
        candidate = cls(image["state"], config)
        candidate.store.log = deepcopy(image["log"])
        candidate.store.snapshots = deepcopy(image["snapshots"])
        candidate.lifecycle = deepcopy(image["lifecycle"])
        candidate.pending = deepcopy(image["pending"])
        candidate.visible_boundaries = deepcopy(image["visible_boundaries"])
        candidate.retained_authorization_checks = deepcopy(image["retained_authorization_checks"])
        if not candidate.store.provenance_valid():
            raise ProvenanceError("durable provenance chain is invalid")
        if candidate.pending is None:
            candidate.store._log("restart_verified", {"pending": False})
            candidate._boundary()
            return candidate, {"recovered": False, "action": "none"}

        pending = deepcopy(candidate.pending)
        transaction_id = pending["transaction_id"]
        preview = pending["preview"]
        if pending["phase"] == "verified" and canonical_hash(candidate.store.state) == preview["expected_post_state_hash"]:
            before = pending["before_state"]
            changed = {
                (row["entity_id"], row["field"])
                for row in state_diff(before, candidate.store.state)
            }
            candidate.retained_authorization_checks.append(
                changed <= authorized_paths(pending["proposal"]["operations"])
            )
            candidate._mark_lifecycle(
                transaction_id,
                "retained",
                before_state=before,
                expected_post_state=deepcopy(preview["expected_post_state"]),
                expected_post_state_hash=preview["expected_post_state_hash"],
            )
            candidate.store._log("recovery_finalize", {"transaction_id": transaction_id})
            action = "finalize_retained"
        else:
            candidate.store.state = deepcopy(pending["before_state"])
            candidate._mark_lifecycle(transaction_id, "recovered_rollback")
            candidate.store._log(
                "recovery_rollback",
                {"transaction_id": transaction_id, "from_phase": pending["phase"]},
            )
            action = "restore_before_state"
        candidate.pending = None
        candidate._boundary()
        return candidate, {"recovered": True, "action": action, "transaction_id": transaction_id}

    def provenance_valid(self) -> bool:
        return self.store.provenance_valid()

    def visible_invariants_hold(self) -> bool:
        return all(not invariant_errors(state, self.config) for state in self.visible_boundaries)

    def authorized_retained_mutations_hold(self) -> bool:
        return all(self.retained_authorization_checks)


def _sequence_id(scenario: str, variant: int) -> str:
    return f"v171-{canonical_hash([scenario, variant, 'stateful'])[:16]}"


def build_sequences(config: dict[str, Any]) -> list[dict[str, Any]]:
    fixtures = []
    for scenario in config["population"]["scenarioFamilies"]:
        for variant in config["population"]["variantIndices"]:
            fixtures.append(
                {
                    "sequence_id": _sequence_id(scenario, variant),
                    "split": "fresh_procedural_confirmation",
                    "scenario": scenario,
                    "variant": variant,
                    "initial_state": initial_state(variant),
                }
            )
    return sorted(fixtures, key=lambda row: row["sequence_id"])


def _tail(harness: DurableSandboxHarness, sequence_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
    item = proposal(
        f"{sequence_id}-tail",
        harness.store.state,
        [owner_operation(harness.store.state, "D2", 1)],
    )
    result = harness.execute(item)
    return item, result


def run_sequence(fixture: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    scenario = fixture["scenario"]
    sequence_id = fixture["sequence_id"]
    initial = deepcopy(fixture["initial_state"])
    harness = DurableSandboxHarness(initial, config)
    oracle = deepcopy(initial)
    expected_disposition = ""
    tail_retained: bool | None = None
    race_rejected: bool | None = None
    replay_idempotent: bool | None = None
    crash_recovered: bool | None = None
    partial_recovered: bool | None = None
    repeated_rollback_idempotent: bool | None = None
    provenance_restart_valid: bool | None = None
    provenance_tamper_detected: bool | None = None
    atomic_retained: bool | None = None

    if scenario == "revision_race":
        stale = proposal(f"{sequence_id}-stale", harness.store.state, [owner_operation(harness.store.state, "D1", 2)])
        stale_preview = harness.store.preview(stale)
        winner = proposal(f"{sequence_id}-winner", harness.store.state, [owner_operation(harness.store.state, "D1", 1)])
        winner_result = harness.execute(winner)
        oracle = apply_operations(oracle, winner["operations"])
        before_rejected = deepcopy(harness.store.state)
        stale_result = harness.store.commit(stale, stale_preview)
        harness._boundary()
        race_rejected = (
            winner_result["committed"]
            and not stale_result["committed"]
            and stale_result["reason"] == "stale_preview"
            and harness.store.state == before_rejected
        )
        tail_item, tail_result = _tail(harness, sequence_id)
        oracle = apply_operations(oracle, tail_item["operations"])
        tail_retained = tail_result["committed"]
        expected_disposition = "race_rejected_then_continued"

    elif scenario == "committed_replay_across_restart":
        item = proposal(f"{sequence_id}-subject", harness.store.state, [owner_operation(harness.store.state, "D1", 1)])
        first = harness.execute(item)
        oracle = apply_operations(oracle, item["operations"])
        harness, _ = DurableSandboxHarness.restart(harness.durable_image(), config)
        before_replay = deepcopy(harness.store.state)
        replay = harness.execute(item)
        replay_idempotent = first["committed"] and replay["replayed"] and harness.store.state == before_replay
        tail_item, tail_result = _tail(harness, sequence_id)
        oracle = apply_operations(oracle, tail_item["operations"])
        tail_retained = tail_result["committed"]
        expected_disposition = "replay_rejected_then_continued"

    elif scenario.startswith("crash_after_"):
        crash_point = scenario.removeprefix("crash_")
        item = proposal(f"{sequence_id}-subject", harness.store.state, [owner_operation(harness.store.state, "D1", 1)])
        image = harness.crash_image(item, crash_point)
        harness, recovery = DurableSandboxHarness.restart(image, config)
        if crash_point == "after_verify_before_finalize":
            oracle = apply_operations(oracle, item["operations"])
            crash_recovered = recovery["action"] == "finalize_retained"
            expected_disposition = "recovered_finalize_then_continued"
        else:
            crash_recovered = recovery["action"] == "restore_before_state"
            expected_disposition = "recovered_rollback_then_continued"
        tail_item, tail_result = _tail(harness, sequence_id)
        oracle = apply_operations(oracle, tail_item["operations"])
        tail_retained = tail_result["committed"]

    elif scenario == "partial_multi_entity_write":
        item = proposal(f"{sequence_id}-subject", harness.store.state, atomic_operations(harness.store.state))
        image = harness.crash_image(item, "after_apply_before_verify", partial_write=True)
        harness, recovery = DurableSandboxHarness.restart(image, config)
        partial_recovered = recovery["action"] == "restore_before_state" and harness.store.state == initial
        tail_item, tail_result = _tail(harness, sequence_id)
        oracle = apply_operations(oracle, tail_item["operations"])
        tail_retained = tail_result["committed"]
        expected_disposition = "partial_write_rolled_back_then_continued"

    elif scenario == "repeated_rollback_across_restart":
        item = proposal(f"{sequence_id}-subject", harness.store.state, [owner_operation(harness.store.state, "D1", 1)])
        executed = harness.execute(item, explicit_rollback=True)
        harness, _ = DurableSandboxHarness.restart(harness.durable_image(), config)
        before_repeat = deepcopy(harness.store.state)
        repeat_one = harness.request_rollback(item["transaction_id"])
        repeat_two = harness.request_rollback(item["transaction_id"])
        repeated_rollback_idempotent = (
            executed["rolled_back"]
            and repeat_one.get("idempotent", False)
            and repeat_two.get("idempotent", False)
            and harness.store.state == before_repeat == initial
        )
        tail_item, tail_result = _tail(harness, sequence_id)
        oracle = apply_operations(oracle, tail_item["operations"])
        tail_retained = tail_result["committed"]
        expected_disposition = "rollback_idempotent_then_continued"

    elif scenario == "provenance_multi_restart":
        first = proposal(f"{sequence_id}-first", harness.store.state, [owner_operation(harness.store.state, "D1", 1)])
        first_result = harness.execute(first)
        oracle = apply_operations(oracle, first["operations"])
        harness, _ = DurableSandboxHarness.restart(harness.durable_image(), config)
        second = proposal(f"{sequence_id}-second", harness.store.state, [owner_operation(harness.store.state, "D2", 2)])
        second_result = harness.execute(second)
        oracle = apply_operations(oracle, second["operations"])
        harness, _ = DurableSandboxHarness.restart(harness.durable_image(), config)
        provenance_restart_valid = first_result["committed"] and second_result["committed"] and harness.provenance_valid()
        tail_item, tail_result = _tail(harness, sequence_id)
        oracle = apply_operations(oracle, tail_item["operations"])
        tail_retained = tail_result["committed"]
        expected_disposition = "multiple_restarts_verified_then_continued"

    elif scenario == "provenance_tamper_detection":
        item = proposal(f"{sequence_id}-subject", harness.store.state, [owner_operation(harness.store.state, "D1", 1)])
        executed = harness.execute(item)
        oracle = apply_operations(oracle, item["operations"])
        image = harness.durable_image()
        image_state = deepcopy(image["state"])
        image["log"][0]["event"] = "tampered_event"
        try:
            DurableSandboxHarness.restart(image, config)
            detected = False
        except ProvenanceError:
            detected = True
        provenance_tamper_detected = executed["committed"] and detected and image["state"] == image_state
        expected_disposition = "provenance_tamper_failed_closed"

    elif scenario == "atomic_multi_entity_retained":
        item = proposal(f"{sequence_id}-subject", harness.store.state, atomic_operations(harness.store.state))
        executed = harness.execute(item)
        oracle = apply_operations(oracle, item["operations"])
        harness, _ = DurableSandboxHarness.restart(harness.durable_image(), config)
        atomic_retained = executed["committed"] and harness.store.state == oracle
        tail_item, tail_result = _tail(harness, sequence_id)
        oracle = apply_operations(oracle, tail_item["operations"])
        tail_retained = tail_result["committed"]
        expected_disposition = "atomic_retained_then_continued"

    else:
        raise ValueError(f"unknown scenario: {scenario}")

    exact_final = harness.store.state == oracle
    scenario_success = {
        "revision_race": bool(race_rejected and tail_retained),
        "committed_replay_across_restart": bool(replay_idempotent and tail_retained),
        "crash_after_preview": bool(crash_recovered and tail_retained),
        "crash_after_snapshot_before_apply": bool(crash_recovered and tail_retained),
        "crash_after_apply_before_verify": bool(crash_recovered and tail_retained),
        "crash_after_verify_before_finalize": bool(crash_recovered and tail_retained),
        "partial_multi_entity_write": bool(partial_recovered and tail_retained),
        "repeated_rollback_across_restart": bool(repeated_rollback_idempotent and tail_retained),
        "provenance_multi_restart": bool(provenance_restart_valid and tail_retained),
        "provenance_tamper_detection": bool(provenance_tamper_detected),
        "atomic_multi_entity_retained": bool(atomic_retained and tail_retained),
    }[scenario]
    disposition_exact = exact_final and scenario_success
    return {
        "sequence_id": sequence_id,
        "scenario": scenario,
        "variant": fixture["variant"],
        "disposition": expected_disposition if disposition_exact else "sequence_contract_failure",
        "expected_disposition": expected_disposition,
        "expected_disposition_exact": disposition_exact,
        "exact_oracle_final_state": exact_final,
        "serializable_boundaries": exact_final and harness.visible_invariants_hold(),
        "revision_race_rejected": race_rejected,
        "replay_idempotent": replay_idempotent,
        "crash_recovered": crash_recovered,
        "partial_write_recovered": partial_recovered,
        "repeated_rollback_idempotent": repeated_rollback_idempotent,
        "provenance_restart_valid": provenance_restart_valid,
        "provenance_tamper_detected": provenance_tamper_detected,
        "atomic_multi_entity_retained": atomic_retained,
        "post_recovery_continuation": tail_retained,
        "invariants_preserved": harness.visible_invariants_hold(),
        "zero_unauthorized_retained_mutation": harness.authorized_retained_mutations_hold(),
        "provenance_chain_valid": harness.provenance_valid(),
        "final_state_hash": canonical_hash(harness.store.state),
        "oracle_final_state_hash": canonical_hash(oracle),
        "log_event_types": [entry["event"] for entry in harness.store.log],
    }


def _rate(rows: list[dict[str, Any]], key: str) -> float:
    relevant = [row for row in rows if row[key] is not None]
    if not relevant:
        raise ValueError(f"metric {key} has no relevant rows")
    return sum(bool(row[key]) for row in relevant) / len(relevant)


def evaluate_sequences(fixtures: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    results = [run_sequence(fixture, config) for fixture in fixtures]
    scenario_counts = Counter(row["scenario"] for row in results)
    summary = {
        "sequence_count": len(results),
        "scenario_count": len(scenario_counts),
        "scenario_counts": dict(sorted(scenario_counts.items())),
        "expected_disposition_accuracy": _rate(results, "expected_disposition_exact"),
        "exact_oracle_final_state": _rate(results, "exact_oracle_final_state"),
        "serializable_boundary_rate": _rate(results, "serializable_boundaries"),
        "revision_race_rejection": _rate(results, "revision_race_rejected"),
        "replay_idempotence": _rate(results, "replay_idempotent"),
        "crash_recovery": _rate(results, "crash_recovered"),
        "partial_write_recovery": _rate(results, "partial_write_recovered"),
        "repeated_rollback_idempotence": _rate(results, "repeated_rollback_idempotent"),
        "provenance_restart_validity": _rate(results, "provenance_restart_valid"),
        "provenance_tamper_detection": _rate(results, "provenance_tamper_detected"),
        "atomic_multi_entity_retention": _rate(results, "atomic_multi_entity_retained"),
        "post_recovery_continuation": _rate(results, "post_recovery_continuation"),
        "invariant_preservation": _rate(results, "invariants_preserved"),
        "zero_unauthorized_retained_mutation": _rate(results, "zero_unauthorized_retained_mutation"),
        "provenance_chain_validity": _rate(results, "provenance_chain_valid"),
    }
    return {"results": results, "summary": summary}


def evaluate_gates(
    evaluation: dict[str, Any],
    access: dict[str, int],
    config: dict[str, Any],
) -> dict[str, bool]:
    summary = evaluation["summary"]
    gates = config["confirmationGates"]
    return {
        "sequence_count": summary["sequence_count"] == gates["requiredSequenceCount"],
        "scenario_count": summary["scenario_count"] == gates["requiredScenarioCount"],
        "sequences_per_scenario": set(summary["scenario_counts"].values()) == {gates["requiredSequencesPerScenario"]},
        "expected_disposition": summary["expected_disposition_accuracy"] == gates["requiredExpectedDispositionAccuracy"],
        "oracle_final_state": summary["exact_oracle_final_state"] == gates["requiredExactOracleFinalState"],
        "serializable_boundaries": summary["serializable_boundary_rate"] == gates["requiredSerializableBoundaryRate"],
        "revision_race": summary["revision_race_rejection"] == gates["requiredRevisionRaceRejection"],
        "replay_idempotence": summary["replay_idempotence"] == gates["requiredReplayIdempotence"],
        "crash_recovery": summary["crash_recovery"] == gates["requiredCrashRecovery"],
        "partial_write_recovery": summary["partial_write_recovery"] == gates["requiredPartialWriteRecovery"],
        "repeated_rollback": summary["repeated_rollback_idempotence"] == gates["requiredRepeatedRollbackIdempotence"],
        "provenance_restart": summary["provenance_restart_validity"] == gates["requiredProvenanceRestartValidity"],
        "provenance_tamper": summary["provenance_tamper_detection"] == gates["requiredProvenanceTamperDetection"],
        "atomic_multi_entity": summary["atomic_multi_entity_retention"] == gates["requiredAtomicMultiEntityRetention"],
        "post_recovery_continuation": summary["post_recovery_continuation"] == gates["requiredPostRecoveryContinuation"],
        "invariant_preservation": summary["invariant_preservation"] == gates["requiredInvariantPreservation"],
        "zero_unauthorized_mutation": summary["zero_unauthorized_retained_mutation"] == gates["requiredZeroUnauthorizedRetainedMutation"],
        "provenance_chain_validity": summary["provenance_chain_validity"] == 1.0,
        "zero_disallowed_access": all(
            access[key] <= gates[maximum]
            for key, maximum in {
                "evaluation_record_count": "maximumEvaluationRecordCount",
                "manual_judgment_count": "maximumManualJudgmentCount",
                "model_load_count": "maximumModelLoadCount",
                "model_generation_count": "maximumModelGenerationCount",
                "API_call_count": "maximumAPICallCount",
                "training_run_count": "maximumTrainingRunCount",
                "provisional_ontology_use_count": "maximumProvisionalOntologyUseCount",
                "real_service_call_count": "maximumRealServiceCallCount",
                "external_side_effect_count": "maximumExternalSideEffectCount",
                "real_execution_count": "maximumRealExecutionCount",
            }.items()
        ),
    }


__all__ = [
    "DurableSandboxHarness",
    "ProvenanceError",
    "atomic_operations",
    "build_sequences",
    "compose_config",
    "evaluate_gates",
    "evaluate_sequences",
    "owner_operation",
    "proposal",
    "run_sequence",
]
