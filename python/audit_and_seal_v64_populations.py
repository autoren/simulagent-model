#!/usr/bin/env python3
"""Audit and seal the V64 populations before evaluator implementation."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from v64_external_eig import (
    assert_public_selection_payload,
    load_family,
    simulate_step,
)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def canonical_hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def initial_observation(family, state: int) -> str:
    support = np.flatnonzero(family.model.observation[0, state] > 0.0)
    if len(support) != 1:
        raise RuntimeError("V64 seal expected deterministic source observations")
    return family.model.observations[int(support[0])]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="data/v64-external-multi-action-eig/manifest.json")
    parser.add_argument(
        "--audit", default="outputs/v64-external-multi-action-eig/population-audit.json"
    )
    parser.add_argument("--output", default="configs/v64-population-seal.json")
    args = parser.parse_args()
    manifest_path = (PROJECT_ROOT / args.manifest).resolve()
    audit_path = (PROJECT_ROOT / args.audit).resolve()
    output_path = (PROJECT_ROOT / args.output).resolve()
    if output_path.exists():
        raise RuntimeError("V64 populations already sealed")
    manifest = json.loads(manifest_path.read_text())
    lock_path = (PROJECT_ROOT / manifest["implementation_lock"]).resolve()
    lock = json.loads(lock_path.read_text())
    design = json.loads((PROJECT_ROOT / lock["design_lock"]).read_text())
    config = design["config_payload"]
    errors: list[str] = []

    upstream_ok = bool(
        file_sha256(lock_path) == manifest["implementation_lock_sha256"]
        and lock["authorization"]["construct_and_audit_sealed_populations"]
        and not lock["authorization"]["run_v64_evaluation"]
        and file_sha256(PROJECT_ROOT / manifest["generator"])
        == manifest["generator_sha256"]
        and all(
            file_sha256(PROJECT_ROOT / relative) == digest
            for relative, digest in lock["source_sha256"].items()
        )
    )
    if not upstream_ok:
        errors.append("V64 implementation or generator binding failed")

    files = {
        name: (PROJECT_ROOT / row["path"]).resolve()
        for name, row in manifest["files"].items()
    }
    file_hashes_ok = all(
        file_sha256(files[name]) == manifest["files"][name]["sha256"]
        for name in files
    )
    if not file_hashes_ok:
        errors.append("V64 population file hash mismatch")
    rows = {name: read_jsonl(path) for name, path in files.items()}
    counts_ok = all(
        len(rows[name]) == manifest["counts"][name] for name in rows
    ) and manifest["counts"] == {
        "selection_public": 192,
        "selection_audit": 192,
        "adaptive_public": 512,
        "adaptive_audit": 512,
        "sbc_public": 256,
        "sbc_audit": 256,
    }
    if not counts_ok:
        errors.append("V64 population counts differ from preregistration")

    family = load_family()
    selection_public = rows["selection_public"]
    selection_audit = rows["selection_audit"]
    selection_ids = [row["record_id"] for row in selection_public]
    selection_pairing_ok = bool(
        selection_ids == [row["record_id"] for row in selection_audit]
        and len(selection_ids) == len(set(selection_ids))
    )
    prefix_counts = {
        length: sum(row["prefix_length"] == length for row in selection_public)
        for length in config["selectionPopulation"]["publicPrefixLengths"]
    }
    selection_public_ok = selection_pairing_ok and all(
        count == config["selectionPopulation"]["recordsPerPublicPrefixLength"]
        for count in prefix_counts.values()
    )
    selection_semantics_ok = True
    public_history_fingerprints: list[str] = []
    for public, truth in zip(selection_public, selection_audit, strict=True):
        try:
            assert_public_selection_payload(public)
        except (PermissionError, ValueError):
            selection_public_ok = False
        if public["prefix_length"] != len(public["actions"]) or len(public["actions"]) != len(public["observations"]):
            selection_public_ok = False
        if truth["public_fingerprint"] != canonical_hash(public):
            selection_public_ok = False
        if public["initial_observation"] != initial_observation(family, truth["initial_state"]):
            selection_semantics_ok = False
        state = truth["initial_state"]
        reconstructed_states = [state]
        reconstructed_observations = []
        for action, uniform in zip(public["actions"], truth["transition_uniforms"], strict=True):
            state, observation, _ = simulate_step(
                family,
                truth["identity"],
                truth["theta"],
                state,
                action,
                uniform,
                0.5,
            )
            reconstructed_states.append(state)
            reconstructed_observations.append(family.model.observations[observation])
        if reconstructed_states != truth["states"] or reconstructed_observations != public["observations"]:
            selection_semantics_ok = False
        public_history_fingerprints.append(
            canonical_hash(
                {
                    "initial_observation": public["initial_observation"],
                    "actions": public["actions"],
                    "observations": public["observations"],
                }
            )
        )
    if not selection_public_ok or not selection_semantics_ok:
        errors.append("V64 selection pairing, public firewall, quota, or replay failed")

    adaptive_public = rows["adaptive_public"]
    adaptive_audit = rows["adaptive_audit"]
    adaptive_ids = [row["scenario_id"] for row in adaptive_public]
    adaptive_ok = bool(
        adaptive_ids == [row["scenario_id"] for row in adaptive_audit]
        and len(adaptive_ids) == len(set(adaptive_ids)) == 512
        and sum(row["identity"] == 0 for row in adaptive_audit) == 256
        and sum(row["identity"] == 1 for row in adaptive_audit) == 256
    )
    stream_hashes: list[str] = []
    for public, truth in zip(adaptive_public, adaptive_audit, strict=True):
        if set(public) != {"scenario_id", "replication", "initial_observation"}:
            adaptive_ok = False
        if truth["public_fingerprint"] != canonical_hash(public):
            adaptive_ok = False
        if public["initial_observation"] != initial_observation(family, truth["initial_state"]):
            adaptive_ok = False
        if not family.theta_support[0] <= truth["theta"] <= family.theta_support[1]:
            adaptive_ok = False
        if set(truth["policy_streams"]) != set(config["pairedAdaptiveEvaluation"]["policies"]):
            adaptive_ok = False
        for policy, streams in truth["policy_streams"].items():
            for name in ("transition_uniforms", "observation_uniforms"):
                values = streams[name]
                if len(values) != 8 or not all(0.0 <= value < 1.0 for value in values):
                    adaptive_ok = False
                stream_hashes.append(canonical_hash(["adaptive", public["scenario_id"], policy, name, values]))
        if len(truth["random_actions"]) != 8 or not set(truth["random_actions"]) <= set(family.model.actions):
            adaptive_ok = False
        stream_hashes.append(canonical_hash(["adaptive-actions", public["scenario_id"], truth["random_actions"]]))
    if not adaptive_ok:
        errors.append("V64 paired adaptive population balance, firewall, or streams failed")

    sbc_public = rows["sbc_public"]
    sbc_audit = rows["sbc_audit"]
    sbc_ids = [row["scenario_id"] for row in sbc_public]
    sbc_ok = bool(
        sbc_ids == [row["scenario_id"] for row in sbc_audit]
        and len(sbc_ids) == len(set(sbc_ids)) == 256
        and set(adaptive_ids).isdisjoint(sbc_ids)
    )
    for public, truth in zip(sbc_public, sbc_audit, strict=True):
        if set(public) != {"scenario_id", "replication", "initial_observation"}:
            sbc_ok = False
        if truth["public_fingerprint"] != canonical_hash(public):
            sbc_ok = False
        if public["initial_observation"] != initial_observation(family, truth["initial_state"]):
            sbc_ok = False
        if not family.theta_support[0] <= truth["theta"] <= family.theta_support[1]:
            sbc_ok = False
        for name in ("transition_uniforms", "observation_uniforms"):
            values = truth[name]
            if len(values) != 8 or not all(0.0 <= value < 1.0 for value in values):
                sbc_ok = False
            stream_hashes.append(canonical_hash(["sbc", public["scenario_id"], name, values]))
        draws = np.asarray(truth["posterior_draw_uniforms"])
        if draws.shape != (127, 3) or np.min(draws) < 0.0 or np.max(draws) >= 1.0:
            sbc_ok = False
        if len(truth["rank_tie_uniforms"]) != 3:
            sbc_ok = False
        stream_hashes.append(canonical_hash(["sbc-draws", public["scenario_id"], truth["posterior_draw_uniforms"]]))
    if not sbc_ok:
        errors.append("V64 SBC population firewall, prior support, or streams failed")

    streams_ok = len(stream_hashes) == len(set(stream_hashes))
    if not streams_ok:
        errors.append("V64 random stream collision detected")
    truth_firewall_ok = all(
        not ({"identity", "theta", "initial_state", "states", "policy_streams"} & set(row))
        for row in selection_public + adaptive_public + sbc_public
    )
    if not truth_firewall_ok:
        errors.append("V64 audit truth leaked into public populations")

    generator_frozen_ok = file_sha256(PROJECT_ROOT / manifest["generator"]) == manifest["generator_sha256"]
    downstream_absent = not any(
        (PROJECT_ROOT / path).exists()
        for path in (
            "configs/v64-evaluation-implementation-lock.json",
            "configs/v64-outcome-lock.json",
            "outputs/v64-external-multi-action-eig/evaluation",
        )
    )
    if not downstream_absent:
        errors.append("V64 evaluation artifact exists before population seal")

    audit = {
        "schema_version": 64,
        "experiment": "v64_population_audit",
        "passed": not errors,
        "decision": "seal_v64_populations_and_authorize_evaluator_implementation" if not errors else "reject_v64_populations",
        "errors": errors,
        "checks": {
            "implementation_and_generator_bindings": upstream_ok,
            "population_file_hashes": file_hashes_ok,
            "preregistered_counts": counts_ok,
            "selection_public_pairing_quotas_and_replay": selection_public_ok and selection_semantics_ok,
            "paired_adaptive_balance_and_streams": adaptive_ok,
            "adaptive_SBC_support_and_streams": sbc_ok,
            "truth_firewall": truth_firewall_ok,
            "stream_fingerprint_uniqueness": streams_ok,
            "generator_frozen": generator_frozen_ok,
            "evaluation_downstream_absent": downstream_absent,
        },
        "population_summary": {
            "counts": manifest["counts"],
            "selection_prefix_counts": prefix_counts,
            "selection_unique_public_history_fraction": len(set(public_history_fingerprints)) / len(public_history_fingerprints),
            "adaptive_identity_counts": {
                "clockwise_failure": sum(row["identity"] == 0 for row in adaptive_audit),
                "counterclockwise_failure": sum(row["identity"] == 1 for row in adaptive_audit),
            },
            "adaptive_theta_mean": float(np.mean([row["theta"] for row in adaptive_audit])),
            "sbc_identity_counts": {
                "clockwise_failure": sum(row["identity"] == 0 for row in sbc_audit),
                "counterclockwise_failure": sum(row["identity"] == 1 for row in sbc_audit),
            },
            "sbc_theta_mean": float(np.mean([row["theta"] for row in sbc_audit])),
            "stream_fingerprints": len(stream_hashes),
            "stream_collisions": len(stream_hashes) - len(set(stream_hashes)),
        },
        "data_access": {
            "selection_public_records_audited": len(selection_public),
            "selection_audit_records_audited": len(selection_audit),
            "adaptive_public_records_audited": len(adaptive_public),
            "adaptive_audit_records_audited": len(adaptive_audit),
            "sbc_public_records_audited": len(sbc_public),
            "sbc_audit_records_audited": len(sbc_audit),
            "candidate_evaluation_runs": 0,
            "human_record_access_count": 0,
            "simulated_human_record_count": 0,
            "model_forward_pass_count": 0,
            "adapter_training_run_count": 0,
        },
    }
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    if errors:
        print(json.dumps(audit, indent=2, sort_keys=True))
        raise SystemExit(1)

    seal = {
        "schema_version": 64,
        "experiment": "v64_population_seal",
        "implementation_lock": str(lock_path.relative_to(PROJECT_ROOT)),
        "implementation_lock_sha256": file_sha256(lock_path),
        "manifest": str(manifest_path.relative_to(PROJECT_ROOT)),
        "manifest_sha256": file_sha256(manifest_path),
        "population_audit": str(audit_path.relative_to(PROJECT_ROOT)),
        "population_audit_sha256": file_sha256(audit_path),
        "generator": manifest["generator"],
        "generator_sha256": manifest["generator_sha256"],
        "seal_auditor": str(Path(__file__).resolve().relative_to(PROJECT_ROOT)),
        "seal_auditor_sha256": file_sha256(Path(__file__).resolve()),
        "files": manifest["files"],
        "counts": manifest["counts"],
        "authorization": {
            "modify_or_rebuild_populations": False,
            "write_and_audit_evaluation_implementation": True,
            "run_one_immutable_evaluation": False,
            "approximate_particle_acquisition": False,
            "reward_planning": False,
            "formal_verification": False,
            "access_human_data": False,
            "simulate_human_data": False,
            "model_access": False,
            "adapter_training": False,
        },
    }
    seal["lock_payload_sha256"] = hashlib.sha256(
        json.dumps(seal, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    output_path.write_text(json.dumps(seal, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"audit": audit, "seal": seal}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
