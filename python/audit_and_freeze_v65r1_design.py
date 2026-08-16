#!/usr/bin/env python3
"""Independently audit and freeze the V65r1 pre-subset predictive repair."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from v64_external_eig import (
    filter_public_history,
    load_family,
    score_all_actions,
    true_transition,
)


def _score_atoms(family, atoms: list[dict]) -> np.ndarray:
    values = []
    for action in family.canonical_actions:
        conditional = []
        weights = []
        for atom in atoms:
            state = np.asarray(atom["state"], dtype=np.float64)
            transition = true_transition(
                family,
                int(atom["identity"]),
                float(atom["theta"]),
                int(action),
            )
            conditional.append((state @ transition) @ family.model.observation[action])
            weights.append(float(atom["weight"]))
        conditional_array = np.asarray(conditional)
        weights_array = np.asarray(weights)
        weights_array /= weights_array.sum()
        predictive = weights_array @ conditional_array
        information = 0.0
        for weight, row in zip(weights_array, conditional_array, strict=True):
            mask = row > 0.0
            information += float(
                weight * np.sum(row[mask] * np.log(row[mask] / predictive[mask]))
            )
        values.append(information)
    return np.asarray(values)


def _fixture_replication(family, fixture: dict, replication: int) -> dict:
    outer = int(fixture["outerParticles"])
    inner = int(fixture["innerParticles"])
    repeat_count = int(fixture["repeats"])
    initial_observation = family.model.observations.index(
        fixture["publicHistory"]["initial_observation"]
    )
    reset_likelihood = family.model.observation[0, :, initial_observation]
    exact_state = family.model.initial * reset_likelihood
    exact_state /= exact_state.sum()
    repeat_plugin: list[list[dict]] = []
    repeat_rb: list[list[dict]] = []
    for repeat in range(repeat_count):
        identity_rows: list[tuple[list[dict], list[dict], float]] = []
        for identity in range(2):
            seed = int.from_bytes(
                hashlib.sha256(
                    json.dumps(
                        [fixture["seed"], replication, repeat, identity],
                        separators=(",", ":"),
                    ).encode()
                ).digest()[:8],
                "big",
            )
            rng = np.random.default_rng(seed)
            theta = 0.6 + 0.35 * rng.beta(2.0, 2.0, size=outer)
            states = rng.choice(
                len(family.model.states),
                size=(outer, inner),
                p=family.model.initial,
            )
            likelihood = reset_likelihood[states]
            evidence = likelihood.mean(axis=1)
            identity_evidence = float(evidence.mean())
            outer_weights = evidence / evidence.sum()
            plugin_rows = []
            rb_rows = []
            for particle in range(outer):
                weighted_counts = np.bincount(
                    states[particle],
                    weights=likelihood[particle],
                    minlength=len(family.model.states),
                ).astype(np.float64)
                weighted_counts /= weighted_counts.sum()
                base = {
                    "identity": identity,
                    "theta": float(theta[particle]),
                    "weight": float(outer_weights[particle]),
                }
                plugin_rows.append({**base, "state": weighted_counts})
                rb_rows.append({**base, "state": exact_state.copy()})
            identity_rows.append((plugin_rows, rb_rows, identity_evidence))
        identity_evidence = np.asarray([row[2] for row in identity_rows])
        identity_weights = identity_evidence / identity_evidence.sum()
        plugin_atoms = []
        rb_atoms = []
        for identity, (plugin_rows, rb_rows, _) in enumerate(identity_rows):
            for row in plugin_rows:
                plugin_atoms.append(
                    {**row, "weight": float(identity_weights[identity]) * row["weight"]}
                )
            for row in rb_rows:
                rb_atoms.append(
                    {**row, "weight": float(identity_weights[identity]) * row["weight"]}
                )
        repeat_plugin.append(plugin_atoms)
        repeat_rb.append(rb_atoms)
    pooled_plugin = [
        {**atom, "weight": atom["weight"] / repeat_count}
        for atoms in repeat_plugin
        for atom in atoms
    ]
    pooled_rb = [
        {**atom, "weight": atom["weight"] / repeat_count}
        for atoms in repeat_rb
        for atom in atoms
    ]
    return {
        "plugin": _score_atoms(family, pooled_plugin),
        "rao_blackwellized": _score_atoms(family, pooled_rb),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repair", default="configs/v65r1-nested-predictive-repair.json"
    )
    parser.add_argument(
        "--plan", default="docs/v65r1-nested-predictive-repair-plan.md"
    )
    parser.add_argument(
        "--audit", default="outputs/v65r1-nested-predictive-repair/design-audit.json"
    )
    parser.add_argument("--output", default="configs/v65r1-design-lock.json")
    args = parser.parse_args()

    repair_path = (PROJECT_ROOT / args.repair).resolve()
    plan_path = (PROJECT_ROOT / args.plan).resolve()
    audit_path = (PROJECT_ROOT / args.audit).resolve()
    output_path = (PROJECT_ROOT / args.output).resolve()
    if output_path.exists():
        raise RuntimeError("V65r1 design already frozen")
    repair = json.loads(repair_path.read_text())
    source_path = (PROJECT_ROOT / repair["sourceV65DesignLock"]).resolve()
    source = json.loads(source_path.read_text())
    base = source["config_payload"]
    errors: list[str] = []

    source_ok = bool(
        source["authorization"]["write_and_audit_smc2_eig_implementation"]
        and not source["authorization"]["materialize_subset"]
        and not source["authorization"]["run_evaluation"]
        and not source["authorization"]["reward_planning"]
        and file_sha256(PROJECT_ROOT / source["config"]) == source["config_sha256"]
        and file_sha256(PROJECT_ROOT / source["preregistration"])
        == source["preregistration_sha256"]
        and file_sha256(PROJECT_ROOT / source["design_audit"])
        == source["design_audit_sha256"]
    )
    if not source_ok:
        errors.append("V65 source design authorization or hash binding is invalid")

    unchanged = repair["unchanged"]
    repair_scope_ok = bool(
        repair["defectDiscoveryStage"]
        == "candidate_implementation_fixtures_before_implementation_lock_or_subset_materialization"
        and repair["repair"]["scope"] == "approximate_acquisition_predictive_only"
        and repair["repair"]["staticPosteriorSource"].startswith(
            "unchanged_three_repeat_equal_weight_pooled_SMC2"
        )
        and repair["repair"]["likelihoodSource"].startswith("unchanged_127_particle")
        and repair["repair"]["posteriorStateAccuracySource"].startswith(
            "unchanged_pooled_inner_particle"
        )
        and repair["repair"]["forbidParticleAncestryInnerStateAndRepeatIdentityAsTargets"]
        and repair["repair"]["poolBeforeRaoBlackwellizationAndScoring"]
        and len(unchanged) == 10
        and base["subset"]["records"] == 48
        and base["smcSquared"]["outerThetaParticleBudgets"] == [31, 127, 509]
        and base["smcSquared"]["innerStateParticleBudget"] == 127
        and base["smcSquared"]["independentRepeatsPerBudget"] == 3
    )
    if not repair_scope_ok:
        errors.append("V65r1 is not confined to the preregistered predictive repair")

    boundary = repair["claimBoundary"]
    boundary_ok = bool(
        boundary["particleStatePosteriorStillEvaluated"]
        and not boundary["particleStateUsedDirectlyForAcquisition"]
        and boundary["knownFiniteStateModelRaoBlackwellizedForAcquisition"]
        and boundary["staticIdentityThetaPosteriorRemainsSMC2Approximate"]
        and boundary["pairedV64HistoryReuse"]
        and not any(
            boundary[key]
            for key in (
                "independentExactBenchmarkReplication",
                "sequentialApproximateAdaptiveRollout",
                "rewardPlanning",
                "formalVerification",
                "humanData",
                "modelAccess",
                "adapterTraining",
            )
        )
    )
    if not boundary_ok:
        errors.append("V65r1 Rao-Blackwellized claim boundary is too broad")

    stage = repair["stageAuthorization"]
    firewall_ok = bool(
        set(repair["firewall"].values()) == {"forbidden"}
        and stage["writeAndAuditRepairedImplementation"]
        and not any(
            value
            for key, value in stage.items()
            if key != "writeAndAuditRepairedImplementation"
        )
    )
    if not firewall_ok:
        errors.append("V65r1 repair firewall or authorization is invalid")

    downstream = (
        "configs/v65-implementation-lock.json",
        "configs/v65r1-implementation-lock.json",
        "configs/v65-subset-seal.json",
        "configs/v65r1-subset-seal.json",
        "data/v65-smc2-eig-portability",
        "outputs/v65-smc2-eig-portability/evaluation",
        "outputs/v65r1-nested-predictive-repair/evaluation",
    )
    downstream_absent = not any((PROJECT_ROOT / path).exists() for path in downstream)
    if not downstream_absent:
        errors.append("V65/V65r1 frozen implementation, subset, or evaluation already exists")

    family = load_family()
    fixture = repair["feasibilityFixture"]
    exact_belief, _ = filter_public_history(
        family,
        fixture["publicHistory"]["initial_observation"],
        fixture["publicHistory"]["actions"],
        fixture["publicHistory"]["observations"],
    )
    exact = np.asarray([row["eig"] for row in score_all_actions(family, exact_belief)])
    replications = [
        _fixture_replication(family, fixture, replication)
        for replication in range(int(fixture["independentReplications"]))
    ]
    plugin_errors = np.asarray(
        [np.mean(np.abs(row["plugin"] - exact)) for row in replications]
    )
    rb_errors = np.asarray(
        [np.mean(np.abs(row["rao_blackwellized"] - exact)) for row in replications]
    )
    feasibility = {
        "exact_eig": exact.tolist(),
        "plugin_eig_by_replication": [row["plugin"].tolist() for row in replications],
        "rao_blackwellized_eig_by_replication": [
            row["rao_blackwellized"].tolist() for row in replications
        ],
        "mean_plugin_eig_vector_bias_nats": float(plugin_errors.mean()),
        "minimum_plugin_eig_vector_bias_nats": float(plugin_errors.min()),
        "mean_rao_blackwellized_eig_vector_error_nats": float(rb_errors.mean()),
        "maximum_rao_blackwellized_eig_vector_error_nats": float(rb_errors.max()),
        "replications": len(replications),
        "not_evaluation_population": bool(fixture["notEvaluationPopulation"]),
    }
    feasibility_ok = bool(
        feasibility["mean_plugin_eig_vector_bias_nats"]
        >= float(fixture["minimumMeanPluginEIGVectorBiasNats"])
        and feasibility["mean_rao_blackwellized_eig_vector_error_nats"]
        <= float(fixture["maximumMeanRaoBlackwellizedEIGVectorErrorNats"])
        and fixture["notEvaluationPopulation"]
    )
    if not feasibility_ok:
        errors.append("independent nested-bias fixture did not qualify the sole repair")

    checks = {
        "source_v65_design_authorization_and_bindings": source_ok,
        "repair_confined_to_acquisition_conditional_state": repair_scope_ok,
        "Rao_Blackwellized_claim_boundary": boundary_ok,
        "repair_only_firewall": firewall_ok,
        "downstream_absent": downstream_absent,
        "independent_pre_subset_bias_and_repair_fixture": feasibility_ok,
    }
    audit = {
        "schema_version": "65r1",
        "experiment": "v65r1_pre_subset_design_repair_audit",
        "passed": not errors and all(checks.values()),
        "decision": (
            "authorize_v65r1_repaired_implementation_only"
            if not errors and all(checks.values())
            else "reject_or_repair_v65r1_design"
        ),
        "errors": errors,
        "checks": checks,
        "feasibility_fixture": feasibility,
        "unchanged_base_design": {
            "subset_records": base["subset"]["records"],
            "outer_budgets": base["smcSquared"]["outerThetaParticleBudgets"],
            "inner_particles": base["smcSquared"]["innerStateParticleBudget"],
            "repeats": base["smcSquared"]["independentRepeatsPerBudget"],
            "gates_sha256": hashlib.sha256(
                json.dumps(base["gates"], sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest(),
            "seeds_sha256": hashlib.sha256(
                json.dumps(base["seeds"], sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest(),
        },
        "data_access": {
            "v64_selection_public_records_loaded": 0,
            "v64_selection_audit_records_loaded": 0,
            "v64_evaluation_records_loaded": 0,
            "v65_subset_records_materialized": 0,
            "candidate_evaluation_runs": 0,
            "human_records": 0,
            "model_forward_passes": 0,
            "adapter_training_runs": 0,
        },
    }
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    if not audit["passed"]:
        print(json.dumps(audit, indent=2, sort_keys=True))
        raise SystemExit(1)

    effective = copy.deepcopy(base)
    effective["schemaVersion"] = "65r1"
    effective["nestedPredictiveRepair"] = repair["repair"]
    effective["approximateAcquisition"][
        "integrateInnerStateBeforeTreatingAnOuterParticleAsAStaticLatentAtom"
    ] = False
    effective["approximateAcquisition"]["raoBlackwellizeKnownConditionalState"] = True
    effective["approximateAcquisition"][
        "particleStatePredictiveRole"
    ] = "mandatory_negative_control_and_diagnostic"
    effective["claimBoundary"].update(boundary)
    effective["claimBoundary"]["primaryClaim"] = boundary["primaryClaim"]
    lock = {
        "schema_version": "65r1",
        "experiment": "v65r1_design_lock",
        "source_v65_design_lock": str(source_path.relative_to(PROJECT_ROOT)),
        "source_v65_design_lock_sha256": file_sha256(source_path),
        "repair": str(repair_path.relative_to(PROJECT_ROOT)),
        "repair_sha256": file_sha256(repair_path),
        "repair_payload": repair,
        "preregistration": str(plan_path.relative_to(PROJECT_ROOT)),
        "preregistration_sha256": file_sha256(plan_path),
        "design_audit": str(audit_path.relative_to(PROJECT_ROOT)),
        "design_audit_sha256": file_sha256(audit_path),
        "config_payload": effective,
        "unchanged_gates_sha256": audit["unchanged_base_design"]["gates_sha256"],
        "unchanged_seeds_sha256": audit["unchanged_base_design"]["seeds_sha256"],
        "feasibility_summary": feasibility,
        "authorization": {
            "modify_v65_or_v65r1_design": False,
            "write_and_audit_repaired_implementation": True,
            "materialize_subset": False,
            "run_evaluation": False,
            "reward_planning": False,
            "formal_verification": False,
            "human_data": False,
            "model_access": False,
            "adapter_training": False,
        },
    }
    lock["lock_payload_sha256"] = hashlib.sha256(
        json.dumps(lock, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    output_path.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"audit": audit, "lock": lock}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
