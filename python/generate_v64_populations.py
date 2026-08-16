#!/usr/bin/env python3
"""Generate the preregistered V64 selection, paired-policy, and adaptive-SBC populations."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from v64_external_eig import load_family, sample_categorical, simulate_step


def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def fingerprint(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def derived_seed(root: int, *parts: object) -> int:
    text = "|".join([str(root), *map(str, parts)])
    return int(hashlib.sha256(text.encode()).hexdigest()[:16], 16)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(canonical_json(row) + "\n" for row in rows))


def source_initial_observation(family, state: int) -> int:
    row = family.model.observation[0, state]
    support = np.flatnonzero(row > 0.0)
    if len(support) != 1 or float(row[support[0]]) != 1.0:
        raise RuntimeError("V64 population generator requires the pinned deterministic observations")
    return int(support[0])


def build_selection(family, config: dict) -> tuple[list[dict], list[dict]]:
    spec = config["selectionPopulation"]
    latent_rng = np.random.default_rng(spec["latentSeed"])
    initial_rng = np.random.default_rng(spec["initialStateSeed"])
    action_rng = np.random.default_rng(spec["behaviorActionSeed"])
    transition_rng = np.random.default_rng(spec["transitionSeed"])
    order_rng = np.random.default_rng(spec["generatorSeed"])
    public_rows: list[dict] = []
    audit_rows: list[dict] = []
    ordinal = 0
    for prefix_length in spec["publicPrefixLengths"]:
        for class_ordinal in range(spec["recordsPerPublicPrefixLength"]):
            identity = int(latent_rng.integers(0, 2))
            theta = float(
                family.theta_support[0]
                + (family.theta_support[1] - family.theta_support[0])
                * latent_rng.beta(2.0, 2.0)
            )
            initial_state = sample_categorical(family.model.initial, float(initial_rng.random()))
            state = initial_state
            initial_observation = source_initial_observation(family, state)
            actions: list[str] = []
            observations: list[str] = []
            states = [state]
            uniforms: list[float] = []
            for _ in range(prefix_length):
                action = family.canonical_actions[int(action_rng.integers(0, 4))]
                uniform = float(transition_rng.random())
                state, observation, _ = simulate_step(
                    family,
                    identity,
                    theta,
                    state,
                    action,
                    uniform,
                    0.5,
                )
                actions.append(family.model.actions[action])
                observations.append(family.model.observations[observation])
                states.append(state)
                uniforms.append(uniform)
            record_id = hashlib.sha256(
                f"v64-selection|{spec['generatorSeed']}|{ordinal}|{class_ordinal}".encode()
            ).hexdigest()[:24]
            public = {
                "record_id": record_id,
                "prefix_length": prefix_length,
                "initial_observation": family.model.observations[initial_observation],
                "actions": actions,
                "observations": observations,
            }
            audit = {
                "record_id": record_id,
                "identity": identity,
                "identity_name": family.identity_names[identity],
                "theta": theta,
                "initial_state": initial_state,
                "states": states,
                "transition_uniforms": uniforms,
                "public_fingerprint": fingerprint(public),
            }
            public_rows.append(public)
            audit_rows.append(audit)
            ordinal += 1
    permutation = order_rng.permutation(len(public_rows))
    public_rows = [public_rows[int(index)] for index in permutation]
    audit_by_id = {row["record_id"]: row for row in audit_rows}
    audit_rows = [audit_by_id[row["record_id"]] for row in public_rows]
    return public_rows, audit_rows


def policy_streams(root: int, scenario_id: str, policies: list[str], budget: int) -> dict:
    result = {}
    for policy in policies:
        transition_rng = np.random.default_rng(
            derived_seed(root, scenario_id, policy, "transition")
        )
        observation_rng = np.random.default_rng(
            derived_seed(root, scenario_id, policy, "observation")
        )
        result[policy] = {
            "transition_uniforms": transition_rng.random(budget).tolist(),
            "observation_uniforms": observation_rng.random(budget).tolist(),
        }
    return result


def build_adaptive(family, config: dict) -> tuple[list[dict], list[dict]]:
    spec = config["pairedAdaptiveEvaluation"]
    latent_rng = np.random.default_rng(spec["pairedEvaluationSeed"])
    initial_rng = np.random.default_rng(derived_seed(spec["pairedEvaluationSeed"], "initial"))
    public_rows: list[dict] = []
    audit_rows: list[dict] = []
    budget = spec["primaryBudget"]
    for replication in range(spec["replications"]):
        identity = replication % 2
        theta = float(
            family.theta_support[0]
            + (family.theta_support[1] - family.theta_support[0])
            * latent_rng.beta(2.0, 2.0)
        )
        initial_state = sample_categorical(family.model.initial, float(initial_rng.random()))
        initial_observation = source_initial_observation(family, initial_state)
        scenario_id = hashlib.sha256(
            f"v64-adaptive|{spec['pairedEvaluationSeed']}|{replication}".encode()
        ).hexdigest()[:24]
        public = {
            "scenario_id": scenario_id,
            "replication": replication,
            "initial_observation": family.model.observations[initial_observation],
        }
        random_action_rng = np.random.default_rng(
            derived_seed(spec["randomActionSeed"], scenario_id, "actions")
        )
        random_actions = [
            family.model.actions[
                family.canonical_actions[int(random_action_rng.integers(0, 4))]
            ]
            for _ in range(budget)
        ]
        audit = {
            "scenario_id": scenario_id,
            "identity": identity,
            "identity_name": family.identity_names[identity],
            "theta": theta,
            "initial_state": initial_state,
            "policy_streams": policy_streams(
                spec["policySeed"], scenario_id, spec["policies"], budget
            ),
            "random_actions": random_actions,
            "public_fingerprint": fingerprint(public),
        }
        public_rows.append(public)
        audit_rows.append(audit)
    return public_rows, audit_rows


def build_sbc(family, config: dict) -> tuple[list[dict], list[dict]]:
    spec = config["adaptiveSBC"]
    public_rows: list[dict] = []
    audit_rows: list[dict] = []
    budget = spec["budget"]
    for replication in range(spec["replications"]):
        latent_rng = np.random.default_rng(
            derived_seed(spec["rankSeed"], replication, "latent")
        )
        identity = int(latent_rng.integers(0, 2))
        theta = float(
            family.theta_support[0]
            + (family.theta_support[1] - family.theta_support[0])
            * latent_rng.beta(2.0, 2.0)
        )
        initial_state = sample_categorical(family.model.initial, float(latent_rng.random()))
        initial_observation = source_initial_observation(family, initial_state)
        scenario_id = hashlib.sha256(
            f"v64-sbc|{spec['rankSeed']}|{replication}".encode()
        ).hexdigest()[:24]
        public = {
            "scenario_id": scenario_id,
            "replication": replication,
            "initial_observation": family.model.observations[initial_observation],
        }
        transition_rng = np.random.default_rng(
            derived_seed(spec["rankSeed"], scenario_id, "transition")
        )
        observation_rng = np.random.default_rng(
            derived_seed(spec["rankSeed"], scenario_id, "observation")
        )
        posterior_draw_rng = np.random.default_rng(
            derived_seed(spec["posteriorDrawSeed"], scenario_id, "posterior")
        )
        audit = {
            "scenario_id": scenario_id,
            "identity": identity,
            "identity_name": family.identity_names[identity],
            "theta": theta,
            "initial_state": initial_state,
            "transition_uniforms": transition_rng.random(budget).tolist(),
            "observation_uniforms": observation_rng.random(budget).tolist(),
            "posterior_draw_uniforms": posterior_draw_rng.random(
                (spec["posteriorDrawsPerReplication"], 3)
            ).tolist(),
            "rank_tie_uniforms": posterior_draw_rng.random(3).tolist(),
            "public_fingerprint": fingerprint(public),
        }
        public_rows.append(public)
        audit_rows.append(audit)
    return public_rows, audit_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", default="configs/v64-implementation-lock.json")
    parser.add_argument("--output", default="data/v64-external-multi-action-eig")
    args = parser.parse_args()
    lock_path = (PROJECT_ROOT / args.lock).resolve()
    output = (PROJECT_ROOT / args.output).resolve()
    if output.exists():
        raise RuntimeError("V64 population directory already exists")
    lock = json.loads(lock_path.read_text())
    if not lock["authorization"]["construct_and_audit_sealed_populations"]:
        raise RuntimeError("V64 implementation lock does not authorize population construction")
    for relative, digest in lock["source_sha256"].items():
        if file_sha256(PROJECT_ROOT / relative) != digest:
            raise RuntimeError(f"frozen V64 implementation changed: {relative}")
    design = json.loads((PROJECT_ROOT / lock["design_lock"]).read_text())
    if file_sha256(PROJECT_ROOT / lock["design_lock"]) != lock["design_lock_sha256"]:
        raise RuntimeError("V64 design lock changed after implementation freeze")
    config = design["config_payload"]
    family = load_family()
    output.mkdir(parents=True)

    selection_public, selection_audit = build_selection(family, config)
    adaptive_public, adaptive_audit = build_adaptive(family, config)
    sbc_public, sbc_audit = build_sbc(family, config)
    files = {
        "selection_public": output / "selection-public.jsonl",
        "selection_audit": output / "selection-audit.jsonl",
        "adaptive_public": output / "adaptive-public.jsonl",
        "adaptive_audit": output / "adaptive-audit.jsonl",
        "sbc_public": output / "sbc-public.jsonl",
        "sbc_audit": output / "sbc-audit.jsonl",
    }
    rows = {
        "selection_public": selection_public,
        "selection_audit": selection_audit,
        "adaptive_public": adaptive_public,
        "adaptive_audit": adaptive_audit,
        "sbc_public": sbc_public,
        "sbc_audit": sbc_audit,
    }
    for name, path in files.items():
        write_jsonl(path, rows[name])
    manifest = {
        "schema_version": 64,
        "experiment": "v64_population_manifest",
        "implementation_lock": str(lock_path.relative_to(PROJECT_ROOT)),
        "implementation_lock_sha256": file_sha256(lock_path),
        "generator": "python/generate_v64_populations.py",
        "generator_sha256": file_sha256(Path(__file__).resolve()),
        "counts": {name: len(value) for name, value in rows.items()},
        "files": {
            name: {
                "path": str(path.relative_to(PROJECT_ROOT)),
                "sha256": file_sha256(path),
            }
            for name, path in files.items()
        },
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
