#!/usr/bin/env python3
"""Construct V63 public records and private truth sidecars without running inference."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any

from v22r2_grounding import PROJECT_ROOT


def stable_seed(base_seed: int, *parts: Any) -> int:
    payload = json.dumps([base_seed, *parts], sort_keys=True, separators=(",", ":"))
    return int.from_bytes(hashlib.sha256(payload.encode()).digest()[:8], "big")


def scaled_beta(seed: int, low: float, high: float, alpha: float, beta: float) -> float:
    unit = random.Random(seed).betavariate(alpha, beta)
    return low + (high - low) * unit


def simulate(identity: int, theta: float, length: int, seed: int) -> tuple[list[int], list[int]]:
    rng = random.Random(seed)
    side = 0 if rng.random() < 0.5 else 1
    reports, states = [], []
    same_probability = theta if identity == 0 else 1.0 - theta
    for _ in range(length):
        if rng.random() >= same_probability:
            side = 1 - side
        accurate = rng.random() < 0.85
        reported_side = side if accurate else 1 - side
        reports.append(1 + reported_side)
        states.append(2 + side)
    return reports, states


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def make_record(
    population: str,
    ordinal: int,
    identity: int,
    theta: float,
    support_lengths: list[int],
    current_length: int | None,
    trajectory_seed: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    record_id = f"v63-{population}-{ordinal:04d}"
    public_episodes, truth_episodes = [], []
    lengths = list(support_lengths)
    roles = ["support"] * len(support_lengths)
    if current_length is not None:
        lengths.append(current_length)
        roles.append("current")
    for episode_ordinal, (length, role) in enumerate(zip(lengths, roles, strict=True)):
        seed = stable_seed(trajectory_seed, population, ordinal, episode_ordinal)
        observations, states = simulate(identity, theta, length, seed)
        public_episodes.append({
            "id": f"{record_id}-episode-{episode_ordinal}",
            "role": role,
            "observations": observations,
        })
        truth_episodes.append({
            "episode_ordinal": episode_ordinal,
            "role": role,
            "trajectory_seed": seed,
            "states": states,
        })
    public = {"id": record_id, "population": population, "episodes": public_episodes}
    truth = {
        "id": record_id,
        "population": population,
        "identity": identity,
        "identity_name": ["persistent", "alternating"][identity],
        "theta": theta,
        "current_state": truth_episodes[-1]["states"][-1],
        "episodes": truth_episodes,
    }
    return public, truth


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--implementation-lock", default="configs/v63-implementation-lock.json")
    parser.add_argument(
        "--output-root", default="data/v63-external-unknown-dynamics/sealed-populations"
    )
    args = parser.parse_args()
    lock_path = (PROJECT_ROOT / args.implementation_lock).resolve()
    lock = json.loads(lock_path.read_text())
    if not lock["authorization"]["construct_and_audit_sealed_populations"]:
        raise RuntimeError("V63 implementation lock does not authorize population construction")
    config = json.loads((PROJECT_ROOT / lock["design_lock"]).read_text())["config_payload"]
    output = (PROJECT_ROOT / args.output_root).resolve()
    if output.exists():
        raise RuntimeError("V63 sealed population directory already exists")
    output.mkdir(parents=True)
    parameter = config["unknownDynamicsFamily"]["continuousParameter"]
    low, high = map(float, parameter["support"])
    alpha, beta = float(parameter["alpha"]), float(parameter["beta"])
    identity_seed = int(config["population"]["identitySeed"])
    theta_seed = int(config["population"]["thetaSeed"])
    trajectory_seed = int(config["population"]["trajectorySeed"])

    exact_identities = [0] * 16 + [1] * 16
    random.Random(identity_seed).shuffle(exact_identities)
    exact_public, exact_truth = [], []
    for ordinal, identity in enumerate(exact_identities):
        theta = scaled_beta(
            stable_seed(theta_seed, "exact", ordinal), low, high, alpha, beta
        )
        support_lengths = [8, 12, 8, 12] if ordinal % 2 == 0 else [12, 8, 12, 8]
        current_length = [6, 10][ordinal % 2]
        public, truth = make_record(
            "exact", ordinal, identity, theta, support_lengths, current_length, trajectory_seed
        )
        exact_public.append(public)
        exact_truth.append(truth)

    sbc_public, sbc_truth = [], []
    for ordinal in range(int(config["simulationBasedCalibration"]["replications"])):
        identity = random.Random(stable_seed(identity_seed, "sbc", ordinal)).randrange(2)
        theta = scaled_beta(stable_seed(theta_seed, "sbc", ordinal), low, high, alpha, beta)
        support_lengths = [8, 12, 8, 12] if ordinal % 2 == 0 else [12, 8, 12, 8]
        current_length = [6, 10][ordinal % 2]
        public, truth = make_record(
            "sbc", ordinal, identity, theta, support_lengths, current_length, trajectory_seed
        )
        sbc_public.append(public)
        sbc_truth.append(truth)

    scale_public, scale_truth = [], []
    episode_counts = config["scaleStress"]["episodeCounts"]
    lengths = config["scaleStress"]["sequenceLengths"]
    for ordinal in range(int(config["scaleStress"]["records"])):
        identity = ordinal % 2
        theta = scaled_beta(stable_seed(theta_seed, "scale", ordinal), low, high, alpha, beta)
        episode_count = int(episode_counts[ordinal % len(episode_counts)])
        sequence_length = int(lengths[ordinal % len(lengths)])
        public, truth = make_record(
            "scale", ordinal, identity, theta,
            [sequence_length] * episode_count, None, trajectory_seed,
        )
        scale_public.append(public)
        scale_truth.append(truth)

    for name, rows in (
        ("exact-public.jsonl", exact_public),
        ("exact-truth.jsonl", exact_truth),
        ("sbc-public.jsonl", sbc_public),
        ("sbc-truth.jsonl", sbc_truth),
        ("scale-public.jsonl", scale_public),
        ("scale-truth.jsonl", scale_truth),
    ):
        write_jsonl(output / name, rows)
    construction = {
        "schema_version": 63,
        "experiment": "v63_population_construction",
        "implementation_lock": str(lock_path.relative_to(PROJECT_ROOT)),
        "counts": {"exact": len(exact_public), "sbc": len(sbc_public), "scale": len(scale_public)},
        "candidate_inference_runs": 0,
        "human_record_access_count": 0,
        "simulated_human_record_count": 0,
        "model_forward_pass_count": 0,
    }
    (output / "construction.json").write_text(
        json.dumps(construction, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(construction, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
