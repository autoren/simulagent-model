#!/usr/bin/env python3
"""Independently audit the V62 external source bundle and parsed arrays."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT


def independent_parse(path: Path) -> dict[str, object]:
    lines = []
    for raw in path.read_text().splitlines():
        line = raw.partition("#")[0].strip()
        if line:
            lines.append(line)
    metadata: dict[str, object] = {}
    for line in lines:
        head, separator, tail = line.partition(":")
        if separator and head in {"discount", "values", "states", "actions", "observations"}:
            tokens = tail.split()
            if head == "discount":
                metadata[head] = float(tokens[0])
            elif head == "values":
                metadata[head] = tokens[0]
            elif len(tokens) == 1 and tokens[0].isdigit():
                metadata[head] = tuple(str(index) for index in range(int(tokens[0])))
            else:
                metadata[head] = tuple(tokens)
    states = metadata["states"]
    actions = metadata["actions"]
    observations = metadata["observations"]
    s_count, a_count, o_count = len(states), len(actions), len(observations)
    transition = np.zeros((a_count, s_count, s_count), dtype=np.float64)
    observation = np.zeros((a_count, s_count, o_count), dtype=np.float64)
    reward = np.zeros((a_count, s_count, s_count), dtype=np.float64)
    initial = None

    def ids(names, token):
        return range(len(names)) if token == "*" else (names.index(token),)

    cursor = 0
    while cursor < len(lines):
        line = lines[cursor]
        if line.startswith("start:"):
            remainder = line.partition(":")[2].strip()
            if remainder:
                values = remainder.split()
            else:
                cursor += 1
                values = lines[cursor].split()
            initial = np.asarray([float(value) for value in values])
        elif line.startswith("T:"):
            action_token = line.partition(":")[2].strip()
            matrix = np.asarray(
                [
                    [float(value) for value in lines[cursor + offset + 1].split()]
                    for offset in range(s_count)
                ]
            )
            for action in ids(actions, action_token):
                transition[action] = matrix
            cursor += s_count
        elif line.startswith("O:"):
            action_token = line.partition(":")[2].strip()
            matrix = np.asarray(
                [
                    [float(value) for value in lines[cursor + offset + 1].split()]
                    for offset in range(s_count)
                ]
            )
            for action in ids(actions, action_token):
                observation[action] = matrix
            cursor += s_count
        elif line.startswith("R:"):
            fields = [field.strip() for field in line.split(":")]
            final = fields[4].split()
            if final[0] != "*":
                raise ValueError("unexpected observation-conditioned reward")
            value = float(final[1])
            for action in ids(actions, fields[1]):
                for state in ids(states, fields[2]):
                    for successor in ids(states, fields[3]):
                        reward[action, state, successor] = value
        cursor += 1
    if initial is None:
        initial = np.full(s_count, 1.0 / s_count)
    return {
        "states": states,
        "actions": actions,
        "observations": observations,
        "discount": metadata["discount"],
        "initial": initial,
        "transition": transition,
        "observation": observation,
        "reward": reward,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", default="data/v62-external-pomdp-transfer/bundle")
    parser.add_argument(
        "--output", default="outputs/v62-external-pomdp-transfer/bundle-audit.json"
    )
    args = parser.parse_args()
    bundle = (PROJECT_ROOT / args.bundle).resolve()
    output = (PROJECT_ROOT / args.output).resolve()
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    implementation_path = PROJECT_ROOT / manifest["implementation_lock"]
    implementation = json.loads(implementation_path.read_text())
    design_path = PROJECT_ROOT / implementation["design_lock"]
    design = json.loads(design_path.read_text())
    config = design["config_payload"]
    external = config["externalSource"]
    errors: list[str] = []

    lock_ok = (
        file_sha256(implementation_path) == manifest["implementation_lock_sha256"]
        and file_sha256(design_path) == manifest["design_lock_sha256"]
        and implementation["authorization"]["build_and_audit_one_external_source_bundle"]
    )
    if not lock_ok:
        errors.append("bundle is not bound to the frozen implementation and design")

    source_hashes = {
        path: file_sha256(bundle / "source" / path) for path in external["files"]
    }
    source_ok = (
        source_hashes == external["files"]
        and file_sha256(bundle / "source/LICENSE") == external["licenseSha256"]
        and manifest["external_commit"] == external["commit"]
    )
    if not source_ok:
        errors.append("external source, commit, or license binding failed")

    manifest_hash_mismatches = 0
    for relative, binding in manifest["files"].items():
        path = bundle / relative
        if (
            not path.is_file()
            or file_sha256(path) != binding["sha256"]
            or path.stat().st_size != binding["bytes"]
        ):
            manifest_hash_mismatches += 1
    unmanifested = {
        str(path.relative_to(bundle))
        for path in bundle.rglob("*")
        if path.is_file() and path != manifest_path
    } - set(manifest["files"])
    if manifest_hash_mismatches or unmanifested:
        errors.append("bundle manifest hash or file census failed")

    array_errors = {
        "transition": 0.0,
        "observation": 0.0,
        "reward": 0.0,
        "initial": 0.0,
        "discount": 0.0,
    }
    parser_agreements = []
    normalization = []
    for entry in config["benchmark"]["models"]:
        model_id = entry["id"]
        source = bundle / "source" / f"pobax/envs/classic/POMDP/{model_id}.POMDP"
        independent = independent_parse(source)
        candidate = json.loads((bundle / f"models/{model_id}/model.json").read_text())
        parser_agreements.append(
            tuple(candidate["states"]) == independent["states"]
            and tuple(candidate["actions"]) == independent["actions"]
            and tuple(candidate["observations"]) == independent["observations"]
        )
        for field in ("transition", "observation", "reward", "initial"):
            error = float(
                np.max(np.abs(np.asarray(candidate[field]) - independent[field]))
            )
            array_errors[field] = max(array_errors[field], error)
        array_errors["discount"] = max(
            array_errors["discount"],
            abs(float(candidate["discount"]) - float(independent["discount"])),
        )
        transition = np.asarray(candidate["transition"])
        observation = np.asarray(candidate["observation"])
        initial = np.asarray(candidate["initial"])
        normalization.append(
            bool(
                np.allclose(transition.sum(axis=2), 1.0, atol=1e-12, rtol=0.0)
                and np.allclose(observation.sum(axis=2), 1.0, atol=1e-12, rtol=0.0)
                and np.isclose(initial.sum(), 1.0, atol=1e-12, rtol=0.0)
            )
        )
    parser_agreement_rate = sum(parser_agreements) / len(parser_agreements)
    if parser_agreement_rate != 1.0 or max(array_errors.values()) > 1e-12:
        errors.append("candidate and independent external parsers disagree")
    if not all(normalization):
        errors.append("one or more external models is not normalized")

    candidate_evaluation_absent = (
        manifest["candidate_evaluations"] == 0
        and not (PROJECT_ROOT / "outputs/v62-external-pomdp-transfer/evaluation-attempt.json").exists()
        and not (PROJECT_ROOT / "outputs/v62-external-pomdp-transfer/evaluation/result.json").exists()
    )
    if not candidate_evaluation_absent:
        errors.append("candidate evaluation occurred before bundle seal")

    result = {
        "schema_version": 62,
        "experiment": "v62_external_bundle_audit",
        "passed": not errors,
        "decision": "seal_v62_external_bundle" if not errors else "repair_v62_external_bundle",
        "errors": errors,
        "bundle": str(bundle.relative_to(PROJECT_ROOT)),
        "manifest": str(manifest_path.relative_to(PROJECT_ROOT)),
        "manifest_sha256": file_sha256(manifest_path),
        "implementation_lock": manifest["implementation_lock"],
        "implementation_lock_sha256": manifest["implementation_lock_sha256"],
        "source_files_sha256": source_hashes,
        "license_sha256": file_sha256(bundle / "source/LICENSE"),
        "manifest_hash_mismatch_count": manifest_hash_mismatches,
        "unmanifested_file_count": len(unmanifested),
        "independent_parser_agreement_rate": parser_agreement_rate,
        "maximum_array_errors": array_errors,
        "model_normalization_rate": sum(normalization) / len(normalization),
        "checks": {
            "frozen_lock_binding": lock_ok,
            "external_source_commit_and_license_binding": source_ok,
            "complete_hash_size_manifest": manifest_hash_mismatches == 0 and not unmanifested,
            "independent_parser_exact_agreement": parser_agreement_rate == 1.0 and max(array_errors.values()) <= 1e-12,
            "all_external_models_normalized": all(normalization),
            "candidate_evaluation_absent": candidate_evaluation_absent,
        },
        "data_access": {
            "external_model_definition_files_read": 3,
            "external_candidate_evaluations": 0,
            "human_authored_v58_records": 0,
            "model_forward_passes": 0,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
