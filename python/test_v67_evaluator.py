#!/usr/bin/env python3
"""Synthetic tests for the durable V67 evaluator."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
import unittest

from test_v67_verification import fixture_model, policy_tree
from evaluate_v67_verification import (
    REQUIRED_FILES,
    reserve_attempt,
    validate_manifest_row_files,
    verify_policy_directory,
)
from v10_protocol import file_sha256
from v67_verification import (
    compile_policy_dtmc,
    condition_public_history,
    construct_independent_family,
    dtmc_statistics,
    policy_tree_hash,
    write_explicit_dtmc,
)


def make_synthetic_bundle(directory: Path) -> tuple[Path, dict]:
    family = construct_independent_family(fixture_model(), nodes=5)
    record = {
        "record_id": "synthetic", "prefix_length": 0,
        "initial_observation": "only", "actions": [], "observations": [],
    }
    belief, _ = condition_public_history(family, record)
    policy = policy_tree(3, 1)
    model, checks = compile_policy_dtmc(family, belief, policy)
    stats = dtmc_statistics(model)
    write_explicit_dtmc(model, directory)
    (directory / "policy-tree.json").write_text(
        json.dumps(policy, indent=2, sort_keys=True) + "\n"
    )
    meta = {
        "policy_tree_hash": policy_tree_hash(policy),
        "exact_root_belief_mass": 1.0,
        "frozen_V66_exact_environment_value": stats["expected_return"],
        "independent_scalar_value": stats["expected_return"],
        "compiled_direct_value": stats["expected_return"],
        "compiler_checks": checks,
    }
    (directory / "model.meta.json").write_text(
        json.dumps(meta, indent=2, sort_keys=True) + "\n"
    )
    row = {
        "policy_id": "synthetic",
        "record_index": 0,
        "record_id": "synthetic",
        "prefix_length": 0,
        "policy_kind": "exact_policy",
        "policy_tree_hash": policy_tree_hash(policy),
        "files": {
            name: {"sha256": file_sha256(directory / name), "size": (directory / name).stat().st_size}
            for name in REQUIRED_FILES
        },
    }
    return directory, row


class V67EvaluatorTests(unittest.TestCase):
    def test_synthetic_external_verification(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory, row = make_synthetic_bundle(Path(temporary))
            result = verify_policy_directory(directory, row)
        self.assertAlmostEqual(result["Storm_termination_probability"], 1.0, places=13)
        self.assertLess(result["absolute_Storm_return_error_against_independent"], 1e-12)

    def test_file_hash_mutation_is_detected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory, row = make_synthetic_bundle(Path(temporary))
            row["files"]["model.tra"]["sha256"] = "0" * 64
            self.assertEqual(validate_manifest_row_files(directory, row), 1)

    def test_file_size_mutation_is_detected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory, row = make_synthetic_bundle(Path(temporary))
            row["files"]["model.rew"]["size"] += 1
            self.assertEqual(validate_manifest_row_files(directory, row), 1)

    def test_policy_hash_mutation_is_rejected_before_Storm(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory, row = make_synthetic_bundle(Path(temporary))
            row["policy_tree_hash"] = "0" * 64
            with self.assertRaises(RuntimeError):
                verify_policy_directory(directory, row)

    def test_attempt_reservation_is_exclusive(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "attempt.json"
            reserve_attempt(path, {"attempt": 1})
            with self.assertRaises(FileExistsError):
                reserve_attempt(path, {"attempt": 2})

    def test_missing_file_is_detected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory, row = make_synthetic_bundle(Path(temporary))
            (directory / "model.lab").unlink()
            self.assertEqual(validate_manifest_row_files(directory, row), 1)

    def test_all_five_required_files_are_bound(self) -> None:
        self.assertEqual(len(REQUIRED_FILES), 5)
        self.assertEqual(set(REQUIRED_FILES), {
            "model.tra", "model.lab", "model.rew", "model.meta.json", "policy-tree.json"
        })

    def test_nonfinite_source_value_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory, row = make_synthetic_bundle(Path(temporary))
            meta_path = directory / "model.meta.json"
            meta = json.loads(meta_path.read_text())
            meta["independent_scalar_value"] = float("inf")
            meta_path.write_text(json.dumps(meta, allow_nan=True) + "\n")
            row["files"]["model.meta.json"] = {
                "sha256": file_sha256(meta_path), "size": meta_path.stat().st_size
            }
            result = verify_policy_directory(directory, row)
            self.assertFalse(result["finite"])


if __name__ == "__main__":
    unittest.main()
