#!/usr/bin/env python3
from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from evaluate_v65r3_eig import (
    TERMINAL_NAMES,
    atomic_write_json,
    atomic_write_jsonl,
    aggregate_implementation_audit,
    failure_payload,
    reserve_attempt,
)
from evaluate_v65r1_eig import aggregate_evaluation
from test_v65r1_evaluator import (
    synthetic_access,
    synthetic_implementation_audit,
    synthetic_rows,
)
from v22r2_grounding import PROJECT_ROOT


class V65r3EvaluatorTests(unittest.TestCase):
    def test_atomic_json_and_jsonl_leave_no_temporary(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            json_path = root / "value.json"
            jsonl_path = root / "rows.jsonl"
            atomic_write_json(json_path, {"b": 2, "a": 1})
            atomic_write_jsonl(jsonl_path, [{"x": 1}, {"x": 2}])
            self.assertEqual(json.loads(json_path.read_text()), {"a": 1, "b": 2})
            self.assertEqual(len(jsonl_path.read_text().splitlines()), 2)
            self.assertFalse(any(path.name.startswith(".") for path in root.iterdir()))

    def test_attempt_marker_consumes_one_shot(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            marker = {"logical_evaluation_attempt": 1}
            path = reserve_attempt(root, marker)
            self.assertTrue(path.exists())
            self.assertEqual(json.loads(path.read_text()), marker)
            with self.assertRaises(RuntimeError):
                reserve_attempt(root, marker)

    def test_every_terminal_artifact_blocks_attempt(self) -> None:
        for name in TERMINAL_NAMES:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                (root / name).write_text("{}\n")
                with self.assertRaises(RuntimeError):
                    reserve_attempt(root, {"logical_evaluation_attempt": 1})

    def test_failure_payload_is_terminal_and_bound(self) -> None:
        lock_path = PROJECT_ROOT / "configs/v65r3-implementation-lock.json"
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as directory:
            attempt = Path(directory) / "attempt.json"
            atomic_write_json(attempt, {"logical_evaluation_attempt": 1})
            try:
                raise ValueError("synthetic durable failure")
            except ValueError as error:
                payload = failure_payload(
                    lock_path=lock_path,
                    attempt_path=attempt,
                    stage="synthetic_stage",
                    progress={"record_budget_rows_completed": 7},
                    access={"logical_evaluation_attempts": 1},
                    error=error,
                )
            self.assertFalse(payload["passed"])
            self.assertTrue(payload["one_shot_authorization_consumed"])
            self.assertEqual(payload["decision"], "do_not_authorize_reward_planning")
            self.assertEqual(payload["exception"]["type"], "ValueError")
            self.assertIn("synthetic durable failure", payload["exception"]["traceback"])
            self.assertFalse(payload["claim_boundary"]["V65r3_rerun_authorized"])

    def test_inherited_aggregate_passing_fixture(self) -> None:
        design = json.loads((PROJECT_ROOT / "configs/v65r3-design-lock.json").read_text())
        result = aggregate_evaluation(
            synthetic_rows(),
            design["config_payload"],
            synthetic_implementation_audit(),
            synthetic_access(),
        )
        self.assertTrue(result["passed"], result["failed_gates"])
        self.assertEqual(len(result["compute_diagnostics"]["cells"]), 432)

    def test_frozen_implementation_audit_shared_stream_alias(self) -> None:
        lock = json.loads(
            (PROJECT_ROOT / "configs/v65r3-implementation-lock.json").read_text()
        )
        audit = json.loads((PROJECT_ROOT / lock["implementation_audit"]).read_text())
        adapted = aggregate_implementation_audit(audit)
        self.assertTrue(
            adapted["mutation_audit"]["checks"][
                "share_inner_streams_across_outer_particles"
            ]
        )
        self.assertNotIn(
            "share_inner_streams_across_outer_particles",
            audit["mutation_audit"]["checks"],
        )

    def test_inherited_aggregate_rejects_access_and_accuracy_mutants(self) -> None:
        design = json.loads((PROJECT_ROOT / "configs/v65r3-design-lock.json").read_text())
        rows = synthetic_rows()
        access = synthetic_access()
        access["truth_field_access_count"] = 1
        access_result = aggregate_evaluation(
            rows,
            design["config_payload"],
            synthetic_implementation_audit(),
            access,
        )
        mutant = copy.deepcopy(rows)
        for row in mutant:
            if row["budget"] == 509:
                row["absolute_eig_errors"] = [0.03] * 4
        accuracy_result = aggregate_evaluation(
            mutant,
            design["config_payload"],
            synthetic_implementation_audit(),
            synthetic_access(),
        )
        self.assertFalse(access_result["passed"])
        self.assertFalse(accuracy_result["passed"])

    def test_source_reserves_attempt_before_subset_read_and_serializes_failure(self) -> None:
        source = (PROJECT_ROOT / "python/evaluate_v65r3_eig.py").read_text()
        run = source[source.index("def run_evaluation"):source.index("def main()")]
        self.assertLess(run.index("reserve_attempt("), run.index("read_jsonl("))
        self.assertIn('atomic_write_json(output_dir / "failure.json", failure)', run)
        self.assertIn('atomic_write_json(output_dir / "result.json", result)', run)


if __name__ == "__main__":
    unittest.main()
