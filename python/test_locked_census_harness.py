#!/usr/bin/env python3
"""Full-path, outcome-blind tests for the durable locked census harness."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from locked_census_harness import (
    named_structural_resources,
    run_locked_census_once,
)


def evaluated(row: dict) -> dict:
    return {
        "name": row["name"],
        "structural": {"belief_normalizes": True},
        "resource": {"belief_normalization_rate": 1.0},
        "score": row["score"],
    }


class LockedCensusHarnessTests(unittest.TestCase):
    def test_success_persists_raw_fixtures_before_result(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "success"
            result = run_locked_census_once(
                output_dir=output,
                attempt={"attempt_number": 1},
                fixture_rows=[
                    {"name": "alpha", "score": 2},
                    {"name": "beta", "score": 3},
                ],
                evaluate_fixture=evaluated,
                evaluate_gates=lambda rows: {
                    "complete": len(rows) == 2,
                    "positive": all(row["score"] > 0 for row in rows.values()),
                },
                result_metadata={"schema_version": "synthetic-success"},
                pass_decision="pass",
                fail_decision="fail",
            )
            self.assertTrue(result["passed"])
            self.assertTrue((output / "result.json").exists())
            self.assertFalse((output / "failure.json").exists())
            self.assertEqual(len(list((output / "raw-fixtures").glob("*.json"))), 2)

    def test_fixture_failure_is_captured_after_durable_partial_result(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "fixture-failure"

            def fail_second(row: dict) -> dict:
                if row["name"] == "beta":
                    raise RuntimeError("fixture exploded")
                return evaluated(row)

            with self.assertRaisesRegex(RuntimeError, "fixture exploded"):
                run_locked_census_once(
                    output_dir=output,
                    attempt={"attempt_number": 1},
                    fixture_rows=[
                        {"name": "alpha", "score": 2},
                        {"name": "beta", "score": 3},
                    ],
                    evaluate_fixture=fail_second,
                    evaluate_gates=lambda _: {"unused": True},
                    result_metadata={"schema_version": "synthetic-failure"},
                    pass_decision="pass",
                    fail_decision="fail",
                )
            failure = json.loads((output / "failure.json").read_text())
            self.assertEqual(failure["stage"], "fixture_evaluation")
            self.assertEqual(failure["completed_fixture_names"], ["alpha"])
            self.assertTrue((output / "raw-fixtures/000-alpha.json").exists())

    def test_gate_failure_preserves_every_fixture_and_traceback(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "gate-failure"

            def broken_gate(_: dict) -> dict[str, bool]:
                raise KeyError("missing-name")

            with self.assertRaises(KeyError):
                run_locked_census_once(
                    output_dir=output,
                    attempt={"attempt_number": 1},
                    fixture_rows=[
                        {"name": "alpha", "score": 2},
                        {"name": "beta", "score": 3},
                    ],
                    evaluate_fixture=evaluated,
                    evaluate_gates=broken_gate,
                    result_metadata={"schema_version": "synthetic-gate-failure"},
                    pass_decision="pass",
                    fail_decision="fail",
                )
            failure = json.loads((output / "failure.json").read_text())
            self.assertEqual(failure["stage"], "gate_aggregation")
            self.assertEqual(failure["completed_fixture_count"], 2)
            self.assertIn("KeyError", failure["traceback"])
            self.assertEqual(len(list((output / "raw-fixtures").glob("*.json"))), 2)

    def test_rejects_nonboolean_gates_and_existing_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "bad-gate"
            with self.assertRaisesRegex(TypeError, "boolean mapping"):
                run_locked_census_once(
                    output_dir=output,
                    attempt={"attempt_number": 1},
                    fixture_rows=[{"name": "alpha", "score": 2}],
                    evaluate_fixture=evaluated,
                    evaluate_gates=lambda _: {"bad": 1},
                    result_metadata={"schema_version": "synthetic-bad-gate"},
                    pass_decision="pass",
                    fail_decision="fail",
                )
            with self.assertRaisesRegex(RuntimeError, "already exists"):
                run_locked_census_once(
                    output_dir=output,
                    attempt={"attempt_number": 2},
                    fixture_rows=[],
                    evaluate_fixture=evaluated,
                    evaluate_gates=lambda _: {"unused": True},
                    result_metadata={},
                    pass_decision="pass",
                    fail_decision="fail",
                )

    def test_named_structural_resources_retains_identity(self) -> None:
        fixtures = {
            "alpha": evaluated({"name": "alpha", "score": 2}),
            "beta": evaluated({"name": "beta", "score": 3}),
        }
        rows = named_structural_resources(fixtures)
        self.assertEqual([row["name"] for row in rows], ["alpha", "beta"])
        self.assertTrue(all(row["structural"]["belief_normalizes"] for row in rows))


if __name__ == "__main__":
    unittest.main()
