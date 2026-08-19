#!/usr/bin/env python3
"""Durable one-shot census harness for prospectively locked experiments."""
from __future__ import annotations

import json
import traceback
from pathlib import Path
from typing import Any, Callable, Sequence


JsonObject = dict[str, Any]


def write_json(path: Path, value: JsonObject) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def named_structural_resources(fixtures: JsonObject) -> list[JsonObject]:
    """Keep fixture identity attached to structural/resource subrecords."""
    return [
        {
            "name": name,
            "structural": row["structural"],
            "resource": row["resource"],
        }
        for name, row in fixtures.items()
    ]


def run_locked_census_once(
    *,
    output_dir: Path,
    attempt: JsonObject,
    fixture_rows: Sequence[JsonObject],
    evaluate_fixture: Callable[[JsonObject], JsonObject],
    evaluate_gates: Callable[[JsonObject], dict[str, bool]],
    result_metadata: JsonObject,
    pass_decision: str,
    fail_decision: str,
) -> JsonObject:
    """Run once while durably preserving each completed fixture and any error."""
    if output_dir.exists():
        raise RuntimeError(f"locked census output already exists: {output_dir}")
    output_dir.mkdir(parents=True)
    write_json(output_dir / "attempt.json", attempt)
    fixtures: JsonObject = {}
    stage = "fixture_evaluation"
    active_fixture: str | None = None
    try:
        for ordinal, fixture_row in enumerate(fixture_rows):
            active_fixture = str(fixture_row["name"])
            evaluated = evaluate_fixture(fixture_row)
            if evaluated.get("name") != active_fixture:
                raise ValueError("fixture evaluator changed or omitted the registered name")
            fixtures[active_fixture] = evaluated
            write_json(
                output_dir / "raw-fixtures" / f"{ordinal:03d}-{active_fixture}.json",
                evaluated,
            )
        stage = "gate_aggregation"
        active_fixture = None
        gates = evaluate_gates(fixtures)
        if not gates or any(not isinstance(value, bool) for value in gates.values()):
            raise TypeError("gate evaluator must return a nonempty boolean mapping")
        passed = all(gates.values())
        result = {
            **result_metadata,
            "passed": passed,
            "decision": pass_decision if passed else fail_decision,
            "gates": gates,
            "fixtures": fixtures,
            "attempt": attempt,
        }
        write_json(output_dir / "result.json", result)
        return result
    except Exception as error:
        failure = {
            "schema_version": "locked-census-execution-failure",
            "status": "execution_failure",
            "stage": stage,
            "active_fixture": active_fixture,
            "completed_fixture_names": list(fixtures),
            "completed_fixture_count": len(fixtures),
            "raw_fixture_artifacts_preserved": len(fixtures),
            "exception_type": type(error).__name__,
            "exception_message": str(error),
            "traceback": traceback.format_exc(),
            "result_artifact_written": False,
            "attempt": attempt,
        }
        write_json(output_dir / "failure.json", failure)
        raise
