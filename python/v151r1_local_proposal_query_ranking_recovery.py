from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from v151_local_proposal_query_ranking import parse_proposal


def derive_partition(
    public_rows: list[dict[str, Any]],
    persisted_paths: list[Path],
    prior_access: dict[str, Any],
    recovery_config: dict[str, Any],
) -> dict[str, Any]:
    ordered = [row["fixture_id"] for row in public_rows]
    persisted = []
    ordinals = []
    for path in sorted(persisted_paths):
        prefix, fixture_with_suffix = path.name.split("-", 1)
        ordinals.append(int(prefix))
        persisted.append(fixture_with_suffix.removesuffix(".json"))
    expected_count = recovery_config["interruption"]["requiredPersistedFixtureCount"]
    if ordinals != list(range(expected_count)) or persisted != ordered[:expected_count]:
        raise ValueError("V151 persisted fixture prefix is not exact")
    attempted = prior_access["model_generation_count"]
    if attempted != expected_count + 1:
        raise ValueError("V151 interrupted attempt count is not exact")
    return {
        "persisted_fixture_ids": persisted,
        "interrupted_fixture_id": ordered[expected_count],
        "never_started_fixture_ids": ordered[attempted:],
        "all_fixture_ids": ordered,
    }


def recovery_evaluation_config(
    base_config: dict[str, Any],
    recovery_config: dict[str, Any],
) -> dict[str, Any]:
    value = deepcopy(base_config)
    value["accessGates"] = deepcopy(recovery_config["recoveryAccessGates"])
    value["decisionRule"]["ifEveryQualificationAndAccessGatePasses"] = recovery_config["decisionRule"][
        "ifEveryOriginalQualificationAndRecoveryAccessGatePasses"
    ]
    value["decisionRule"]["otherwise"] = recovery_config["decisionRule"]["otherwise"]
    return value


def interrupted_fail_closed(
    fixture_id: str,
    catalog: dict[str, Any],
    base_config: dict[str, Any],
) -> dict[str, Any]:
    output = parse_proposal("", catalog, base_config)
    output["validation_reason"] = "technical_interruption_no_persisted_response"
    return {
        "name": fixture_id,
        "fixture_id": fixture_id,
        **output,
        "raw_response_sha256": None,
        "prompt_sha256": None,
        "prompt_token_count": 0,
        "generated_token_count": 0,
        "maximum_new_tokens_hit": False,
        "generation_seconds": 0.0,
        "raw_response_persisted": False,
        "technical_interruption_without_response": True,
    }


__all__ = ["derive_partition", "interrupted_fail_closed", "recovery_evaluation_config"]
