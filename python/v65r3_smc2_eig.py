#!/usr/bin/env python3
"""V65r3 synthetic-only implementation scoring firewall over the V65r2 algorithm."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import v65r2_smc2_eig as base
from v22r2_grounding import PROJECT_ROOT


ImpossiblePublicHistory = base.ImpossiblePublicHistory
ParticleExtinctionWithPositiveSupport = base.ParticleExtinctionWithPositiveSupport
attempted_outcome_leak = base.attempted_outcome_leak
boolean_identity_support = base.boolean_identity_support
canonicalize_atoms = base.canonicalize_atoms
classify_particle_extinction = base.classify_particle_extinction
collapse_map_identity = base.collapse_map_identity
collapse_theta_mean = base.collapse_theta_mean
force_equal_identity_evidence = base.force_equal_identity_evidence
normalize_identity_log_evidence = base.normalize_identity_log_evidence
pool_repeats = base.pool_repeats
posterior_summary = base.posterior_summary
rao_blackwellize_measure = base.rao_blackwellize_measure
score_action = base.score_action
score_all_actions = base.score_all_actions
score_state_as_target = base.score_state_as_target
select_action = base.select_action
smc2_inference = base.smc2_inference
stable_seed = base.stable_seed


PUBLIC_FIELDS = (
    "record_id",
    "prefix_length",
    "initial_observation",
    "actions",
    "observations",
)


def load_config(path: str | Path = "configs/v65r3-design-lock.json") -> dict[str, Any]:
    value = Path(path)
    if not value.is_absolute():
        value = PROJECT_ROOT / value
    return json.loads(value.read_text())["config_payload"]


def _history_key(record: dict[str, Any]) -> str:
    if set(record) != set(PUBLIC_FIELDS):
        raise ValueError("V65r3 scoring-firewall record must contain exactly the public fields")
    return json.dumps(
        {
            "prefix_length": int(record["prefix_length"]),
            "initial_observation": record["initial_observation"],
            "actions": record["actions"],
            "observations": record["observations"],
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def assert_synthetic_implementation_fixture(
    record: dict[str, Any], sealed_records: Sequence[dict[str, Any]]
) -> None:
    record_id = str(record["record_id"])
    history = _history_key(record)
    sealed_ids = {str(row["record_id"]) for row in sealed_records}
    sealed_histories = {_history_key(row) for row in sealed_records}
    if record_id in sealed_ids:
        raise PermissionError("V65r3 implementation scorer rejected a sealed record ID")
    if history in sealed_histories:
        raise PermissionError("V65r3 implementation scorer rejected a sealed public history")


def score_all_actions_for_implementation_fixture(
    family,
    measure: dict[str, Any] | Sequence[dict[str, Any]],
    record: dict[str, Any],
    sealed_records: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    """The sole candidate-scoring entry point authorized during V65r3 implementation audit."""
    assert_synthetic_implementation_fixture(record, sealed_records)
    return base.score_all_actions(family, measure)
