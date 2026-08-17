#!/usr/bin/env python3
"""Strict Cassandra-POMDP parser for the pinned V68 source inventory.

This extends the deliberately narrow V62 parser without changing the frozen
V62/V67 implementation.  The supported grammar is exactly the union used by
the POBAX ``envs/classic/POMDP`` files at the pinned source commit: full
matrices, sparse scalar entries, row vectors, ``identity``/``uniform``
keywords, wildcards, numeric or symbolic labels, and observation-independent
scalar rewards.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from v62_external_pomdp import POMDPModel


HEADER_KEYS = ("discount", "values", "states", "actions", "observations")


def _clean_lines(text: str) -> list[str]:
    lines: list[str] = []
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip()
        if line:
            lines.append(line)
    return lines


def _names(tokens: list[str], *, field: str) -> tuple[str, ...]:
    if len(tokens) == 1 and tokens[0].isdigit():
        count = int(tokens[0])
        if count <= 0:
            raise ValueError(f"{field} count must be positive")
        return tuple(str(index) for index in range(count))
    if not tokens or len(tokens) != len(set(tokens)):
        raise ValueError(f"{field} labels must be nonempty and unique")
    return tuple(tokens)


def _indices(names: tuple[str, ...], token: str, *, field: str) -> tuple[int, ...]:
    if token == "*":
        return tuple(range(len(names)))
    try:
        return (names.index(token),)
    except ValueError as exc:
        raise ValueError(f"unknown {field} label {token!r}") from exc


def _next_values(lines: list[str], index: int, *, context: str) -> tuple[list[str], int]:
    if index + 1 >= len(lines):
        raise ValueError(f"missing values after {context}")
    return lines[index + 1].split(), index + 2


def _scalar_from_tail_or_next(
    tail: list[str], lines: list[str], index: int, *, context: str
) -> tuple[float, int]:
    if len(tail) == 2:
        try:
            return float(tail[1]), index + 1
        except ValueError as exc:
            raise ValueError(f"invalid scalar in {context}") from exc
    if len(tail) != 1:
        raise ValueError(f"malformed scalar entry in {context}")
    values, next_index = _next_values(lines, index, context=context)
    if len(values) != 1:
        raise ValueError(f"{context} requires exactly one scalar")
    try:
        return float(values[0]), next_index
    except ValueError as exc:
        raise ValueError(f"invalid scalar in {context}") from exc


def _float_vector(tokens: list[str], length: int, *, context: str) -> np.ndarray:
    if len(tokens) != length:
        raise ValueError(f"{context} has length {len(tokens)}; expected {length}")
    try:
        values = np.asarray([float(token) for token in tokens], dtype=np.float64)
    except ValueError as exc:
        raise ValueError(f"non-numeric value in {context}") from exc
    if not np.isfinite(values).all():
        raise ValueError(f"non-finite value in {context}")
    return values


def _parse_headers(lines: list[str]) -> tuple[
    float, tuple[str, ...], tuple[str, ...], tuple[str, ...]
]:
    headers: dict[str, list[str]] = {}
    for line in lines:
        for key in HEADER_KEYS:
            prefix = f"{key}:"
            if line.startswith(prefix):
                if key in headers:
                    raise ValueError(f"duplicate {key} header")
                headers[key] = line[len(prefix):].strip().split()
                break
    missing = set(HEADER_KEYS) - set(headers)
    if missing:
        raise ValueError(f"missing headers: {sorted(missing)}")
    if headers["values"] != ["reward"]:
        raise ValueError("only reward-valued POMDP files are supported")
    if len(headers["discount"]) != 1:
        raise ValueError("discount header requires one scalar")
    discount = float(headers["discount"][0])
    states = _names(headers["states"], field="state")
    actions = _names(headers["actions"], field="action")
    observations = _names(headers["observations"], field="observation")
    return discount, states, actions, observations


def parse_cassandra_pomdp_text(text: str, *, name: str = "model") -> POMDPModel:
    """Parse the strict POBAX-source subset of the Cassandra POMDP grammar."""
    lines = _clean_lines(text)
    discount, states, actions, observations = _parse_headers(lines)
    state_count = len(states)
    action_count = len(actions)
    observation_count = len(observations)
    transition = np.zeros((action_count, state_count, state_count), dtype=np.float64)
    observation = np.zeros((action_count, state_count, observation_count), dtype=np.float64)
    reward = np.zeros((action_count, state_count, state_count), dtype=np.float64)
    initial: np.ndarray | None = None

    index = 0
    while index < len(lines):
        line = lines[index]
        if any(line.startswith(f"{key}:") for key in HEADER_KEYS):
            index += 1
            continue

        if line.startswith("start:"):
            if initial is not None:
                raise ValueError("duplicate start directive")
            inline = line[len("start:"):].strip().split()
            if not inline:
                inline, index = _next_values(lines, index, context="start")
            else:
                index += 1
            if inline == ["uniform"]:
                initial = np.full(state_count, 1.0 / state_count, dtype=np.float64)
            elif inline and inline[0] in {"include", "exclude"}:
                raise ValueError("start include/exclude syntax is outside the frozen V68 subset")
            else:
                initial = _float_vector(inline, state_count, context="start vector")
            continue

        if line.startswith("T:"):
            parts = [part.strip() for part in line.split(":")]
            action_indices = _indices(actions, parts[1], field="action") if len(parts) >= 2 else ()
            if len(parts) == 2:
                values, next_index = _next_values(lines, index, context=line)
                if values == ["identity"]:
                    matrix = np.eye(state_count, dtype=np.float64)
                    index = next_index
                elif values == ["uniform"]:
                    matrix = np.full(
                        (state_count, state_count), 1.0 / state_count, dtype=np.float64
                    )
                    index = next_index
                else:
                    rows = [_float_vector(values, state_count, context=f"{line} row 0")]
                    for row_index in range(1, state_count):
                        values, next_index = _next_values(
                            lines, next_index - 1, context=f"{line} row {row_index}"
                        )
                        rows.append(
                            _float_vector(values, state_count, context=f"{line} row {row_index}")
                        )
                    matrix = np.stack(rows)
                    index = next_index
                for action in action_indices:
                    transition[action] = matrix
                continue
            if len(parts) == 3:
                state_indices = _indices(states, parts[2], field="state")
                values, index = _next_values(lines, index, context=line)
                row = _float_vector(values, state_count, context=f"{line} row")
                for action in action_indices:
                    for state in state_indices:
                        transition[action, state] = row
                continue
            if len(parts) == 4:
                tail = parts[3].split()
                if not tail:
                    raise ValueError(f"missing successor in {line}")
                state_indices = _indices(states, parts[2], field="state")
                successor_indices = _indices(states, tail[0], field="successor state")
                value, index = _scalar_from_tail_or_next(tail, lines, index, context=line)
                for action in action_indices:
                    for state in state_indices:
                        for successor in successor_indices:
                            transition[action, state, successor] = value
                continue
            raise ValueError(f"malformed transition directive: {line}")

        if line.startswith("O:"):
            parts = [part.strip() for part in line.split(":")]
            action_indices = _indices(actions, parts[1], field="action") if len(parts) >= 2 else ()
            if len(parts) == 2:
                values, next_index = _next_values(lines, index, context=line)
                if values == ["identity"]:
                    if state_count > observation_count:
                        raise ValueError("observation identity requires at least as many observations as states")
                    matrix = np.zeros((state_count, observation_count), dtype=np.float64)
                    matrix[np.arange(state_count), np.arange(state_count)] = 1.0
                    index = next_index
                elif values == ["uniform"]:
                    matrix = np.full(
                        (state_count, observation_count), 1.0 / observation_count, dtype=np.float64
                    )
                    index = next_index
                else:
                    rows = [
                        _float_vector(values, observation_count, context=f"{line} row 0")
                    ]
                    for row_index in range(1, state_count):
                        values, next_index = _next_values(
                            lines, next_index - 1, context=f"{line} row {row_index}"
                        )
                        rows.append(
                            _float_vector(
                                values, observation_count, context=f"{line} row {row_index}"
                            )
                        )
                    matrix = np.stack(rows)
                    index = next_index
                for action in action_indices:
                    observation[action] = matrix
                continue
            if len(parts) == 3:
                state_indices = _indices(states, parts[2], field="successor state")
                values, index = _next_values(lines, index, context=line)
                row = _float_vector(values, observation_count, context=f"{line} row")
                for action in action_indices:
                    for state in state_indices:
                        observation[action, state] = row
                continue
            if len(parts) == 4:
                tail = parts[3].split()
                if not tail:
                    raise ValueError(f"missing observation in {line}")
                state_indices = _indices(states, parts[2], field="successor state")
                observation_indices = _indices(
                    observations, tail[0], field="observation"
                )
                value, index = _scalar_from_tail_or_next(tail, lines, index, context=line)
                for action in action_indices:
                    for state in state_indices:
                        for observed in observation_indices:
                            observation[action, state, observed] = value
                continue
            raise ValueError(f"malformed observation directive: {line}")

        if line.startswith("R:"):
            parts = [part.strip() for part in line.split(":")]
            if len(parts) != 5:
                raise ValueError("V68 supports only scalar reward entries")
            tail = parts[4].split()
            if not tail:
                raise ValueError(f"missing reward observation selector in {line}")
            if tail[0] != "*":
                raise ValueError("V68 supports only observation-independent rewards")
            value, index = _scalar_from_tail_or_next(tail, lines, index, context=line)
            for action in _indices(actions, parts[1], field="action"):
                for state in _indices(states, parts[2], field="state"):
                    for successor in _indices(states, parts[3], field="successor state"):
                        reward[action, state, successor] = value
            continue

        raise ValueError(f"unsupported line: {line}")

    if initial is None:
        initial = np.full(state_count, 1.0 / state_count, dtype=np.float64)
    return POMDPModel(
        name=name,
        states=states,
        actions=actions,
        observations=observations,
        discount=discount,
        initial=initial,
        transition=transition,
        observation=observation,
        reward=reward,
    )


def parse_cassandra_pomdp_file(path: str | Path) -> POMDPModel:
    source = Path(path)
    return parse_cassandra_pomdp_text(source.read_text(), name=source.stem)
