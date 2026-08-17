#!/usr/bin/env python3
"""Source-only Cassandra parser for the prospectively pinned V71 models.

This is a fresh parser rather than a modification of the frozen V62/V68
implementations. It supports the exact syntax used by the selected
``pomdp-solve`` examples, including multiline headers, inline rows,
include/exclude starts, reward/cost semantics, and observation-conditioned
immediate values. Sequentially later entries override earlier ones, matching
the source parser.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np

from v62_external_pomdp import POMDPModel, validate_model


HEADER_KEYS = ("discount", "values", "states", "actions", "observations")
DIRECTIVE = re.compile(
    r"^(discount|values|states|actions|observations|start|T|O|R)\s*(?:[^:]*)?:",
    re.IGNORECASE,
)
HEADER = re.compile(
    r"^(discount|values|states|actions|observations)\s*:\s*(.*)$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ParsedPOMDPSource:
    model: POMDPModel
    value_type: str
    raw_reward: np.ndarray  # action, state, successor, source observation
    reward_observation_dependent: bool

    def __post_init__(self) -> None:
        expected = (
            len(self.model.actions),
            len(self.model.states),
            len(self.model.states),
            len(self.model.observations),
        )
        reward = np.asarray(self.raw_reward, dtype=np.float64)
        if reward.shape != expected:
            raise ValueError(f"V71 raw reward shape {reward.shape} != {expected}")
        if not np.isfinite(reward).all():
            raise ValueError("V71 raw reward contains a non-finite value")
        reward.setflags(write=False)
        object.__setattr__(self, "raw_reward", reward)
        if self.value_type not in {"reward", "cost"}:
            raise ValueError("V71 value type must be reward or cost")


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
    if token.isdigit():
        index = int(token)
        if index < len(names):
            return (index,)
        raise ValueError(f"{field} index {index} is out of range")
    try:
        return (names.index(token),)
    except ValueError as exc:
        raise ValueError(f"unknown {field} label {token!r}") from exc


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


def _collect_headers(lines: list[str]) -> tuple[dict[str, list[str]], set[int]]:
    headers: dict[str, list[str]] = {}
    consumed: set[int] = set()
    index = 0
    while index < len(lines):
        match = HEADER.match(lines[index])
        if match is None:
            index += 1
            continue
        key = match.group(1).lower()
        if key in headers:
            raise ValueError(f"duplicate {key} header")
        values = match.group(2).split()
        consumed.add(index)
        next_index = index + 1
        if key in {"states", "actions", "observations"} and not values:
            while next_index < len(lines) and DIRECTIVE.match(lines[next_index]) is None:
                values.extend(lines[next_index].split())
                consumed.add(next_index)
                next_index += 1
        headers[key] = values
        index = next_index
    missing = set(HEADER_KEYS) - set(headers)
    if missing:
        raise ValueError(f"missing headers: {sorted(missing)}")
    return headers, consumed


def _next_tokens(
    lines: list[str], index: int, consumed: set[int], *, context: str
) -> tuple[list[str], int]:
    next_index = index + 1
    while next_index in consumed:
        next_index += 1
    if next_index >= len(lines) or DIRECTIVE.match(lines[next_index]):
        raise ValueError(f"missing values after {context}")
    return lines[next_index].split(), next_index + 1


def _matrix(
    first_tokens: list[str] | None,
    lines: list[str],
    index: int,
    consumed: set[int],
    *,
    rows: int,
    columns: int,
    context: str,
) -> tuple[np.ndarray, int]:
    values: list[np.ndarray] = []
    cursor = index
    if first_tokens is not None:
        values.append(_float_vector(first_tokens, columns, context=f"{context} row 0"))
    while len(values) < rows:
        while cursor in consumed:
            cursor += 1
        if cursor >= len(lines) or DIRECTIVE.match(lines[cursor]):
            raise ValueError(f"missing values after {context} row {len(values)}")
        tokens = lines[cursor].split()
        values.append(
            _float_vector(tokens, columns, context=f"{context} row {len(values)}")
        )
        cursor += 1
    return np.stack(values), cursor


def _scalar(tokens: list[str], *, context: str) -> float:
    if len(tokens) != 1:
        raise ValueError(f"{context} requires exactly one scalar")
    try:
        value = float(tokens[0])
    except ValueError as exc:
        raise ValueError(f"invalid scalar in {context}") from exc
    if not np.isfinite(value):
        raise ValueError(f"non-finite scalar in {context}")
    return value


def _start_distribution(
    lines: list[str],
    index: int,
    consumed: set[int],
    states: tuple[str, ...],
) -> tuple[np.ndarray, int]:
    line = lines[index]
    include = re.match(r"^start\s+(include|exclude)\s*:\s*(.*)$", line, re.I)
    if include:
        mode = include.group(1).lower()
        selected = include.group(2).split()
        if not selected:
            selected, next_index = _next_tokens(
                lines, index, consumed, context="start include/exclude"
            )
        else:
            next_index = index + 1
        selected_indices = {
            _indices(states, token, field="start state")[0] for token in selected
        }
        support = (
            selected_indices
            if mode == "include"
            else set(range(len(states))) - selected_indices
        )
        if not support:
            raise ValueError("start include/exclude leaves empty support")
        initial = np.zeros(len(states), dtype=np.float64)
        initial[list(sorted(support))] = 1.0 / len(support)
        return initial, next_index

    match = re.match(r"^start\s*:\s*(.*)$", line, re.I)
    if match is None:
        raise ValueError(f"malformed start directive: {line}")
    tokens = match.group(1).split()
    if not tokens:
        tokens, next_index = _next_tokens(lines, index, consumed, context="start")
    else:
        next_index = index + 1
    if tokens == ["uniform"]:
        return np.full(len(states), 1.0 / len(states)), next_index
    return _float_vector(tokens, len(states), context="start vector"), next_index


def parse_cassandra_pomdp_text(
    text: str, *, name: str = "model"
) -> ParsedPOMDPSource:
    lines = _clean_lines(text)
    headers, consumed = _collect_headers(lines)
    if len(headers["discount"]) != 1:
        raise ValueError("discount header requires one scalar")
    discount = float(headers["discount"][0])
    raw_value_type = headers["values"]
    if len(raw_value_type) != 1:
        raise ValueError("values header requires one label")
    value_type = raw_value_type[0].lower()
    if value_type == "rewards":
        value_type = "reward"
    if value_type not in {"reward", "cost"}:
        raise ValueError(f"unsupported values type {raw_value_type[0]!r}")
    states = _names(headers["states"], field="state")
    actions = _names(headers["actions"], field="action")
    observations = _names(headers["observations"], field="observation")
    state_count = len(states)
    action_count = len(actions)
    observation_count = len(observations)

    transition = np.zeros((action_count, state_count, state_count), dtype=np.float64)
    observation = np.zeros(
        (action_count, state_count, observation_count), dtype=np.float64
    )
    raw_reward = np.zeros(
        (action_count, state_count, state_count, observation_count),
        dtype=np.float64,
    )
    initial: np.ndarray | None = None

    index = 0
    while index < len(lines):
        if index in consumed:
            index += 1
            continue
        line = lines[index]
        if re.match(r"^start(?:\s|:)", line, re.I):
            if initial is not None:
                raise ValueError("duplicate start directive")
            initial, index = _start_distribution(lines, index, consumed, states)
            continue

        parts = [part.strip() for part in re.split(r"\s*:\s*", line)]
        directive = parts[0].upper()
        if directive == "T":
            if len(parts) < 2:
                raise ValueError(f"malformed transition directive: {line}")
            action_tokens = parts[1].split()
            if len(action_tokens) != 1:
                raise ValueError(f"invalid transition action in {line}")
            action_indices = _indices(actions, action_tokens[0], field="action")
            if len(parts) == 2:
                tokens, next_index = _next_tokens(lines, index, consumed, context=line)
                if tokens == ["identity"]:
                    matrix = np.eye(state_count)
                    index = next_index
                elif tokens == ["uniform"]:
                    matrix = np.full(
                        (state_count, state_count), 1.0 / state_count
                    )
                    index = next_index
                else:
                    matrix, index = _matrix(
                        tokens,
                        lines,
                        next_index,
                        consumed,
                        rows=state_count,
                        columns=state_count,
                        context=line,
                    )
                for action in action_indices:
                    transition[action] = matrix
                continue
            if len(parts) == 3:
                tail = parts[2].split()
                if not tail:
                    raise ValueError(f"missing transition state in {line}")
                state_indices = _indices(states, tail[0], field="state")
                tokens = tail[1:]
                if not tokens:
                    tokens, index = _next_tokens(lines, index, consumed, context=line)
                else:
                    index += 1
                row = _float_vector(tokens, state_count, context=f"{line} row")
                for action in action_indices:
                    for state in state_indices:
                        transition[action, state] = row
                continue
            if len(parts) == 4:
                state_tokens = parts[2].split()
                tail = parts[3].split()
                if len(state_tokens) != 1 or len(tail) != 2:
                    raise ValueError(f"malformed scalar transition in {line}")
                value = _scalar(tail[1:], context=line)
                for action in action_indices:
                    for state in _indices(states, state_tokens[0], field="state"):
                        for successor in _indices(
                            states, tail[0], field="successor state"
                        ):
                            transition[action, state, successor] = value
                index += 1
                continue
            raise ValueError(f"malformed transition directive: {line}")

        if directive == "O":
            if len(parts) < 2:
                raise ValueError(f"malformed observation directive: {line}")
            action_tokens = parts[1].split()
            if len(action_tokens) != 1:
                raise ValueError(f"invalid observation action in {line}")
            action_indices = _indices(actions, action_tokens[0], field="action")
            if len(parts) == 2:
                tokens, next_index = _next_tokens(lines, index, consumed, context=line)
                if tokens == ["uniform"]:
                    matrix = np.full(
                        (state_count, observation_count), 1.0 / observation_count
                    )
                    index = next_index
                elif tokens == ["identity"]:
                    if observation_count < state_count:
                        raise ValueError(
                            "observation identity requires at least as many observations as states"
                        )
                    matrix = np.zeros((state_count, observation_count))
                    matrix[np.arange(state_count), np.arange(state_count)] = 1.0
                    index = next_index
                else:
                    matrix, index = _matrix(
                        tokens,
                        lines,
                        next_index,
                        consumed,
                        rows=state_count,
                        columns=observation_count,
                        context=line,
                    )
                for action in action_indices:
                    observation[action] = matrix
                continue
            if len(parts) == 3:
                tail = parts[2].split()
                if not tail:
                    raise ValueError(f"missing observation state in {line}")
                state_indices = _indices(states, tail[0], field="successor state")
                tokens = tail[1:]
                if not tokens:
                    tokens, index = _next_tokens(lines, index, consumed, context=line)
                else:
                    index += 1
                row = _float_vector(
                    tokens, observation_count, context=f"{line} row"
                )
                for action in action_indices:
                    for state in state_indices:
                        observation[action, state] = row
                continue
            if len(parts) == 4:
                state_tokens = parts[2].split()
                tail = parts[3].split()
                if len(state_tokens) != 1 or len(tail) != 2:
                    raise ValueError(f"malformed scalar observation in {line}")
                value = _scalar(tail[1:], context=line)
                for action in action_indices:
                    for state in _indices(
                        states, state_tokens[0], field="successor state"
                    ):
                        for observed in _indices(
                            observations, tail[0], field="observation"
                        ):
                            observation[action, state, observed] = value
                index += 1
                continue
            raise ValueError(f"malformed observation directive: {line}")

        if directive == "R":
            if len(parts) != 5:
                raise ValueError(f"V71 supports scalar reward entries only: {line}")
            action_tokens = parts[1].split()
            state_tokens = parts[2].split()
            successor_tokens = parts[3].split()
            tail = parts[4].split()
            if (
                len(action_tokens) != 1
                or len(state_tokens) != 1
                or len(successor_tokens) != 1
                or len(tail) != 2
            ):
                raise ValueError(f"malformed scalar reward in {line}")
            value = _scalar(tail[1:], context=line)
            if value_type == "cost":
                value = -value
            for action in _indices(actions, action_tokens[0], field="action"):
                for state in _indices(states, state_tokens[0], field="state"):
                    for successor in _indices(
                        states, successor_tokens[0], field="successor state"
                    ):
                        for observed in _indices(
                            observations, tail[0], field="observation"
                        ):
                            raw_reward[action, state, successor, observed] = value
            index += 1
            continue

        raise ValueError(f"unsupported line: {line}")

    if initial is None:
        initial = np.full(state_count, 1.0 / state_count)
    # The delivered sensor label is transformed only after the source reward is
    # generated. Collapse the source observation-conditioned reward exactly.
    reward = np.einsum("aszo,azo->asz", raw_reward, observation)
    model = POMDPModel(
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
    dependent = bool(
        np.any(np.max(raw_reward, axis=-1) - np.min(raw_reward, axis=-1) > 0.0)
    )
    return ParsedPOMDPSource(
        model=model,
        value_type=value_type,
        raw_reward=raw_reward,
        reward_observation_dependent=dependent,
    )


def parse_cassandra_pomdp_file(path: str | Path) -> ParsedPOMDPSource:
    source = Path(path)
    return parse_cassandra_pomdp_text(source.read_text(), name=source.stem)


def source_validation(parsed: ParsedPOMDPSource) -> dict[str, bool]:
    checks = validate_model(parsed.model, atol=1e-12)
    checks["at_least_two_observations"] = len(parsed.model.observations) >= 2
    checks["finite_raw_reward"] = bool(np.isfinite(parsed.raw_reward).all())
    return checks
