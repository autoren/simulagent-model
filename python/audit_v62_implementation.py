#!/usr/bin/env python3
"""Audit V62 without opening the external benchmark bundle."""
from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from functools import lru_cache
from pathlib import Path

import numpy as np

from test_v62_external_pomdp import make_tiger, make_tmaze
from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from v62_external_pomdp import (
    ExactPlanner,
    POMDPModel,
    condition_initial,
    expected_reward,
    parse_pomdp_text,
    public_policy_value,
    terminal_mask,
    update_belief,
    validate_model,
)


FIXTURE_TEXT = """
discount: 0.8
values: reward
states: s0 s1 terminal
actions: sense commit
observations: left right done
start: 0.75 0.25 0
T: sense
0 1 0
1 0 0
0 0 1
T: commit
0 0 1
0 0 1
0 0 1
O: *
0.8 0.2 0
0.1 0.9 0
0 0 1
R: * : * : * : * -1
R: commit : s0 : terminal : * 3
R: commit : s1 : terminal : * 7
"""


def scalar_parse_fixture(text: str) -> dict[str, object]:
    """Independent parser for the one audit fixture; no candidate parser calls."""
    rows = []
    for raw in text.splitlines():
        stripped = raw.split("#", 1)[0].strip()
        if stripped:
            rows.append(stripped)
    metadata: dict[str, object] = {}
    for row in rows:
        if row.startswith("discount:"):
            metadata["discount"] = float(row.split(":", 1)[1])
        elif row.startswith("states:"):
            metadata["states"] = tuple(row.split(":", 1)[1].split())
        elif row.startswith("actions:"):
            metadata["actions"] = tuple(row.split(":", 1)[1].split())
        elif row.startswith("observations:"):
            metadata["observations"] = tuple(row.split(":", 1)[1].split())
    states = metadata["states"]
    actions = metadata["actions"]
    observations = metadata["observations"]
    s_count, a_count, o_count = len(states), len(actions), len(observations)
    transition = np.zeros((a_count, s_count, s_count))
    observation = np.zeros((a_count, s_count, o_count))
    reward = np.zeros((a_count, s_count, s_count))
    cursor = 0
    initial = None
    while cursor < len(rows):
        row = rows[cursor]
        if row.startswith("start:"):
            initial = np.asarray([float(x) for x in row.split(":", 1)[1].split()])
        elif row.startswith("T:"):
            action = actions.index(row.split(":", 1)[1].strip())
            matrix = [[float(x) for x in rows[cursor + offset + 1].split()] for offset in range(s_count)]
            transition[action] = matrix
            cursor += s_count
        elif row.startswith("O:"):
            matrix = np.asarray(
                [[float(x) for x in rows[cursor + offset + 1].split()] for offset in range(s_count)]
            )
            observation[:] = matrix
            cursor += s_count
        elif row.startswith("R:"):
            pieces = [piece.strip() for piece in row.split(":")]
            obs_and_value = pieces[-1].split()
            value = float(obs_and_value[-1])
            action_ids = range(a_count) if pieces[1] == "*" else (actions.index(pieces[1]),)
            state_ids = range(s_count) if pieces[2] == "*" else (states.index(pieces[2]),)
            successor_ids = range(s_count) if pieces[3] == "*" else (states.index(pieces[3]),)
            for action in action_ids:
                for state in state_ids:
                    for successor in successor_ids:
                        reward[action, state, successor] = value
        cursor += 1
    return {
        **metadata,
        "initial": initial,
        "transition": transition,
        "observation": observation,
        "reward": reward,
    }


def ref_terminal(model: POMDPModel) -> np.ndarray:
    result = np.ones(len(model.states), dtype=bool)
    for state in range(len(model.states)):
        for action in range(len(model.actions)):
            for successor in range(len(model.states)):
                expected = 1.0 if successor == state else 0.0
                if abs(model.transition[action, state, successor] - expected) > 1e-12:
                    result[state] = False
    return result


def ref_normalize(values: np.ndarray) -> tuple[np.ndarray, float]:
    mass = sum(float(value) for value in values)
    if mass <= 0.0:
        raise ValueError("zero reference mass")
    return np.asarray([float(value) / mass for value in values]), mass


def ref_update(
    model: POMDPModel, belief: np.ndarray, action: int, observation: int
) -> tuple[np.ndarray, float]:
    weights = []
    for successor in range(len(model.states)):
        prediction = 0.0
        for state in range(len(model.states)):
            prediction += float(belief[state]) * float(model.transition[action, state, successor])
        weights.append(prediction * float(model.observation[action, successor, observation]))
    return ref_normalize(np.asarray(weights))


def ref_expected_reward(model: POMDPModel, belief: np.ndarray, action: int) -> float:
    total = 0.0
    for state in range(len(model.states)):
        for successor in range(len(model.states)):
            total += (
                float(belief[state])
                * float(model.transition[action, state, successor])
                * float(model.reward[action, state, successor])
            )
    return total


def ref_initial_value(model: POMDPModel, horizon: int) -> tuple[float, dict[int, tuple[int, ...]]]:
    terminals = ref_terminal(model)

    @lru_cache(maxsize=None)
    def recurse(belief_tuple: tuple[float, ...], remaining: int) -> tuple[float, tuple[int, ...]]:
        belief = np.asarray(belief_tuple)
        support = [state for state, mass in enumerate(belief) if mass > 1e-14]
        if remaining <= 0 or all(terminals[state] for state in support):
            return 0.0, tuple(range(len(model.actions)))
        values = []
        for action in range(len(model.actions)):
            value = ref_expected_reward(model, belief, action)
            if remaining > 1:
                continuation = 0.0
                for observation in range(len(model.observations)):
                    try:
                        posterior, probability = ref_update(model, belief, action, observation)
                    except ValueError:
                        continue
                    continuation += probability * recurse(
                        tuple(float(x) for x in np.round(posterior, 15)), remaining - 1
                    )[0]
                value += float(model.discount) * continuation
            values.append(value)
        maximum = max(values)
        optimal = tuple(index for index, value in enumerate(values) if maximum - value <= 1e-12)
        return maximum, optimal

    initial_observation_probabilities = []
    for observation in range(len(model.observations)):
        probability = 0.0
        for state in range(len(model.states)):
            probability += float(model.initial[state]) * float(model.observation[0, state, observation])
        initial_observation_probabilities.append(probability)
    value = 0.0
    actions: dict[int, tuple[int, ...]] = {}
    for observation, probability in enumerate(initial_observation_probabilities):
        if probability <= 1e-15:
            continue
        weights = np.asarray(
            [
                float(model.initial[state]) * float(model.observation[0, state, observation])
                for state in range(len(model.states))
            ]
        )
        belief = ref_normalize(weights)[0]
        cell_value, optimal = recurse(
            tuple(float(x) for x in np.round(belief, 15)), horizon
        )
        value += probability * cell_value
        actions[observation] = optimal
    return value, actions


def runtime_versions(python: Path) -> dict[str, str]:
    script = (
        "import json,sys,jax,jaxlib,chex,numpy,importlib.metadata as m;"
        "print(json.dumps({'python':'.'.join(map(str,sys.version_info[:3])),"
        "'jax':jax.__version__,'jaxlib':jaxlib.__version__,'chex':chex.__version__,"
        "'gymnax':m.version('gymnax'),'numpy':numpy.__version__},sort_keys=True))"
    )
    return json.loads(subprocess.check_output([str(python), "-c", script], text=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--design-lock", default="configs/v62-design-lock.json")
    parser.add_argument(
        "--runtime-python", default="data/v62-external-pomdp-transfer/runtime/bin/python"
    )
    parser.add_argument(
        "--output", default="outputs/v62-external-pomdp-transfer/implementation-audit.json"
    )
    args = parser.parse_args()
    design_path = (PROJECT_ROOT / args.design_lock).resolve()
    # Keep the virtual-environment launcher path; resolving its symlink would
    # bypass pyvenv.cfg and invoke the base interpreter without the packages.
    runtime_python = PROJECT_ROOT / args.runtime_python
    output = (PROJECT_ROOT / args.output).resolve()
    design = json.loads(design_path.read_text())
    config = design["config_payload"]
    errors: list[str] = []

    if not design["authorization"]["write_and_audit_v62_implementation"]:
        errors.append("V62 design does not authorize implementation audit")
    external_bundle = PROJECT_ROOT / "data/v62-external-pomdp-transfer/bundle"
    source_inaccessible = not external_bundle.exists()
    if not source_inaccessible:
        errors.append("external source bundle exists before implementation lock")

    candidate = parse_pomdp_text(FIXTURE_TEXT, name="audit_fixture")
    reference = scalar_parse_fixture(FIXTURE_TEXT)
    parser_errors = {
        "discount": abs(candidate.discount - reference["discount"]),
        "initial": float(np.max(np.abs(candidate.initial - reference["initial"]))),
        "transition": float(np.max(np.abs(candidate.transition - reference["transition"]))),
        "observation": float(np.max(np.abs(candidate.observation - reference["observation"]))),
        "reward": float(np.max(np.abs(candidate.reward - reference["reward"]))),
    }
    independent_parser_passed = max(parser_errors.values()) <= 1e-12

    tiger = make_tiger()
    tmaze2 = make_tmaze(2)
    tmaze5 = make_tmaze(5)
    planner_agreements = []
    planner_value_errors = []
    for model, horizon in ((candidate, 3), (tiger, 3), (tmaze2, 4), (tmaze5, 7)):
        candidate_planner = ExactPlanner(model)
        candidate_value = candidate_planner.initial_value(horizon)
        reference_value, reference_actions = ref_initial_value(model, horizon)
        planner_value_errors.append(abs(candidate_value - reference_value))
        for observation, optimal in reference_actions.items():
            belief = condition_initial(model, observation)[0]
            planner_agreements.append(candidate_planner.decision(belief, horizon).action in optimal)
    independent_planner_passed = (
        max(planner_value_errors) <= 1e-10 and all(planner_agreements)
    )

    fixtures = {
        "deterministic_one_state_absorber": bool(terminal_mask(candidate)[-1]),
        "binary_noisy_sensor_bayes_update": bool(
            np.allclose(
                update_belief(tiger, condition_initial(tiger, 0)[0], 0, 1)[0],
                (0.0, 0.0, 0.85, 0.15, 0.0),
                atol=1e-12,
            )
        ),
        "discounted_delayed_reward": bool(
            abs(ExactPlanner(tmaze2).initial_value(4) - 4.0 * 0.9**3) <= 1e-12
        ),
        "listen_before_commit_information_value": bool(
            ExactPlanner(tiger).decision(condition_initial(tiger, 0)[0], 3).action == 0
        ),
        "all_action_absorbing_terminal_rule": bool(
            terminal_mask(tmaze2).sum() == 1 and terminal_mask(tmaze2)[-1]
        ),
        "zero_probability_observation_is_omitted": bool(
            len(ExactPlanner(tmaze2).reachable_decisions(4)) > 0
        ),
    }
    analytic_fixture_pass_rate = sum(fixtures.values()) / len(fixtures)

    asymmetric_transition = np.array(candidate.transition, copy=True).transpose(0, 2, 1)
    transposed = POMDPModel(
        candidate.name, candidate.states, candidate.actions, candidate.observations,
        candidate.discount, candidate.initial, asymmetric_transition,
        candidate.observation, candidate.reward,
    )
    initial_tiger = condition_initial(tiger, 0)[0]
    correct_posterior = update_belief(tiger, initial_tiger, 0, 1)[0]
    dropped_likelihood = initial_tiger @ tiger.transition[0]
    pretransition_weights = initial_tiger * tiger.observation[0, :, 1]
    nonuniform = condition_initial(candidate, 0)[0]
    uniform_initial = np.full(len(candidate.states), 1.0 / len(candidate.states))
    uniform_conditioned = uniform_initial * candidate.observation[0, :, 0]
    uniform_conditioned /= uniform_conditioned.sum()
    tmaze_exact = ExactPlanner(tmaze2).initial_value(4)
    tiger_decision = ExactPlanner(tiger).decision(initial_tiger, 3)
    ignored_initial = public_policy_value(tmaze2, 4, "observation_only")
    any_self_terminal = np.asarray(
        [
            any(
                abs(tmaze2.transition[action, state, state] - 1.0) <= 1e-12
                for action in range(len(tmaze2.actions))
            )
            for state in range(len(tmaze2.states))
        ]
    )
    reward_swapped = 0.0
    for state, mass in enumerate(candidate.initial):
        for successor, probability in enumerate(candidate.transition[1, state]):
            reward_swapped += mass * probability * candidate.reward[1, successor, state]
    wildcard_first_only = np.array(candidate.reward, copy=True)
    wildcard_first_only[1] += 1.0
    mutants = {
        "transpose_transition_axes": not all(validate_model(transposed).values()),
        "use_pretransition_observation": pretransition_weights.sum() <= 1e-15,
        "drop_observation_likelihood": not np.allclose(dropped_likelihood, correct_posterior),
        "replace_initial_prior_with_uniform": not np.allclose(nonuniform, uniform_conditioned),
        "omit_discount_factor": abs(tmaze_exact - 4.0) > 1e-6,
        "choose_minimum_action_value": tiger_decision.action != int(np.argmin(tiger_decision.q_values)),
        "collapse_belief_to_map_state": (
            public_policy_value(tiger, 3, "exact_history")
            - public_policy_value(tiger, 3, "map_collapse") > 1.0
        ),
        "ignore_initial_observation": tmaze_exact - ignored_initial > 0.9,
        "declare_terminal_if_any_action_self_loops": bool(any_self_terminal[0] and not terminal_mask(tmaze2)[0]),
        "swap_reward_source_and_successor_indices": abs(
            expected_reward(candidate, candidate.initial, 1) - reward_swapped
        ) > 1e-6,
        "apply_wildcard_to_only_first_action": not np.allclose(wildcard_first_only, candidate.reward),
        "reverse_canonical_tie_break": ExactPlanner(tmaze2).decision(condition_initial(tmaze2, 0)[0], 1).action != 3,
    }
    fixtures = {name: bool(value) for name, value in fixtures.items()}
    mutants = {name: bool(value) for name, value in mutants.items()}
    expected_mutants = config["implementationAudit"]["mutants"]
    mutant_census_ok = list(mutants) == expected_mutants
    mutation_kill_rate = sum(mutants.values()) / len(mutants)
    if not mutant_census_ok:
        errors.append("mutation census differs from the frozen design")
    if mutation_kill_rate != 1.0:
        errors.append("not all V62 semantic mutants were killed")
    if analytic_fixture_pass_rate != 1.0:
        errors.append("not all V62 analytic fixtures passed")
    if not independent_parser_passed or not independent_planner_passed:
        errors.append("candidate disagrees with an independent parser or planner")

    core_path = PROJECT_ROOT / "python/v62_external_pomdp.py"
    core_text = core_path.read_text()
    forbidden_imports = config["candidateSystem"]["candidateDoesNotImport"]
    import_firewall_ok = all(token not in core_text for token in forbidden_imports)
    if not import_firewall_ok:
        errors.append("candidate imports a forbidden POBAX or internal implementation")

    versions = runtime_versions(runtime_python)
    expected_versions = {"python": config["runtime"]["python"], **config["runtime"]["packages"]}
    runtime_ok = versions == expected_versions
    if not runtime_ok:
        errors.append("isolated external runtime versions do not match the design")

    implementation_files = [
        "python/v62_external_pomdp.py",
        "python/test_v62_external_pomdp.py",
    ]
    result = {
        "schema_version": 62,
        "experiment": "v62_implementation_audit",
        "passed": not errors,
        "decision": "freeze_v62_implementation" if not errors else "repair_v62_implementation",
        "errors": errors,
        "design_lock": str(design_path.relative_to(PROJECT_ROOT)),
        "design_lock_sha256": file_sha256(design_path),
        "implementation_files_sha256": {
            path: file_sha256(PROJECT_ROOT / path) for path in implementation_files
        },
        "independent_parser_max_errors": parser_errors,
        "independent_planner_max_value_error": max(planner_value_errors),
        "independent_planner_action_agreement_rate": sum(planner_agreements) / len(planner_agreements),
        "analytic_fixtures": fixtures,
        "analytic_fixture_pass_rate": analytic_fixture_pass_rate,
        "mutants": mutants,
        "mutation_census_ok": mutant_census_ok,
        "mutation_kill_rate": mutation_kill_rate,
        "runtime_versions": versions,
        "checks": {
            "external_source_bundle_absent": source_inaccessible,
            "independent_parser_agreement": independent_parser_passed,
            "independent_scalar_planner_agreement": independent_planner_passed,
            "six_analytic_fixtures": analytic_fixture_pass_rate == 1.0,
            "twelve_semantic_mutants": mutant_census_ok and mutation_kill_rate == 1.0,
            "candidate_import_firewall": import_firewall_ok,
            "isolated_runtime_versions": runtime_ok,
        },
        "data_access": {
            "external_model_definition_files_read": 0,
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
