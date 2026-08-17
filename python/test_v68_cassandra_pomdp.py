#!/usr/bin/env python3
"""Parser and pinned-source parity tests for the V68 feasibility stage."""
from __future__ import annotations

import ast
from pathlib import Path
from typing import List, Union
import unittest

import numpy as np

from v22r2_grounding import PROJECT_ROOT
from v62_external_pomdp import validate_model
from v68_cassandra_pomdp import (
    parse_cassandra_pomdp_file,
    parse_cassandra_pomdp_text,
)


SOURCE_ROOT = (
    PROJECT_ROOT
    / "data/v63-external-unknown-dynamics/source-checkout/pobax/envs/classic/POMDP"
)
REFERENCE_SOURCE = SOURCE_ROOT.parent / "__init__.py"


def load_pobax_reference_parser():
    """Load only POBAX's parser class, avoiding its optional runtime deps."""
    tree = ast.parse(REFERENCE_SOURCE.read_text(), filename=str(REFERENCE_SOURCE))
    selected = [
        node
        for node in tree.body
        if (isinstance(node, ast.ClassDef) and node.name == "POMDPFile")
        or (isinstance(node, ast.FunctionDef) and node.name == "is_numeric")
    ]
    if len(selected) != 2:
        raise AssertionError("could not isolate the pinned POBAX reference parser")
    module = ast.Module(body=selected, type_ignores=[])
    namespace = {"np": np, "List": List, "Union": Union}
    exec(compile(module, str(REFERENCE_SOURCE), "exec"), namespace)
    return namespace["POMDPFile"]


class V68CassandraPOMDPTests(unittest.TestCase):
    def test_all_pinned_models_match_pobax_reference_arrays(self) -> None:
        reference_parser = load_pobax_reference_parser()
        sources = sorted(SOURCE_ROOT.glob("*.POMDP"))
        self.assertEqual(len(sources), 14)
        for source in sources:
            with self.subTest(source=source.name):
                candidate = parse_cassandra_pomdp_file(source)
                reference = reference_parser(source)
                self.assertEqual(candidate.states, tuple(reference.states))
                self.assertEqual(candidate.actions, tuple(reference.actions))
                self.assertEqual(candidate.observations, tuple(reference.observations))
                self.assertEqual(candidate.discount, reference.discount)
                self.assertTrue(np.array_equal(candidate.initial, reference.start))
                self.assertTrue(np.array_equal(candidate.transition, reference.T))
                self.assertTrue(np.array_equal(candidate.observation, reference.Z))
                self.assertTrue(np.array_equal(candidate.reward, reference.R))

    def test_source_validation_identifies_only_paint_observation_defect(self) -> None:
        failures: dict[str, list[str]] = {}
        for source in sorted(SOURCE_ROOT.glob("*.POMDP")):
            checks = validate_model(parse_cassandra_pomdp_file(source))
            failed = sorted(key for key, passed in checks.items() if not passed)
            if failed:
                failures[source.name] = failed
        self.assertEqual(failures, {"paint.POMDP": ["observation_normalized"]})

    def test_sparse_keywords_rows_and_scalar_continuations(self) -> None:
        text = """
        discount: 0.9
        values: reward
        states: s0 s1
        actions: stay mix
        observations: o0 o1
        start: 1 0
        T: *
        identity
        T: mix : s0
        0.25 0.75
        T: mix : s1 : s0
        0.5
        T: mix : s1 : s1 0.5
        O: stay
        identity
        O: mix
        uniform
        R: * : * : * : *
        -1
        R: mix : s0 : s1 : * 3
        """
        model = parse_cassandra_pomdp_text(text)
        self.assertTrue(all(validate_model(model).values()))
        self.assertTrue(np.array_equal(model.transition[0], np.eye(2)))
        self.assertTrue(np.array_equal(model.transition[1], [[0.25, 0.75], [0.5, 0.5]]))
        self.assertTrue(np.array_equal(model.observation[0], np.eye(2)))
        self.assertTrue(np.array_equal(model.observation[1], np.full((2, 2), 0.5)))
        self.assertEqual(model.reward[1, 0, 1], 3.0)
        self.assertEqual(model.reward[0, 0, 0], -1.0)

    def test_observation_dependent_reward_is_rejected(self) -> None:
        text = """
        discount: 1
        values: reward
        states: 1
        actions: 1
        observations: 1
        start: 1
        T: 0
        identity
        O: 0
        identity
        R: 0 : 0 : 0 : 0 1
        """
        with self.assertRaisesRegex(ValueError, "observation-independent"):
            parse_cassandra_pomdp_text(text)

    def test_start_include_exclude_is_explicitly_out_of_scope(self) -> None:
        text = """
        discount: 1
        values: reward
        states: s0 s1
        actions: a
        observations: o
        start: include s0
        T: a
        identity
        O: a
        1
        1
        R: * : * : * : * 0
        """
        with self.assertRaisesRegex(ValueError, "outside the frozen V68 subset"):
            parse_cassandra_pomdp_text(text)


if __name__ == "__main__":
    unittest.main()
