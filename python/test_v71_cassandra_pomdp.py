#!/usr/bin/env python3
"""Source-only tests for the prospectively pinned V71 model parser."""
from __future__ import annotations

from pathlib import Path
import unittest

import numpy as np

from v71_cassandra_pomdp import parse_cassandra_pomdp_file, source_validation


SOURCE = Path(
    "data/v71-sensor-codebook/source-checkout/examples/pomdp-files"
)
SELECTED = {
    "concert.POMDP": (2, 3, 2, "reward"),
    "ejs1.POMDP": (3, 4, 2, "reward"),
    "manuel-hartman.2013-09-19.POMDP": (2, 2, 2, "reward"),
    "ejs2.POMDP": (2, 2, 2, "reward"),
    "ejs3.POMDP": (2, 2, 2, "cost"),
    "bridge-repair.POMDP": (5, 12, 5, "cost"),
    "parr95.95.POMDP": (7, 3, 6, "reward"),
    "uav-search.raissa-bravo.orig.POMDP": (8, 8, 2, "reward"),
}


class V71CassandraParserTests(unittest.TestCase):
    def parse(self, filename: str):
        return parse_cassandra_pomdp_file(SOURCE / filename)

    def test_selected_models_are_strictly_valid(self) -> None:
        for filename, expected in SELECTED.items():
            with self.subTest(filename=filename):
                parsed = self.parse(filename)
                model = parsed.model
                actual = (
                    len(model.states),
                    len(model.actions),
                    len(model.observations),
                    parsed.value_type,
                )
                self.assertEqual(actual, expected)
                self.assertTrue(all(source_validation(parsed).values()))

    def test_costs_are_converted_to_negative_rewards(self) -> None:
        parsed = self.parse("bridge-repair.POMDP")
        action = parsed.model.actions.index("no-repair-and-visual-inspect")
        state = parsed.model.states.index("less-5")
        np.testing.assert_allclose(parsed.raw_reward[action, state], -4.0)
        np.testing.assert_allclose(parsed.model.reward[action, state], -4.0)

    def test_start_include_is_uniform_on_selected_support(self) -> None:
        parsed = self.parse("parr95.95.POMDP")
        expected = np.zeros(7)
        expected[parsed.model.states.index("I")] = 1.0
        np.testing.assert_array_equal(parsed.model.initial, expected)

    def test_later_reward_entry_overrides_earlier_entry(self) -> None:
        parsed = self.parse("ejs2.POMDP")
        np.testing.assert_array_equal(parsed.raw_reward[0, 0], 0.0)

    def test_observation_dependent_reward_is_collapsed_exactly(self) -> None:
        parsed = self.parse("uav-search.raissa-bravo.orig.POMDP")
        self.assertTrue(parsed.reward_observation_dependent)
        expected = np.einsum(
            "aszo,azo->asz", parsed.raw_reward, parsed.model.observation
        )
        np.testing.assert_array_equal(parsed.model.reward, expected)
        successor = parsed.model.states.index("a")
        self.assertEqual(parsed.model.reward[0, 0, successor], 0.125)

    def test_rounded_transition_sources_are_rejected_without_repair(self) -> None:
        for filename in ("1d.POMDP", "aloha-10max.POMDP"):
            with self.subTest(filename=filename):
                parsed = self.parse(filename)
                self.assertFalse(source_validation(parsed)["transition_normalized"])

    def test_missing_discount_sources_are_rejected(self) -> None:
        for filename in (
            "ejs4.POMDP",
            "ejs5.POMDP",
            "ejs6.POMDP",
            "ejs7.POMDP",
        ):
            with self.subTest(filename=filename):
                with self.assertRaisesRegex(ValueError, "missing headers.*discount"):
                    self.parse(filename)


if __name__ == "__main__":
    unittest.main()
