from __future__ import annotations

import unittest

from v59_planning import (
    assert_search_payload_is_public,
    evaluate_policy_pair,
    forbidden_latent_conditioned_rollout,
    run_root_sampled_uct,
    sample_root_counts,
)


ACTIONS = [
    {"action": {"id": name}, "key": name}
    for name in ("left", "probe", "right")
]


def toy_rows():
    return [
        {"mode": "left", "success": False, "weight": 0.5},
        {"mode": "right", "success": False, "weight": 0.5},
    ]


def toy_transition(state, action, tick, draw):
    del draw
    result = dict(state)
    if action["id"] == "probe":
        observation = state["mode"]
    else:
        if tick >= 1:
            result["success"] = action["id"] == state["mode"]
        observation = "uninformative"
    return result, observation


def cost(action):
    return 0.01 if action["id"] == "probe" else 0.0


def terminal(state):
    return float(state["success"])


def label(state):
    return state["mode"]


class RootSampledPlanningTests(unittest.TestCase):
    def search(self, *, seed=7, budget=3000, blind=False, **kwargs):
        return run_root_sampled_uct(
            toy_rows(), ACTIONS, 2, 0, budget, seed, toy_transition,
            terminal, cost, label, merge_observations=blind, **kwargs,
        )

    def test_root_sampling_distribution(self):
        counts = sample_root_counts(toy_rows(), 20000, 11, label)
        self.assertLess(abs(counts["left"] / 20000 - 0.5), 0.015)

    def test_budget_accounting_and_complete_root_actions(self):
        result = self.search(budget=64)
        self.assertEqual(result.simulations_run, 64)
        self.assertEqual(result.root.visits, 64)
        self.assertEqual({row["action_key"] for row in result.root_action_rows}, {
            "left", "probe", "right",
        })
        self.assertTrue(all(row["visits"] > 0 for row in result.root_action_rows))

    def test_deterministic_replay(self):
        first = self.search(seed=19, budget=256)
        second = self.search(seed=19, budget=256)
        self.assertEqual(first.tree_sha256, second.tree_sha256)
        self.assertEqual(first.selected_action_key, second.selected_action_key)

    def test_observation_contingent_search_finds_probe(self):
        candidate = self.search(seed=23)
        blind = self.search(seed=23, blind=True)
        self.assertEqual(candidate.selected_action_key, "probe")
        comparison = evaluate_policy_pair(
            candidate, blind, toy_rows(), ACTIONS, 2, 0, 4000, 29,
            toy_transition, terminal, cost, 31,
        )
        self.assertGreater(comparison["candidate_mean_return"], 0.90)
        self.assertGreater(comparison["paired_mean_difference"], 0.35)

    def test_observation_blind_search_merges_children(self):
        blind = self.search(seed=37, blind=True, budget=512)
        self.assertEqual(blind.branching_action_nodes, 0)
        self.assertTrue(all(
            set(stats.children) <= {"*"}
            for stats in blind.root.actions.values()
        ))

    def test_action_cost_is_backed_up(self):
        normal = self.search(seed=41, budget=512)
        mutant = self.search(seed=41, budget=512, omit_action_cost=True)
        normal_probe = next(
            row for row in normal.root_action_rows if row["action_key"] == "probe"
        )
        mutant_probe = next(
            row for row in mutant.root_action_rows if row["action_key"] == "probe"
        )
        self.assertGreater(mutant_probe["mean_return"], normal_probe["mean_return"])

    def test_budget_off_by_one_is_observable(self):
        mutant = self.search(
            seed=43, budget=64, simulation_limit_override=63
        )
        self.assertNotEqual(mutant.simulations_run, mutant.budget)

    def test_latent_rollout_and_truth_firewalls(self):
        with self.assertRaises(PermissionError):
            forbidden_latent_conditioned_rollout({"mode": "left"})
        with self.assertRaises(PermissionError):
            assert_search_payload_is_public({"public": {}, "truth": {}})
        assert_search_payload_is_public({"public": {"goal": "active(unit_0)"}})


if __name__ == "__main__":
    unittest.main()

