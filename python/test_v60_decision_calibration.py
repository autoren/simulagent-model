from __future__ import annotations

import unittest

import v59_planning as v59
from v22_relational import canonical_json
from v60_decision_calibration import (
    atom_marginals,
    belief_comparison,
    forbidden_truth_conditioned_belief,
    run_root_sampled_uct_fast,
    smc2_atoms_for_planning,
)


class V60DecisionCalibrationTests(unittest.TestCase):
    def test_cdf_sampler_replays_linear_sampler(self):
        rows = [
            {"key": "a", "weight": 0.2, "state": 0},
            {"key": "b", "weight": 0.3, "state": 1},
            {"key": "c", "weight": 0.5, "state": 2},
        ]
        actions = [
            {"key": "left", "action": {"id": "left"}},
            {"key": "right", "action": {"id": "right"}},
        ]
        transition = lambda state, action, tick, draw: (
            {**state, "state": (state["state"] + (action["id"] == "right")) % 3},
            str((state["state"] + tick + int(draw > 0.5)) % 2),
        )
        kwargs = dict(
            root_rows=rows, action_rows=actions, horizon=4, tick=0,
            budget=257, seed=6001, transition_fn=transition,
            terminal_fn=lambda state: float(state["state"] == 2),
            action_cost_fn=lambda action: 0.01 * (action["id"] == "right"),
            static_label_fn=lambda state: state["key"],
        )
        linear = v59.run_root_sampled_uct(**kwargs)
        fast = run_root_sampled_uct_fast(**kwargs)
        self.assertEqual(linear.tree_sha256, fast.tree_sha256)
        self.assertEqual(linear.selected_action_key, fast.selected_action_key)
        self.assertEqual(linear.root_sample_counts, fast.root_sample_counts)

    def test_smc_conversion_preserves_marginals(self):
        configs = [
            canonical_json({"world": [["x", False]], "queue": []}),
            canonical_json({"world": [["x", True]], "queue": []}),
        ]
        pooled = {
            "atoms": [
                {"program_index": 0, "theta": 0.2, "configuration_key": configs[0], "weight": 0.2},
                {"program_index": 0, "theta": 0.2, "configuration_key": configs[0], "weight": 0.1},
                {"program_index": 1, "theta": 0.8, "configuration_key": configs[1], "weight": 0.7},
            ]
        }
        converted = smc2_atoms_for_planning(pooled)
        marginal = atom_marginals(converted, 2, 10, {"support": [0.05, 0.95]})
        expected = {
            "program": [0.3, 0.7],
            "theta_values": [0.2, 0.8],
            "theta_weights": [0.3, 0.7],
            "joint_bins": marginal["joint_bins"],
            "configuration": {configs[0]: 0.3, configs[1]: 0.7},
        }
        comparison = belief_comparison(expected, marginal)
        self.assertTrue(all(value <= 1e-15 for value in comparison.values()))
        self.assertEqual(len(converted), 2)

    def test_truth_conditioned_belief_rejected(self):
        with self.assertRaises(PermissionError):
            forbidden_truth_conditioned_belief({"target_program_index": 1})


if __name__ == "__main__":
    unittest.main()
