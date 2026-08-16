from decimal import Decimal
from fractions import Fraction
import json
import unittest

from v42_stateful import deterministic_world, entities, epistemic_rows
from v49_belief import (
    conditional_suffix_distribution,
    full_evidence,
    masked_trace,
    mechanic_registry,
    query_predictive,
    support_posterior,
    trajectory_map,
)


def fixture():
    registry = mechanic_registry()
    mechanic = registry[0]
    entity_rows = entities(2)
    world = deterministic_world(entity_rows, "v49-test")
    actions = [
        {"id": "pulse", "binding": {"actor": "unit_0", "target": "unit_1"}},
        {"id": "route", "binding": {"actor": "unit_0", "target": "unit_1"}},
        {"id": "wait", "binding": {}},
    ]
    return registry, mechanic, entity_rows, world, actions


class V49BeliefTests(unittest.TestCase):
    def test_registry_is_fresh_balanced_and_unique(self):
        registry = mechanic_registry()
        self.assertEqual(len(registry), 48)
        self.assertEqual(len({row["key"] for row in registry}), 48)
        self.assertEqual({row["probability"] for row in registry}, {"1/4", "1/2", "3/4"})
        self.assertTrue(all(
            sum(row["family"] == family for row in registry) == 12
            for family in {row["family"] for row in registry}
        ))

    def test_masked_likelihood_and_query_conditioning_normalize(self):
        registry, mechanic, entity_rows, world, actions = fixture()
        distribution = trajectory_map(mechanic["program"], entity_rows, world, actions)
        trajectory = next(iter(distribution))
        decoded = json.loads(trajectory)
        atoms = sorted(world)
        masks = [atoms[::2] for _ in actions]
        trace = masked_trace(decoded, masks)
        evidence_mass, truth = conditional_suffix_distribution(
            mechanic["program"], entity_rows, world, actions, trace[:1], 1
        )
        self.assertGreater(evidence_mass, 0)
        self.assertEqual(sum(truth.values(), Fraction(0)), 1)

        support = {
            "entities": entity_rows,
            "initial_state": epistemic_rows(world),
            "actions": actions,
            "masks": masks,
            "full_trajectory_catalog": {"full": decoded},
            "masked_trace_catalog": {"masked": trace},
            "realized_full_trajectory_ids": ["full"],
            "realized_masked_trace_ids": ["masked"],
        }
        weights = support_posterior(registry, [support])
        self.assertLess(abs(sum(weights, Decimal(0)) - 1), Decimal("1e-80"))
        predictive, query_weights, _ = query_predictive(
            registry, weights, entity_rows, world, actions, trace[:1], 1
        )
        self.assertLess(abs(sum(predictive.values(), Decimal(0)) - 1), Decimal("1e-80"))
        self.assertLess(abs(sum(query_weights, Decimal(0)) - 1), Decimal("1e-80"))
        self.assertEqual(len(full_evidence(decoded, 1)[0]), len(world))


if __name__ == "__main__":
    unittest.main()
