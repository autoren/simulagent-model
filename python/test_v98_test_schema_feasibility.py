from __future__ import annotations

import unittest

from v98_test_schema_feasibility import build_test_schema_inventory, service_family


def service(name: str, intent_count: int = 2, slot_count: int = 1) -> dict:
    return {
        "service_name": name,
        "intents": [
            {"name": f"Intent{index}", "description": "private description"}
            for index in range(intent_count)
        ],
        "slots": [
            {"name": f"slot{index}", "description": "private slot"}
            for index in range(slot_count)
        ],
    }


CONFIG = {
    "familyRule": {
        "minimumTypedIntentCountPerEligibleService": 2,
        "minimumSlotCountPerEligibleService": 1,
    }
}


class V98TestSchemaFeasibilityTests(unittest.TestCase):
    def test_family_ignores_numeric_version(self) -> None:
        self.assertEqual(service_family("Banks_2"), "Banks")
        with self.assertRaises(ValueError):
            service_family("Banks")

    def test_only_new_typed_families_are_eligible(self) -> None:
        development = [service("Banks_2"), service("Hotels_1")]
        test = [
            service("Banks_1"), service("Messaging_1"), service("Payments_1"),
            service("Trains_1"), service("Tiny_1", intent_count=1),
        ]
        result = build_test_schema_inventory(development, test, CONFIG)
        self.assertEqual(result["novel_service_families"], ["Messaging", "Payments", "Trains"])
        self.assertEqual(result["eligible_novel_service_count"], 3)

    def test_inventory_is_structural_and_language_free(self) -> None:
        result = build_test_schema_inventory(
            [service("Banks_2")], [service("Messaging_1")], CONFIG
        )
        serialized = str(result)
        self.assertNotIn("private description", serialized)
        self.assertNotIn("private slot", serialized)
        self.assertEqual(result["emitted_intent_name_count"], 0)
        self.assertEqual(result["emitted_slot_name_count"], 0)
        self.assertFalse(result["contains_schema_language_or_surface_tokens"])


if __name__ == "__main__":
    unittest.main()
