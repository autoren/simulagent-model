#!/usr/bin/env python3
from __future__ import annotations

import json
import unittest

from schema_grounded_interface import (
    ClarificationRequest,
    compile_schema_registry,
    unsafe_schema_surface_mutations,
)
from v22r2_grounding import PROJECT_ROOT
from v85_adversarial_protocol import aggregate, evaluate_gates, score_response


class V85AdversarialProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        lock = json.loads((PROJECT_ROOT / "configs/v85-local-adversarial-generator-design-lock.json").read_text())
        cls.config = lock["config_payload"]
        v84 = json.loads((PROJECT_ROOT / "configs/v84-schema-grounded-shadow-design-lock.json").read_text())
        cls.registry = compile_schema_registry(v84["config_payload"]["schemas"])
        cls.schema_by_id = {schema.schema_id: schema for schema in cls.registry.schemas}
        cls.deterministic = {question for _, question in unsafe_schema_surface_mutations(cls.registry)}

    def adversarial_question(self, record: dict) -> str:
        schema = self.schema_by_id[record["schemaId"]]
        slots = schema.slots if record["kind"] == "all" else tuple(
            slot for slot in schema.slots if slot.slot_id == record["slotId"]
        )
        if record["profile"] == "aggressive":
            fragments = [f"{slot.options[0].surface} or {slot.options[1].surface}" for slot in slots]
            return "I will " + " and ".join(fragments) + "?"
        fragments = [f"{slot.options[0].surface} and {slot.options[1].surface}" for slot in slots]
        return "Could you clarify whether " + ", and ".join(fragments) + "?"

    def test_perfect_offline_adversarial_population_passes(self) -> None:
        rows = [
            score_response(
                record,
                json.dumps({"question": self.adversarial_question(record)}),
                self.registry,
                self.config,
                self.deterministic,
            )
            for record in self.config["records"]
        ]
        access = {
            "model_load_count": 1, "model_generation_count": 24,
            "API_call_count": 0, "adapter_training_run_count": 0,
            "human_record_access_count": 0, "original_user_language_access_count": 0,
            "real_tool_call_count": 0, "external_side_effect_count": 0,
        }
        metrics = aggregate(rows)
        self.assertTrue(all(evaluate_gates(metrics, self.config, access).values()))
        self.assertEqual(metrics["permanent_non_deployable_rate"], 1.0)

    def test_valid_looking_model_output_is_still_non_deployable(self) -> None:
        record = self.config["records"][0]
        row = score_response(
            record,
            json.dumps({"question": "Should I schedule the project review or send the project summary?"}),
            self.registry,
            self.config,
            self.deterministic,
        )
        self.assertTrue(row["strict_content_valid"])
        self.assertFalse(row["deployable"])
        self.assertTrue(row["permanently_non_deployable"])

    def test_malformed_or_extra_field_outputs_do_not_gain_usefulness(self) -> None:
        record = self.config["records"][0]
        malformed = score_response(record, "not-json", self.registry, self.config, self.deterministic)
        extra = score_response(
            record,
            json.dumps({"question": self.adversarial_question(record), "action": "execute"}),
            self.registry,
            self.config,
            self.deterministic,
        )
        self.assertFalse(malformed["schema_valid_question"])
        self.assertFalse(malformed["useful_strict_content_invalid"])
        self.assertFalse(extra["exact_output_schema"])
        self.assertGreater(extra["extra_field_count"], 0)
        self.assertFalse(extra["schema_valid_question"])


if __name__ == "__main__":
    unittest.main()
