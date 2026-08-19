#!/usr/bin/env python3
from __future__ import annotations

import json
import unittest

from schema_grounded_interface import (
    FINITE_GRAMMAR_STYLES,
    ClarificationRequest,
    compile_schema_registry,
    unsafe_schema_surface_mutations,
)
from schema_grounded_interface_v86 import (
    decorate_v79_policy_node_v86,
    hardened_certify_schema_surface,
    inspect_untrusted_hardened_surface,
    partial_option_injection_mutations,
    render_hardened_schema_clarification,
)
from v22r2_grounding import PROJECT_ROOT


class HardenedSchemaGroundedInterfaceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        lock = json.loads((PROJECT_ROOT / "configs/v84-schema-grounded-shadow-design-lock.json").read_text())
        cls.registry = compile_schema_registry(lock["config_payload"]["schemas"])

    def test_all_safe_surfaces_remain_deployable(self) -> None:
        for schema in self.registry.schemas:
            requests = [ClarificationRequest(schema.schema_id, "slot", slot.slot_id) for slot in schema.slots]
            requests.append(ClarificationRequest(schema.schema_id, "all", None))
            for request in requests:
                self.assertTrue(render_hardened_schema_clarification(self.registry, request).certificate.deployable)
                for style in FINITE_GRAMMAR_STYLES:
                    self.assertTrue(render_hardened_schema_clarification(self.registry, request, source="finite_grammar", style=style).certificate.deployable)

    def test_all_partial_option_injections_are_rejected(self) -> None:
        rows = partial_option_injection_mutations(self.registry)
        self.assertEqual(len(rows), 16)
        for request, question, injected in rows:
            certificate = hardened_certify_schema_surface(self.registry, request, question, "canonical")
            self.assertGreater(certificate.individual_unrequested_option_surface_count, 0, injected)
            self.assertFalse(certificate.content_valid)
            self.assertFalse(certificate.deployable)

    def test_original_unsafe_population_remains_rejected(self) -> None:
        for request, question in unsafe_schema_surface_mutations(self.registry):
            self.assertFalse(hardened_certify_schema_surface(self.registry, request, question, "canonical").deployable)

    def test_exact_V85_false_positive_is_rejected(self) -> None:
        request = ClarificationRequest("project_workflow", "slot", "operation")
        question = "Should I schedule the project review or send the project summary, and should the recipient be Alex Chen?"
        certificate = hardened_certify_schema_surface(self.registry, request, question, "local_model_adversarial")
        self.assertFalse(certificate.content_valid)
        self.assertEqual(certificate.individual_unrequested_option_surface_count, 1)
        self.assertFalse(certificate.deployable)

    def test_untrusted_valid_surface_remains_non_deployable(self) -> None:
        request = ClarificationRequest("project_workflow", "all", None)
        rendered = render_hardened_schema_clarification(self.registry, request)
        certificate = inspect_untrusted_hardened_surface(self.registry, request, rendered.question)
        self.assertTrue(certificate.content_valid)
        self.assertFalse(certificate.source_authorized)
        self.assertFalse(certificate.deployable)

    def test_V79_bridge_preserves_fields(self) -> None:
        node = {"action": "ask_operation", "history": [], "hypothesis_masses": [0.2] * 5}
        decorated = decorate_v79_policy_node_v86(node, self.registry)
        self.assertEqual(node, {key: decorated[key] for key in node})
        self.assertEqual(decorated["action"], node["action"])


if __name__ == "__main__":
    unittest.main()
