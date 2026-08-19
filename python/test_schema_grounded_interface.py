#!/usr/bin/env python3
from __future__ import annotations

import json
import unittest

from schema_grounded_interface import (
    FINITE_GRAMMAR_STYLES,
    ClarificationRequest,
    SchemaBoundaryError,
    canonical_schema_surface,
    certify_schema_surface,
    compile_schema_registry,
    decorate_v79_policy_node,
    inspect_untrusted_schema_surface,
    invalid_request_population,
    invalid_schema_mutations,
    parse_clarification_request,
    render_schema_clarification,
    unsafe_schema_surface_mutations,
)
from v22r2_grounding import PROJECT_ROOT


class SchemaGroundedInterfaceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        lock = json.loads(
            (PROJECT_ROOT / "configs/v84-schema-grounded-shadow-design-lock.json").read_text()
        )
        cls.raw_schemas = lock["config_payload"]["schemas"]
        cls.registry = compile_schema_registry(cls.raw_schemas)

    def test_all_schema_targets_render_in_every_model_free_mode(self) -> None:
        for schema in self.registry.schemas:
            requests = [
                ClarificationRequest(schema.schema_id, "slot", slot.slot_id)
                for slot in schema.slots
            ] + [ClarificationRequest(schema.schema_id, "all", None)]
            for request in requests:
                self.assertTrue(render_schema_clarification(self.registry, request).certificate.deployable)
                for style in FINITE_GRAMMAR_STYLES:
                    rendered = render_schema_clarification(
                        self.registry, request, source="finite_grammar", style=style
                    )
                    self.assertTrue(rendered.certificate.deployable)
                    self.assertEqual(rendered.typed_request, request)

    def test_invalid_schema_population_is_rejected_completely(self) -> None:
        for name, population in invalid_schema_mutations(self.raw_schemas):
            with self.assertRaises(SchemaBoundaryError, msg=name):
                compile_schema_registry(population)

    def test_invalid_requests_fail_closed(self) -> None:
        for raw in invalid_request_population(self.registry):
            try:
                request = parse_clarification_request(raw)
                canonical_schema_surface(self.registry, request)
            except SchemaBoundaryError:
                continue
            self.fail(f"invalid request did not fail closed: {raw}")

    def test_unsafe_surface_mutations_are_rejected(self) -> None:
        for request, question in unsafe_schema_surface_mutations(self.registry):
            certificate = certify_schema_surface(
                self.registry, request, question, "canonical"
            )
            self.assertFalse(certificate.content_valid, (request, question))
            self.assertFalse(certificate.deployable)

    def test_valid_looking_untrusted_surface_remains_non_deployable(self) -> None:
        for schema in self.registry.schemas:
            request = ClarificationRequest(schema.schema_id, "all", None)
            question = canonical_schema_surface(self.registry, request)
            certificate = inspect_untrusted_schema_surface(
                self.registry, request, question
            )
            self.assertTrue(certificate.content_valid)
            self.assertFalse(certificate.source_authorized)
            self.assertFalse(certificate.deployable)

    def test_disabled_renderer_sources_raise(self) -> None:
        request = ClarificationRequest("project_workflow", "slot", "operation")
        for source in ("local_model", "API_model", "adapter_model", "untrusted_passthrough"):
            with self.assertRaises(PermissionError):
                render_schema_clarification(self.registry, request, source=source)

    def test_V79_bridge_preserves_authoritative_fields(self) -> None:
        node = {"action": "ask_operation", "history": [], "hypothesis_masses": [0.2] * 5}
        decorated = decorate_v79_policy_node(node, self.registry)
        self.assertEqual(node, {key: decorated[key] for key in node})
        self.assertEqual(decorated["action"], "ask_operation")
        self.assertNotIn("schema_clarification_surface", node)
        nonask = {"action": "safe_preview", "history": [], "hypothesis_masses": [0.2] * 5}
        self.assertEqual(decorate_v79_policy_node(nonask, self.registry), nonask)


if __name__ == "__main__":
    unittest.main()
