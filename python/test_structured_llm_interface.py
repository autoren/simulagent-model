#!/usr/bin/env python3
from __future__ import annotations

import unittest

from structured_llm_interface import (
    CANONICAL_SURFACES,
    CLARIFICATION_CODES,
    FINITE_GRAMMAR_STYLES,
    certify_surface,
    decorate_policy_node,
    inspect_untrusted_surface,
    render_clarification,
    unsafe_surface_mutations,
)


class StrictStructuredLLMInterfaceTests(unittest.TestCase):
    def test_every_model_free_surface_is_strict_and_deployable(self) -> None:
        for code in CLARIFICATION_CODES:
            canonical = render_clarification(code)
            self.assertTrue(canonical.certificate.content_valid)
            self.assertTrue(canonical.certificate.source_authorized)
            self.assertTrue(canonical.certificate.deployable)
            for style in FINITE_GRAMMAR_STYLES:
                rendered = render_clarification(
                    code, source="finite_grammar", style=style
                )
                self.assertTrue(rendered.certificate.deployable)
                self.assertEqual(rendered.action_code, code)

    def test_every_locked_unsafe_mutation_is_rejected(self) -> None:
        for code, question in unsafe_surface_mutations():
            certificate = certify_surface(code, question, "canonical")
            self.assertFalse(certificate.content_valid, (code, question))
            self.assertFalse(certificate.deployable)

    def test_generated_API_adapter_and_passthrough_sources_are_disabled(self) -> None:
        for source in (
            "local_model",
            "API_model",
            "adapter_model",
            "untrusted_passthrough",
        ):
            with self.assertRaises(PermissionError):
                render_clarification("ask_operation", source=source)
        certificate = inspect_untrusted_surface(
            "ask_operation", CANONICAL_SURFACES["ask_operation"]
        )
        self.assertTrue(certificate.content_valid)
        self.assertFalse(certificate.source_authorized)
        self.assertFalse(certificate.deployable)

    def test_policy_decision_fields_are_preserved(self) -> None:
        node = {
            "action": "ask_recipient",
            "history": ["operation_other"],
            "hypothesis_masses": [0.1, 0.2, 0.3, 0.2, 0.2],
        }
        decorated = decorate_policy_node(node)
        self.assertEqual(node, {key: decorated[key] for key in node})
        self.assertEqual(decorated["action"], node["action"])
        self.assertEqual(
            decorated["clarification_surface"]["action_code"], node["action"]
        )
        self.assertTrue(
            decorated["clarification_surface"]["certificate"]["deployable"]
        )
        self.assertNotIn("clarification_surface", node)

    def test_nonclarification_nodes_are_structurally_identical(self) -> None:
        for action in ("execute_schedule_chen", "safe_preview", "abstain"):
            node = {"action": action, "history": [], "hypothesis_masses": [1.0]}
            self.assertEqual(decorate_policy_node(node), node)

    def test_none_or_execution_codes_cannot_be_rendered(self) -> None:
        for code in ("none_of_the_above", "execute_schedule_chen", "safe_preview"):
            with self.assertRaises(ValueError):
                render_clarification(code)


if __name__ == "__main__":
    unittest.main()
