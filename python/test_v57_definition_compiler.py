#!/usr/bin/env python3
from __future__ import annotations

import unittest

from v57_definition_compiler import (
    compile_agent_input, compiled_truth, render_controlled_definition,
)


def concept(identifier, kind, signature, forms, family="signature_first"):
    return {
        "opaque_id": identifier,
        "kind": kind,
        "typed_signature": signature,
        "controlled_definition": render_controlled_definition(
            identifier, kind, signature, forms, family
        ),
        "positive_or_command_form": forms[
            "command" if kind == "bound_action" else (
                "positive" if kind == "unary_predicate" else "direct_positive"
            )
        ],
        "lexical_forms": forms,
        "definition_template_family": family,
    }


class DefinitionCompilerTests(unittest.TestCase):
    def setUp(self):
        self.entities = [
            {"id": "carrier_0", "entity_type": "carrier"},
            {"id": "carrier_1", "entity_type": "carrier"},
            {"id": "station_0", "entity_type": "station"},
        ]
        self.unary = concept(
            "p_a1", "unary_predicate", {"entity": "carrier"},
            {"positive": "{entity} is glinting", "negative": "{entity} is not glinting"},
        )
        self.relation = concept(
            "r_b2", "binary_relation", {"source": "carrier", "target": "carrier"},
            {
                "direct_positive": "{source} nexes {target}",
                "direct_negative": "{source} does not nex {target}",
                "inverse_positive": "{target} is nexed by {source}",
                "inverse_negative": "{target} is not nexed by {source}",
            }, "meaning_first",
        )
        self.action = concept(
            "a_c3", "bound_action", {"actor": "carrier", "target": "station"},
            {"command": "send {actor} toward {target}"}, "example_first",
        )

    def compile(self, text, definitions=None, mutation=None):
        return compile_agent_input({
            "entities": self.entities,
            "concept_definitions": (
                [self.unary, self.relation, self.action]
                if definitions is None else definitions
            ),
            "evidence_text": text,
        }, mutation)

    def test_unary_sign_and_truth(self):
        result = self.compile(
            "Focal report: carrier_0 is not glinting; Operation cue: support; "
            "Context only: maintenance note."
        )
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["parse"]["symbol"], "p_a1")
        self.assertEqual(result["parse"]["lexical_sign"], "negative")
        self.assertEqual(compiled_truth(result["parse"]), "false")

    def test_relation_inverse_preserves_canonical_roles(self):
        result = self.compile(
            "Focal report: carrier_1 is nexed by carrier_0; Operation cue: endorsement; "
            "Context only: maintenance note."
        )
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["parse"]["arguments"], ["carrier_0", "carrier_1"])
        self.assertEqual(result["parse"]["orientation"], "inverse")

    def test_bound_action(self):
        result = self.compile("Action request: send carrier_0 toward station_0.")
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["parse"]["kind"], "bound_action")
        self.assertEqual(result["parse"]["arguments"], ["carrier_0", "station_0"])

    def test_missing_definitions_abstains(self):
        result = self.compile("Action request: send carrier_0 toward station_0.", [])
        self.assertEqual(result["status"], "abstain")

    def test_type_mismatch_abstains(self):
        result = self.compile("Action request: send station_0 toward carrier_0.")
        self.assertEqual(result["status"], "abstain")

    def test_duplicate_lexeme_is_ambiguous(self):
        duplicate = concept(
            "p_z9", "unary_predicate", {"entity": "carrier"},
            dict(self.unary["lexical_forms"]), "example_first",
        )
        result = self.compile(
            "Focal report: carrier_0 is glinting; Operation cue: support; "
            "Context only: maintenance note.",
            [self.unary, duplicate],
        )
        self.assertEqual(result["status"], "ambiguous")


if __name__ == "__main__":
    unittest.main()
