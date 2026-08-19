from __future__ import annotations

import tempfile
from pathlib import Path
import unittest

from v216_bounded_external_artifact_population import (
    asserted_axioms,
    build_population_records,
    change_types,
    parse_obo_text,
    text_state,
    w3c_rdfxml_control,
)


FIXTURE = """format-version: 1.2

[Term]
id: UBERON:0000001
name: alpha structure
def: \"An alpha structure.\" [PMID:1]
synonym: \"alpha organ\" EXACT []
is_a: UBERON:0000100 ! parent

[Term]
id: UBERON:0000002
name: beta structure
def: \"A beta structure.\" []
relationship: part_of UBERON:0000100 ! parent
"""


class V216ExternalArtifactPopulationTests(unittest.TestCase):
    def test_obo_parser_separates_text_and_asserted_axioms(self) -> None:
        terms = parse_obo_text(FIXTURE)
        self.assertEqual({"UBERON:0000001", "UBERON:0000002"}, set(terms))
        self.assertEqual("An alpha structure.", text_state(terms["UBERON:0000001"])["definition"])
        self.assertEqual(
            ["is_a=UBERON:0000100"],
            asserted_axioms(terms["UBERON:0000001"], ["is_a", "relationship"]),
        )

    def test_change_types_detect_text_and_logic(self) -> None:
        older = parse_obo_text(FIXTURE)["UBERON:0000001"]
        newer_text = FIXTURE.replace("An alpha structure.", "A revised alpha structure.").replace(
            "UBERON:0000100 ! parent", "UBERON:0000200 ! parent", 1
        )
        newer = parse_obo_text(newer_text)["UBERON:0000001"]
        self.assertEqual(
            ["DEFINITION_CHANGED", "LOGICAL_AXIOM_CHANGED"],
            change_types(older, newer, ["is_a", "relationship"]),
        )

    def test_population_redacts_source_ids_and_preserves_version_space(self) -> None:
        older = parse_obo_text(FIXTURE)
        newer_text = FIXTURE.replace("An alpha structure.", "An alpha structure UBERON:0000999.")
        newer_text += """

[Term]
id: UBERON:0000003
name: gamma structure
def: \"A gamma structure.\" []
is_a: UBERON:0000100
"""
        newer = parse_obo_text(newer_text)
        config = {
            "experiment": "fixture",
            "parserDesign": {"logicalFields": ["is_a", "relationship", "intersection_of", "equivalent_to", "disjoint_from"]},
            "populationDesign": {
                "eligibleChangeTypes": ["ADDED", "DEFINITION_CHANGED", "LOGICAL_AXIOM_CHANGED"],
                "primaryChangePrecedence": ["ADDED", "LOGICAL_AXIOM_CHANGED", "DEFINITION_CHANGED"],
                "newerPayloadId": "new",
            },
        }
        public, truth, manifest = build_population_records(older, newer, config)
        self.assertEqual(2, len(public))
        self.assertNotIn("UBERON:0000999", str(public))
        self.assertTrue(all(record["candidate_class_ids"] for record in truth))
        self.assertEqual(2, manifest["eligible_record_count"])

    def test_w3c_control_counts_RDF_subjects(self) -> None:
        rdf = """<?xml version=\"1.0\"?>
<rdf:RDF xmlns:rdf=\"http://www.w3.org/1999/02/22-rdf-syntax-ns#\">
  <rdf:Description rdf:about=\"urn:a\" />
  <rdf:Description rdf:nodeID=\"b\" />
</rdf:RDF>"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "control.rdf"
            path.write_text(rdf)
            result = w3c_rdfxml_control(path)
        self.assertTrue(result["parse_success"])
        self.assertEqual(2, result["rdf_subject_count"])


if __name__ == "__main__":
    unittest.main()

