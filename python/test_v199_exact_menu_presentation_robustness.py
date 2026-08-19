from __future__ import annotations

import json
import unittest

from v199_exact_menu_presentation_robustness import audit_transformation_family, build_transformation_family
from v22r2_grounding import PROJECT_ROOT


class V199TransformationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = json.loads((PROJECT_ROOT / "configs/v199-exact-menu-presentation-robustness.json").read_text())
        cls.inputs = [
            json.loads((PROJECT_ROOT / cls.config[key]).read_text())
            for key in ("developmentIdentities", "hiddenTargets", "canonicalVisibleMenu", "canonicalHiddenOptionMap")
        ]
        cls.family = build_transformation_family(*cls.inputs, cls.config)

    def test_family_is_deterministic_and_passes(self) -> None:
        self.assertEqual(self.family, build_transformation_family(*self.inputs, self.config))
        self.assertTrue(audit_transformation_family(self.family, self.config)["passed"])

    def test_visible_artifact_contains_no_targets_or_language(self) -> None:
        encoded = json.dumps(self.family["visible_variants"], sort_keys=True)
        for forbidden in (
            "capability_contract_id", "target_contract_id", "truth_kind", "conversation", "utterance",
            "source_dialogue_id", "source_candidate_id",
        ):
            self.assertNotIn(forbidden, encoded)

    def test_opaque_ids_are_record_specific_bijections(self) -> None:
        hidden = self.family["hidden_variant_maps"]["records"]
        opaque = [
            next(variant for variant in record["variants"] if variant["variant_id"] == "ORDER_AND_OPAQUE_ID")
            for record in hidden
        ]
        expected = {f"Q{index:02d}" for index in range(1, 15)}
        self.assertTrue(all({row["option_id"] for row in variant["mappings"]} == expected for variant in opaque))
        signatures = {
            tuple(sorted((row["option_id"], row["capability_contract_id"]) for row in variant["mappings"]))
            for variant in opaque
        }
        self.assertGreaterEqual(len(signatures), 90)

    def test_future_gates_are_prospectively_nontrivial(self) -> None:
        gates = self.config["futurePairedDevelopmentGates"]
        self.assertGreater(gates["canonicalPrimaryTop3Recall"], 0.9)
        self.assertLessEqual(gates["maximumPerVariantPrimaryTop3RecallDrop"], 0.05)
        self.assertGreaterEqual(gates["minimumPerVariantTop1ContractAgreementWithCanonical"], 0.8)
        self.assertGreaterEqual(gates["minimumPerVariantMeanTop3ContractSetJaccardWithCanonical"], 0.8)


if __name__ == "__main__":
    unittest.main()
