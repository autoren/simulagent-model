from __future__ import annotations

import json
import unittest

from v22r2_grounding import PROJECT_ROOT
from v166_model_free_factored_ontology_baselines import (
    BASELINE_NAMES,
    build_predictions,
    evaluate_gates,
    evaluate_predictions,
    evidence_status,
)


class V166ModelFreeFactoredOntologyBaselinesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        outcome = json.loads(
            (
                PROJECT_ROOT
                / "configs/v165r1-outcome-verifier-repair-outcome-lock.json"
            ).read_text()
        )
        cls.public = json.loads((PROJECT_ROOT / outcome["public_records"]).read_text())
        cls.hidden = json.loads((PROJECT_ROOT / outcome["hidden_records"]).read_text())
        cls.ontology = json.loads((PROJECT_ROOT / outcome["frozen_ontology"]).read_text())
        cls.config = json.loads(
            (
                PROJECT_ROOT
                / "configs/v166-model-free-factored-ontology-baselines.json"
            ).read_text()
        )
        cls.predictions = build_predictions(cls.public, cls.hidden, cls.ontology)
        cls.evaluation = evaluate_predictions(cls.predictions, cls.hidden)

    def test_cardinality_status_contract(self) -> None:
        self.assertEqual(evidence_status([]), "contradictory")
        self.assertEqual(evidence_status(["C001"]), "sufficient")
        self.assertEqual(evidence_status(["C001", "C002"]), "ambiguous")

    def test_all_registered_baselines_are_present(self) -> None:
        self.assertEqual(
            set(self.predictions[0]["predictions"]), set(BASELINE_NAMES)
        )

    def test_combined_and_oracle_are_exact(self) -> None:
        for name in ("exact_parser_plus_version_space", "oracle_hidden_contract"):
            metrics = self.evaluation["baseline_metrics"][name]
            self.assertEqual(metrics["exact_version_space_accuracy"], 1.0)
            self.assertEqual(metrics["evidence_status_accuracy"], 1.0)

    def test_ambiguity_is_retained_not_forced(self) -> None:
        self.assertEqual(
            self.evaluation["intentionally_ambiguous_record_count"], 48
        )
        self.assertEqual(
            set(self.evaluation["intentionally_ambiguous_candidate_counts"]), {64}
        )
        self.assertEqual(self.evaluation["model_eligible_residual_count"], 0)

    def test_all_preregistered_gates_pass(self) -> None:
        access = {
            "evaluation_record_count": 0,
            "manual_judgment_count": 0,
            "model_load_count": 0,
            "model_generation_count": 0,
            "API_call_count": 0,
            "training_run_count": 0,
            "ontology_registration_count": 0,
            "real_service_call_count": 0,
            "external_side_effect_count": 0,
            "actual_execution_count": 0,
        }
        self.assertTrue(all(evaluate_gates(self.evaluation, access, self.config).values()))


if __name__ == "__main__":
    unittest.main()
