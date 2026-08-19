import json
import unittest
from pathlib import Path

from v126_sgd_retrieval_selectivity import joint_distribution, run_evaluation, truth_choice


class V126SelectivityTests(unittest.TestCase):
    def test_generic_eleven_hypothesis_evaluation_is_aggregate(self):
        config = json.loads(Path("configs/v126-sgd-retrieval-selectivity.json").read_text())
        catalog = json.loads(Path(config["choiceCatalog"]).read_text())
        baseline = json.loads(Path(config["baselineConfig"]).read_text())
        v119 = json.loads(Path(config["V119Config"]).read_text())
        known = next(row for row in catalog["choices"] if row["kind"] == "KNOWN")
        novel = next(row for row in catalog["choices"] if row["kind"] == "NOVEL_COMPOSITE")
        unsupported = next(row for row in catalog["choices"] if row["kind"] == "UNSUPPORTED_COMPOSITE")
        records = [
            {"record_id": "k", "class_label": "known", "domain": known["domain"], "service": known["service"], "intent": known["intent"]},
            {"record_id": "n", "class_label": "novel_valid", "domain": novel["domain"], "service": "Novel_1", "intent": "NovelIntent"},
            {"record_id": "u", "class_label": "unsupported", "domain": unsupported["domains"][0], "service": "Outside_1", "intent": "OutsideIntent"},
        ]
        retrieval = {
            "k": {"similarity": 0.9, "nearest_intent": known["intent_id"]},
            "n": {"similarity": 0.5, "nearest_intent": known["intent_id"]},
            "u": {"similarity": 0.2, "nearest_intent": known["intent_id"]},
        }
        by_id = {row["choice_id"]: row for row in catalog["choices"]}
        distribution = joint_distribution(truth_choice(records[0], catalog), known["choice_id"], 0.95, 0.5, by_id, v119)
        self.assertAlmostEqual(sum(distribution.values()), 1.0)
        result = run_evaluation(records, retrieval, catalog, baseline, v119, config)
        self.assertEqual(len(result["conditions"]), 9)
        self.assertEqual(result["individual_record_emission_count"], 0)
        self.assertEqual(result["threshold_fit_count"], 0)
        self.assertNotIn("records", result)


if __name__ == "__main__":
    unittest.main()
