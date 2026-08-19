from __future__ import annotations

import unittest

from v106_open_world_benchmark import (
    build_declared_training_records, decision_cost, evaluate_predictions,
    identifier_grammar_prediction, oracle_prediction, retrieval_prediction,
    split_development_records, tune_retrieval_thresholds,
)


CONFIG = {
    "developmentSplit": {
        "salt": "test", "classes": ["known_familiar", "known_unfamiliar", "novel_valid", "unsupported"],
        "calibrationCountPerClass": 1, "evaluationCountPerClass": 1,
    },
    "decisionCosts": {
        "known": {"exact_known": 0.0, "wrong_known": 10.0, "novel": 3.0, "unsupported": 6.0, "abstain": 1.0},
        "novel": {"known": 10.0, "exact_novel": 0.0, "wrong_novel_scenario": 5.0, "unsupported": 4.0, "abstain": 1.5},
        "unsupported": {"known": 10.0, "novel": 4.0, "unsupported": 0.0, "abstain": 1.0},
        "insufficient": {"known": 10.0, "novel": 6.0, "unsupported": 4.0, "abstain": 0.0},
    },
    "deterministicBaselines": {"character_ngram_retrieval": {
        "knownThresholdGrid": [0.4, 0.6], "unsupportedThresholdGrid": [0.1, 0.3],
    }},
}


def row(identifier: str, class_label: str, utterance: str = "") -> dict:
    mapping = {
        "known_familiar": ("calendar", "calendar_set"),
        "known_unfamiliar": ("calendar", "calendar_remove"),
        "novel_valid": ("calendar", "calendar_query"),
        "unsupported": ("email", "email_sendemail"),
    }
    scenario, intent = mapping[class_label]
    return {
        "record_id": identifier, "role": "development", "class_label": class_label,
        "scenario": scenario, "intent": intent, "utterance": utterance,
    }


class V106BenchmarkTests(unittest.TestCase):
    def test_hash_split_is_balanced_and_disjoint(self) -> None:
        records = [row(f"{label}-{index}", label) for label in CONFIG["developmentSplit"]["classes"] for index in range(2)]
        split = split_development_records(records, CONFIG)
        self.assertEqual(split["counts"], {"calibration": 4, "evaluation": 4})
        self.assertFalse({r["record_id"] for r in split["calibration"]} & {r["record_id"] for r in split["evaluation"]})

    def test_training_records_include_declared_train_only(self) -> None:
        source = [
            {"id": 1, "partition": "train", "scenario": "calendar", "intent": "calendar_set", "utt": "set meeting"},
            {"id": 2, "partition": "validation", "scenario": "calendar", "intent": "calendar_set", "utt": "ignored"},
            {"id": 3, "partition": "train", "scenario": "calendar", "intent": "calendar_query", "utt": "hidden"},
        ]
        catalog = {"intents": [{"intent_id": "calendar::calendar_set"}]}
        selected = build_declared_training_records(source, catalog)
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0]["source_id"], "1")

    def test_grammar_and_cost_distinguish_wrong_known(self) -> None:
        catalog = {
            "scenarios": ["calendar"],
            "intents": [{"intent_id": "calendar::calendar_set", "intent": "calendar_set"}],
        }
        record = row("k", "known_familiar", "set calendar")
        predicted = identifier_grammar_prediction(record, catalog)
        self.assertEqual(predicted["status"], "KNOWN")
        self.assertEqual(decision_cost(record, predicted, CONFIG), 0.0)
        wrong = dict(predicted, known_intent="calendar::calendar_remove")
        self.assertEqual(decision_cost(record, wrong, CONFIG), 10.0)

    def test_threshold_tuning_and_metrics_are_deterministic(self) -> None:
        records = [
            row("k", "known_familiar"), row("u", "known_unfamiliar"),
            row("n", "novel_valid"), row("x", "unsupported"),
        ]
        observations = {
            "k": {"similarity": 0.8, "nearest_intent": "calendar::calendar_set", "nearest_scenario": "calendar"},
            "u": {"similarity": 0.7, "nearest_intent": "calendar::calendar_remove", "nearest_scenario": "calendar"},
            "n": {"similarity": 0.35, "nearest_intent": "calendar::calendar_set", "nearest_scenario": "calendar"},
            "x": {"similarity": 0.05, "nearest_intent": "calendar::calendar_set", "nearest_scenario": "calendar"},
        }
        tuned = tune_retrieval_thresholds(records, observations, CONFIG)
        self.assertEqual(tuned["selected"]["mean_regret"], 0.0)
        selected = tuned["selected"]
        predictions = {
            r["record_id"]: retrieval_prediction(
                observations[r["record_id"]], selected["known_threshold"], selected["unsupported_threshold"],
            ) for r in records
        }
        metrics = evaluate_predictions(records, predictions, CONFIG)
        self.assertEqual(metrics["observed_exact_decision_accuracy"], 1.0)
        self.assertEqual(metrics["mean_regret"], 0.0)

    def test_oracle_is_exact(self) -> None:
        records = [row("k", "known_familiar"), row("n", "novel_valid"), row("x", "unsupported")]
        metrics = evaluate_predictions(records, {r["record_id"]: oracle_prediction(r) for r in records}, CONFIG)
        self.assertEqual(metrics["observed_exact_decision_accuracy"], 1.0)
        self.assertEqual(metrics["mean_regret"], 0.0)


if __name__ == "__main__":
    unittest.main()
