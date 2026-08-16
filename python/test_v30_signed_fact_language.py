import json
import unittest
from collections import Counter, defaultdict
from pathlib import Path

from audit_v30_signed_fact_language import audit
from generate_v30_signed_fact_language import build_records, corpus_hash
from v10_protocol import file_sha256
from v30_language import (
    candidate_nli_prompt, canonical_json, field_options, primary_field_prompt,
    select_option, v26_baseline_prompt,
)
from v30_evaluation import primary_summary, truth_summary
from v30_integration import assemble_scene_prediction
from v30_language import TRUTH_VALUES


class V30SignedFactLanguageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config_path = Path("configs/v30-signed-fact-language.json")
        cls.config = json.loads(cls.config_path.read_text())
        cls.rows = build_records(cls.config)

    def test_registered_population(self):
        self.assertEqual(len(self.rows), 910)
        self.assertEqual(Counter(row["split"] for row in self.rows), {
            "language_fit": 390,
            "language_calibration": 130,
            "language_evaluation": 390,
        })
        families = Counter(row["oracle_metadata"]["surface_family"] for row in self.rows)
        self.assertEqual(len(families), 35)
        self.assertEqual(set(families.values()), {26})
        self.assertEqual(len({row["scene_id"] for row in self.rows}), 280)

    def test_primary_prompt_is_candidate_independent(self):
        row = self.rows[0]
        for field in self.config["methods"]["primary"]["fields"]:
            prompt, options = primary_field_prompt(row, field, self.config)
            self.assertIn("Evidence statement:", prompt)
            self.assertNotIn("Candidate fact:", prompt)
            self.assertNotIn("candidate_statement", prompt)
            self.assertLessEqual(len(options), 6)
        self.assertIn("Candidate fact:", v26_baseline_prompt(row))
        self.assertIn("Candidate atom:", candidate_nli_prompt(row))

    def test_field_options_and_tie_break(self):
        row = next(value for value in self.rows if len(value["agent_input"]["entities"]) == 5)
        self.assertEqual(len(field_options(row, "predicate", self.config)), 5)
        self.assertEqual(len(field_options(row, "argument_1", self.config)), 5)
        self.assertEqual(len(field_options(row, "argument_2", self.config)), 6)
        options = field_options(row, "truth_status", self.config)
        self.assertEqual(select_option([0.2, 0.9, 0.1], options)["value"], "false")
        self.assertEqual(select_option([0.5, 0.5, 0.1], options)["value"], "true")

    def test_controlled_pairs_have_two_members(self):
        groups = defaultdict(list)
        for row in self.rows:
            for pair in row["oracle_metadata"]["pairs"]:
                groups[(pair["kind"], pair["id"])].append((row, pair))
        self.assertTrue(groups)
        self.assertEqual({len(members) for members in groups.values()}, {2})
        kinds = Counter(kind for kind, _ in groups)
        self.assertEqual(set(kinds), {
            "affirmative_negated", "argument_reversal", "distractor",
            "false_unknown", "inverse",
        })

    def test_in_memory_audit_passes(self):
        artifact_names = [f"{split}.jsonl" for split in self.config["splits"]]
        manifest = {
            "config_sha256": file_sha256(self.config_path),
            "corpus_sha256": corpus_hash(self.rows),
            "artifact_sha256": {},
        }
        # The in-memory audit checks on-disk hashes only when entries are registered.
        result = audit(
            self.rows, self.config, manifest, self.config_path.resolve(),
            enforce_pre_model_firewall=False,
        )
        self.assertTrue(result["passed"], result["errors"])
        self.assertEqual(result["semantic_audit"]["oracle_round_trip_accuracy"], 1.0)

    def test_oracle_predictions_pass_every_language_gate(self):
        predictions = []
        truth_predictions = []
        for row in self.rows:
            target = row["target"]
            selected = {
                "predicate": target["predicate"],
                "argument_1": target["arguments"][0],
                "argument_2": (
                    target["arguments"][1] if target["predicate_kind"] == "relation" else "N/A"
                ),
                "truth_status": target["truth_status"],
            }
            field_logits = {}
            option_rows = {}
            for field, value in selected.items():
                options = field_options(row, field, self.config)
                option_rows[field] = options
                field_logits[field] = {
                    option["token"]: (1.0 if option["value"] == value else 0.0)
                    for option in options
                }
            predictions.append({
                "id": row["id"], "selected_fields": selected,
                "field_logits": field_logits, "field_options": option_rows,
            })
            options = field_options(row, "truth_status", self.config)
            truth_predictions.append({
                "id": row["id"], "predicted_truth_status": target["truth_status"],
                "logits": {
                    option["token"]: (1.0 if option["value"] == target["truth_status"] else 0.0)
                    for option in options
                },
                "options": options,
            })
        primary = primary_summary(self.rows, predictions, self.config)
        truth = truth_summary(self.rows, truth_predictions, self.config)
        self.assertTrue(primary["passed"], primary["checks"])
        self.assertEqual(
            truth["by_split"]["language_evaluation"]["truth_status_accuracy"], 1.0
        )

    def test_deterministic_public_candidate_alignment(self):
        direct = [
            row for row in self.rows
            if row["oracle_metadata"]["surface_family"] == "affirmative_gold.fit_a"
            and row["oracle_metadata"]["base_scene_index"] == 0
            and row["oracle_metadata"]["scene_variant"] == "direct_clean"
        ]
        self.assertEqual(len(direct), 5)
        scene = {
            "id": direct[0]["scene_id"], "episode_id": "episode_test",
            "split": direct[0]["split"], "role": "support",
            "agent_input": {
                "entities": direct[0]["agent_input"]["entities"],
                "evidence": [
                    {"id": row["id"], "text": row["agent_input"]["evidence_text"]}
                    for row in reversed(direct)
                ],
                "atom_candidates": [
                    {"id": f"candidate_{index}", "statement": row["target"]["candidate_statement"]}
                    for index, row in enumerate(direct)
                ],
            },
        }
        statement_to_candidate = {
            row["statement"]: row["id"] for row in scene["agent_input"]["atom_candidates"]
        }
        scores = []
        for row in direct:
            target = row["target"]
            selected = {
                "predicate": target["predicate"],
                "argument_1": target["arguments"][0],
                "argument_2": (
                    target["arguments"][1] if target["predicate_kind"] == "relation" else "N/A"
                ),
                "truth_status": target["truth_status"],
            }
            field_logits, option_rows = {}, {}
            for field, value in selected.items():
                options = field_options(row, field, self.config)
                option_rows[field] = options
                field_logits[field] = {
                    option["token"]: (8.0 if option["value"] == value else -8.0)
                    for option in options
                }
            scores.append({
                "evidence_id": row["id"], "selected_fields": selected,
                "field_logits": field_logits, "field_options": option_rows,
            })
        prediction = assemble_scene_prediction(scene, scores, self.config)
        self.assertEqual(len({row["candidate_id"] for row in prediction["rows"]}), 5)
        expected = {
            row["id"]: (
                statement_to_candidate[row["target"]["candidate_statement"]],
                row["target"]["truth_status"],
            )
            for row in direct
        }
        self.assertEqual({
            row["evidence_id"]: (row["candidate_id"], row["truth_label"])
            for row in prediction["rows"]
        }, expected)
        self.assertEqual(
            {tuple(row["allowed_values"]) for row in prediction["epistemic_state"]},
            {tuple(TRUTH_VALUES["true"]), tuple(TRUTH_VALUES["false"])},
        )


if __name__ == "__main__":
    unittest.main()
