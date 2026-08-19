import unittest

from v133_sgd_capability_label_identifiability import compare_definitions, definition_record, normalize_name


class V133CapabilityIdentifiabilityTests(unittest.TestCase):
    def test_name_normalization(self):
        self.assertEqual(normalize_name("Find_Movies"), normalize_name("find movies"))

    def test_exact_definition_collision(self):
        intent = {"name": "FindMovies", "description": "Find a movie.", "required_slots": ["title"], "optional_slots": []}
        left = definition_record("Movies_1", "movies", intent)
        right = definition_record("Movies_3", "movies", dict(intent))
        relation = compare_definitions(left, right)
        self.assertTrue(all(relation.values()), relation)

    def test_name_collision_does_not_require_full_collision(self):
        left = definition_record("Movies_1", "movies", {"name": "FindMovies", "description": "Find one.", "required_slots": [], "optional_slots": []})
        right = definition_record("Movies_3", "movies", {"name": "FindMovies", "description": "Rent one.", "required_slots": ["title"], "optional_slots": []})
        relation = compare_definitions(left, right)
        self.assertTrue(relation["exact_name"])
        self.assertFalse(relation["exact_full_signature"])


if __name__ == "__main__":
    unittest.main()
