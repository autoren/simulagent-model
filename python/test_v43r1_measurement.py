import unittest
from v43r1_measurement import canonical_graph, duplicate_free, graph_equal

class V43r1MeasurementTests(unittest.TestCase):
    def test_permutation_invariant(self):
        rows=[{"atom":"u:p:e2","allowed_values":[True]},{"atom":"u:p:e1","allowed_values":[False]}]
        self.assertTrue(graph_equal(rows,list(reversed(rows))))
        self.assertEqual(canonical_graph(rows),canonical_graph(list(reversed(rows))))
    def test_content_sensitive(self):
        left=[{"atom":"u:p:e1","allowed_values":[True]}]
        right=[{"atom":"u:p:e1","allowed_values":[False]}]
        self.assertFalse(graph_equal(left,right))
    def test_duplicates_rejected(self):
        row={"atom":"u:p:e1","allowed_values":[True]}
        self.assertFalse(duplicate_free([row,row]))
        self.assertFalse(graph_equal([row,row],[row,row]))
if __name__=="__main__": unittest.main()
