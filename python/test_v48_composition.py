import json,unittest
from collections import Counter
from v22_relational import canonical_json
from v46_stochastic import mechanic_registry as v46_registry
from v47_sampling import mechanic_registry as v47_registry
from v48_composition import alias_signature,mechanic_registry

class V48CompositionTests(unittest.TestCase):
 def test_registry_is_fresh_and_balanced(self):
  rows=mechanic_registry(); keys={x["key"] for x in rows}; previous={x["key"] for x in v46_registry()}|{x["key"] for x in v47_registry()}; self.assertEqual(48,len(rows)); self.assertFalse(keys&previous); self.assertEqual({"1/4":16,"1/2":16,"3/4":16},dict(Counter(x["probability"] for x in rows)))
 def test_alias_signature_sorts_after_aliasing(self):
  signature=canonical_json([{"atom":"u:active:unit_0","value":True},{"atom":"u:active:unit_1","value":False}]); result=json.loads(alias_signature(signature,{"unit_0":"z","unit_1":"a"})); self.assertEqual(["u:active:a","u:active:z"],[x["atom"] for x in result])

if __name__=="__main__": unittest.main()
