import json,unittest
from v42_stateful import deterministic_world,entities,epistemic_rows,unary,effect
from v44_delayed import canonical_program,execute_partial,execute_sequence,mechanic_registry,schedule,_rule
from generate_v44_delayed import build_population
from v22r2_grounding import PROJECT_ROOT
class V44DelayedTests(unittest.TestCase):
 def test_delay_one_requires_next_tick(self):
  es=entities(2); world=deterministic_world(es,"v44-test"); world["u:active:unit_1"]=False; program=canonical_program({"rules":[_rule("pulse",delayed=[schedule(1,effect("set_true",unary("active","target")))]),_rule("route")]}); pulse={"id":"pulse","binding":{"actor":"unit_0","target":"unit_1"}}; wait={"id":"wait","binding":{}}; first=execute_sequence(program,es,world,[pulse])[0]; second=execute_sequence(program,es,world,[pulse,wait])[-1]; self.assertFalse(first["u:active:unit_1"]); self.assertTrue(second["u:active:unit_1"])
 def test_pending_events_not_flushed(self):
  es=entities(2); world=deterministic_world(es,"v44-flush"); world["u:active:unit_1"]=False; program=canonical_program({"rules":[_rule("pulse",delayed=[schedule(2,effect("set_true",unary("active","target")))]),_rule("route")]}); actions=[{"id":"pulse","binding":{"actor":"unit_0","target":"unit_1"}},{"id":"wait","binding":{}}]; self.assertFalse(execute_sequence(program,es,world,actions)[-1]["u:active:unit_1"]); self.assertTrue(execute_sequence(program,es,world,actions,"end_flush")[-1]["u:active:unit_1"])
 def test_partial_executor_unions_worlds(self):
  es=entities(2); world=deterministic_world(es,"v44-partial"); state=epistemic_rows(world,["u:active:unit_1"]); program=mechanic_registry()[0]["program"]; result=execute_partial([program],es,state,[{"id":"pulse","binding":{"actor":"unit_0","target":"unit_1"}},{"id":"wait","binding":{}}]); self.assertGreaterEqual(len(result["possible_final_observations"]),1)
 def test_registry_and_population_quotas(self):
  registry=mechanic_registry(); self.assertEqual(40,len(registry)); self.assertEqual(40,len({x["key"] for x in registry})); config=json.loads((PROJECT_ROOT/"configs/v44-deterministic-delayed-effects.json").read_text()); rows=build_population(config); self.assertEqual(40,len(rows)); self.assertEqual(960,sum(len(x["agent_input"]["queries"]) for x in rows))
if __name__=="__main__": unittest.main()
