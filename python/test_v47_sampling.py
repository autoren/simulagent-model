import unittest
from collections import Counter
from fractions import Fraction

from v42_stateful import deterministic_world, effect, entities, unary
from v46_stochastic import _rule, canonical_program, delayed, stochastic
from v47_sampling import execute_joint_distribution, joint_map, mechanic_registry, sample_trajectory


class V47SamplingTests(unittest.TestCase):
    def test_joint_distribution_preserves_cross_step_dependence(self):
        es=entities(2); world=deterministic_world(es,"v47-joint"); world["u:active:unit_1"]=False
        program=canonical_program({"rules":[_rule("pulse",stochastic_immediate=[stochastic("1/2",effect("toggle",unary("active","target")))]),_rule("route")]})
        actions=[{"id":"pulse","binding":{"actor":"unit_0","target":"unit_1"}},{"id":"wait","binding":{}}]
        distribution=execute_joint_distribution(program,es,world,actions)
        self.assertEqual(2,len(distribution)); self.assertEqual(Fraction(1),sum(joint_map(distribution).values()))
        self.assertTrue(all(row["trajectory"][0]==row["trajectory"][1] for row in distribution))
    def test_delayed_samples_match_exact_support(self):
        es=entities(2); world=deterministic_world(es,"v47-sample"); program=canonical_program({"rules":[_rule("pulse",stochastic_delayed=[delayed(1,stochastic("1/4",effect("toggle",unary("ready","target"))))]),_rule("route")]}); actions=[{"id":"pulse","binding":{"actor":"unit_0","target":"unit_1"}},{"id":"wait","binding":{}}]
        support=set(joint_map(execute_joint_distribution(program,es,world,actions)))
        self.assertTrue(all(__import__('json').dumps(sample_trajectory(program,es,world,actions,seed),sort_keys=True,separators=(",",":")) in support for seed in range(100)))
    def test_fresh_registry_is_balanced_and_disjoint(self):
        registry=mechanic_registry(); self.assertEqual(48,len(registry)); self.assertEqual({"1/4":16,"1/2":16,"3/4":16},dict(Counter(row["probability"] for row in registry)))


if __name__=="__main__": unittest.main()
