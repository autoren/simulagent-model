"""Fresh registry and alias utilities for V48 stochastic language composition."""
from __future__ import annotations
import json
from fractions import Fraction
from v22_relational import canonical_json,parse_atom,relation_atom,sha256_text,unary_atom
from v46_stochastic import PROBABILITIES,mechanic_registry as v46_registry,program_key
from v47_sampling import _candidate_rows,mechanic_registry as v47_registry

FAMILIES=("immediate_bernoulli_mutation","delayed_bernoulli_scheduling","state_conditional_probability","interleaved_deterministic_and_stochastic")

def mechanic_registry():
 excluded={x["key"] for x in v46_registry()}|{x["key"] for x in v47_registry()}; grouped={(family,p):[] for family in FAMILIES for p in PROBABILITIES}
 for family,probability,program in _candidate_rows():
  key=program_key(program)
  if key not in excluded: grouped[(family,probability)].append((sha256_text(key),key,program))
 rows=[]
 for family in FAMILIES:
  ordinal=0
  for probability in PROBABILITIES:
   candidates=sorted(grouped[(family,probability)])
   if len(candidates)<4: raise RuntimeError(f"V48 lacks fresh candidates for {family}/{probability}")
   for _,key,program in candidates[:4]:
    rows.append({"family":family,"ordinal":ordinal,"probability":str(probability),"timing":"delayed" if any(r["stochastic_delayed"] for r in program["rules"]) else "immediate","program":program,"key":key,"id":f"mechanic_{sha256_text(key)[:16]}"}); ordinal+=1
 keys={x["key"] for x in rows}
 if len(rows)!=48 or len(keys)!=48 or keys&excluded: raise RuntimeError("V48 registry must be fresh, unique, and balanced")
 return rows

def aliases_for(mechanic_id,entity_ids): return {identifier:f"e{sha256_text(f'v48|{mechanic_id}|{identifier}')[:10]}" for identifier in sorted(entity_ids)}

def alias_atom(atom,aliases):
 parsed=parse_atom(atom)
 return unary_atom(parsed[1],aliases[parsed[2]]) if parsed[0]=="u" else relation_atom(parsed[1],aliases[parsed[2]],aliases[parsed[3]])

def alias_state(rows,aliases): return [{"atom":alias_atom(x["atom"],aliases),"allowed_values":list(x["allowed_values"])} for x in rows]

def signature_rows(signature): return [{"atom":x["atom"],"allowed_values":[x["value"]]} for x in json.loads(signature)]

def alias_signature(signature,aliases):
 rows=[{"atom":alias_atom(x["atom"],aliases),"value":x["value"]} for x in json.loads(signature)]
 return canonical_json(sorted(rows,key=lambda x:x["atom"]))

def alias_distribution(distribution,aliases):
 return [{"trajectory":[alias_signature(signature,aliases) for signature in row["trajectory"]],"mass":dict(row["mass"])} for row in distribution]

def probability_posterior(registry,weights):
 result={str(p):0.0 for p in PROBABILITIES}
 for mechanic,weight in zip(registry,weights,strict=True): result[mechanic["probability"]]+=float(weight)
 return result
