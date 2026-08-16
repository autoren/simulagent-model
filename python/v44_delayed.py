"""Deterministic delayed-effect DSL and hidden event-queue executor for V44."""
from __future__ import annotations
from itertools import product
from typing import Any,Sequence
from v22_relational import canonical_expression,canonical_json,evaluate_expression,sha256_text
from v42_stateful import (
    ONTOLOGY,atom_universe,compatible_worlds,deterministic_world,entities,epistemic_rows,
    action_bindings,unary,relation,negate,effect,canonical_effect,effect_key,
    _effect_assignment,world_signature,
)

ACTIONS=("pulse","route","wait")
RULE_ACTIONS=("pulse","route")
DELAYED_OPS=("set_true","set_false","toggle")

def schedule(delay:int,payload:dict[str,Any],condition:dict[str,Any]|None=None)->dict[str,Any]:
    row={"delay":delay,"effect":payload}
    if condition is not None: row["condition"]=condition
    return row

def _rule(action:str,immediate=(),delayed=()):
    return {"action":action,"immediate_effects":list(immediate),"delayed_effects":list(delayed)}

def canonical_schedule(value:dict[str,Any])->dict[str,Any]:
    delay=value.get("delay")
    if delay not in (1,2): raise ValueError("V44 delay must be one or two ticks")
    payload=canonical_effect(value.get("effect",{}))
    if payload["op"] not in DELAYED_OPS: raise ValueError("V44 delayed payload must be set, clear, or toggle")
    row={"delay":delay,"effect":payload}
    if "condition" in value:
        row["condition"]=canonical_expression(value["condition"])
    return row

def canonical_program(program:dict[str,Any])->dict[str,Any]:
    rules=[]; seen=set(); delayed_total=0
    for source in program.get("rules",[]):
        action=source.get("action")
        if action not in RULE_ACTIONS or action in seen: raise ValueError("V44 requires unique pulse and route rules")
        seen.add(action); immediate=[canonical_effect(x) for x in source.get("immediate_effects",[])]; delayed=[canonical_schedule(x) for x in source.get("delayed_effects",[])]
        if len(immediate)>1 or len(delayed)>1: raise ValueError("V44 permits at most one immediate and one delayed effect per action")
        delayed_total+=len(delayed); immediate.sort(key=effect_key); delayed.sort(key=lambda x:canonical_json(x)); rules.append({"action":action,"immediate_effects":immediate,"delayed_effects":delayed})
    if seen!=set(RULE_ACTIONS) or delayed_total<1: raise ValueError("V44 programs require pulse, route, and at least one delayed effect")
    rules.sort(key=lambda x:x["action"]); return {"dsl_version":3,"rules":rules}

def program_key(program): return canonical_json(canonical_program(program))

def _validate_action(action,entity_rows):
    action_id=action.get("id")
    if action_id not in ACTIONS: raise ValueError(f"Unknown V44 action: {action_id}")
    binding=action.get("binding",{})
    if action_id=="wait":
        if binding: raise ValueError("V44 wait has no binding")
        return action_id,binding
    identifiers={row["id"] for row in entity_rows}
    if set(binding)!={"actor","target"} or binding["actor"]==binding["target"] or not set(binding.values())<=identifiers: raise ValueError("V44 action requires two distinct known entities")
    return action_id,binding

def _assign_effects(effects,world,binding,entity_rows):
    assignments={}
    for payload in effects:
        assigned=_effect_assignment(payload,world,binding,entity_rows)
        if assigned is None: continue
        atom,value=assigned
        if atom in assignments and assignments[atom]!=value: raise ValueError("V44 simultaneous effects conflict")
        assignments[atom]=value
    return assignments

def _deliver_due(queue,tick,world,entity_rows):
    due=[event for event in queue if event["due"]==tick]; remaining=[event for event in queue if event["due"]!=tick]; assignments={}
    for event in due:
        atom,value=_effect_assignment(event["effect"],world,event["binding"],entity_rows)
        if atom in assignments: raise ValueError("V44 same-target same-tick event conflict")
        assignments[atom]=value
    return {**world,**assignments},remaining

def _scheduled(rule,world,binding,entity_rows,tick):
    events=[]
    for item in rule["delayed_effects"]:
        if "condition" in item and not evaluate_expression(item["condition"],ONTOLOGY,entity_rows,world,binding): continue
        events.append({"due":tick+item["delay"],"effect":item["effect"],"binding":dict(binding)})
    return events

def execute_sequence(program,entity_rows,world,actions,control="queued"):
    if set(world)!=set(atom_universe(entity_rows)): raise ValueError("V44 requires a complete Boolean world")
    normalized=canonical_program(program); by_action={row["action"]:row for row in normalized["rules"]}; current=dict(world); queue=[]; trajectory=[]
    for tick,action in enumerate(actions):
        action_id,binding=_validate_action(action,entity_rows)
        if control=="queued": current,queue=_deliver_due(queue,tick,current,entity_rows)
        rule=by_action.get(action_id)
        if rule is not None:
            new_events=_scheduled(rule,current,binding,entity_rows,tick)
            immediate=_assign_effects(rule["immediate_effects"],current,binding,entity_rows)
            current={**current,**immediate}
            if control=="collapsed_delay":
                collapsed=_assign_effects([event["effect"] for event in new_events],current,binding,entity_rows)
                current={**current,**collapsed}
            else: queue.extend(new_events)
        trajectory.append(dict(current))
    if not trajectory: raise ValueError("V44 sequences must contain an action")
    if control=="end_flush" and queue:
        for due in sorted({event["due"] for event in queue}):
            assignments={}
            for event in [row for row in queue if row["due"]==due]:
                atom,value=_effect_assignment(event["effect"],current,event["binding"],entity_rows)
                if atom in assignments: raise ValueError("V44 end-flush conflict")
                assignments[atom]=value
            current={**current,**assignments}
        trajectory[-1]=dict(current)
    return trajectory

def execute_partial(programs,entity_rows,initial_state,actions,control="queued"):
    if not programs: return {"possible_step_states":[],"possible_final_observations":[],"identifiable":False}
    steps=[set() for _ in actions]
    for world in compatible_worlds(initial_state):
        for program in programs:
            trajectory=execute_sequence(program,entity_rows,world,actions,control)
            for index,state in enumerate(trajectory): steps[index].add(world_signature(state))
    values=[sorted(row) for row in steps]; final=values[-1] if values else []
    return {"possible_step_states":values,"possible_final_observations":final,"identifiable":len(final)==1}

def mechanic_registry():
    mechanics=[]
    unary_targets=[unary(predicate,var) for predicate,var in product(("active","marked","ready"),("actor","target"))]
    variants=list(product(unary_targets,("set_true","set_false")))[:10]
    for index,(target,op) in enumerate(variants):
        program={"rules":[_rule("pulse",delayed=[schedule(1,effect(op,target))]),_rule("route",immediate=[effect("toggle",relation("actor","target"))])]}
        mechanics.append({"family":"one_tick_unary_set_clear","ordinal":index,"trigger_action":"pulse","delay":1,"program":canonical_program(program)})
    relational_variants=list(product(("pulse","route"),(relation("actor","target"),relation("target","actor")),DELAYED_OPS))[:10]
    for index,(trigger,target,op) in enumerate(relational_variants):
        other="route" if trigger=="pulse" else "pulse"; rules={trigger:_rule(trigger,delayed=[schedule(2,effect(op,target))]),other:_rule(other,immediate=[effect("toggle",unary("ready","actor"))])}
        program={"rules":[rules["pulse"],rules["route"]]}; mechanics.append({"family":"two_tick_relational_toggle","ordinal":index,"trigger_action":trigger,"delay":2,"program":canonical_program(program)})
    conditions=(unary("active","actor"),negate(unary("active","actor")),unary("marked","target"),relation("actor","target"),negate(relation("actor","target")))
    for index,(condition,op) in enumerate(product(conditions,("set_true","toggle"))):
        delay=1 if index%2==0 else 2; program={"rules":[_rule("pulse",immediate=[effect("toggle",unary("active","actor"))]),_rule("route",delayed=[schedule(delay,effect(op,unary("ready","target")),condition)])]}
        mechanics.append({"family":"state_conditional_scheduling","ordinal":index,"trigger_action":"route","delay":delay,"program":canonical_program(program)})
    interleaved=list(product((1,2),DELAYED_OPS,("active","marked")))[:10]
    for index,(delay,op,predicate) in enumerate(interleaved):
        target=unary(predicate,"target"); program={"rules":[_rule("pulse",delayed=[schedule(delay,effect(op,target))]),_rule("route",immediate=[effect("toggle",target)])]}
        mechanics.append({"family":"interleaved_immediate_and_delayed","ordinal":index,"trigger_action":"pulse","delay":delay,"program":canonical_program(program)})
    keys=[program_key(row["program"]) for row in mechanics]
    if len(mechanics)!=40 or len(set(keys))!=40: raise RuntimeError("V44 registry must contain 40 unique mechanics")
    for row,key in zip(mechanics,keys,strict=True): row["key"]=key; row["id"]=f"mechanic_{sha256_text(key)[:16]}"
    return mechanics
