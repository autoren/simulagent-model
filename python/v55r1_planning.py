"""Planning-specific latent mechanics for V55r1 delay adequacy confirmation."""
from __future__ import annotations

import copy

from v22_relational import canonical_json, sha256_text
from v42_stateful import effect, relation, unary
from v46_stochastic import _rule, canonical_program, delayed, stochastic
from v53_smc2 import parameterize_program, template_key


DISTRACTOR_TARGETS = {
    "toggle_marked_actor": unary("marked", "actor"),
    "toggle_marked_target": unary("marked", "target"),
    "toggle_ready_actor": unary("ready", "actor"),
    "toggle_ready_target": unary("ready", "target"),
    "toggle_linked_actor_target": relation("actor", "target"),
    "toggle_linked_target_actor": relation("target", "actor"),
}


def _blueprint_program(blueprint: dict) -> dict:
    active_target = unary("active", blueprint["targetVar"])
    branch = stochastic(
        "1/2", effect(blueprint["operation"], active_target)
    )
    if blueprint["timing"] == "immediate":
        trigger_rule = _rule(
            blueprint["trigger"], stochastic_immediate=[branch]
        )
        delay_ticks = 0
    else:
        delay_ticks = {
            "delay_one": 1,
            "delay_two": 2,
        }[blueprint["timing"]]
        trigger_rule = _rule(
            blueprint["trigger"],
            stochastic_delayed=[delayed(delay_ticks, branch)],
        )
    distractor_rule = _rule(
        blueprint["distractorAction"],
        deterministic_immediate=[
            effect("toggle", DISTRACTOR_TARGETS[blueprint["distractor"]])
        ],
    )
    program = canonical_program({"rules": [trigger_rule, distractor_rule]})
    return program, delay_ticks


def planning_registry(config: dict) -> list[dict]:
    spec = config["planningSpecificRegistry"]
    rows = []
    for ordinal, blueprint in enumerate(spec["templateBlueprints"]):
        program, delay_ticks = _blueprint_program(blueprint)
        key = template_key(program)
        rows.append({
            "family": "v55r1_delay_adequacy",
            "family_ordinal": ordinal,
            "timing": blueprint["timing"],
            "delay": delay_ticks,
            "template": parameterize_program(program),
            "key": key,
            "id": f"v55r1_template_{sha256_text(key)[:16]}",
            "program_ordinal": ordinal,
            "blueprint": copy.deepcopy(blueprint),
        })
    if len(rows) != spec["templates"] or len({row["key"] for row in rows}) != len(rows):
        raise RuntimeError("V55r1 registry must contain eight unique templates")
    return rows


def registry_audit(registry: list[dict]) -> dict:
    delay_counts = {"delay_two": 0, "delay_one": 0, "immediate": 0}
    theta_branches = 0
    active_stochastic_targets = 0
    active_deterministic_targets = 0
    for row in registry:
        delay_counts[row["timing"]] += 1
        for rule in row["template"]["rules"]:
            branches = [
                *rule["stochastic_immediate"],
                *rule["stochastic_delayed"],
            ]
            theta_branches += len(branches)
            active_stochastic_targets += sum(
                branch["effect"]["target"].get("predicate") == "active"
                for branch in branches
            )
            active_deterministic_targets += sum(
                payload["target"].get("predicate") == "active"
                for payload in rule["deterministic_immediate"]
            )
    return {
        "templates": len(registry),
        "unique_template_keys": len({row["key"] for row in registry}),
        "delay_class_counts": delay_counts,
        "theta_branches": theta_branches,
        "active_stochastic_targets": active_stochastic_targets,
        "active_deterministic_targets": active_deterministic_targets,
        "registry_key": canonical_json([row["key"] for row in registry]),
    }


def delay_suppressed_registry(registry: list[dict], horizon: int) -> list[dict]:
    result = copy.deepcopy(registry)
    changed = 0
    for row in result:
        for rule in row["template"]["rules"]:
            for branch in rule["stochastic_delayed"]:
                if branch["delay"] == 2:
                    branch["delay"] = horizon + 1
                    changed += 1
    if changed != 4:
        raise RuntimeError("V55r1 delay counterfactual must suppress four branches")
    return result


def trigger_action(row: dict, actor: str, target: str) -> dict:
    return {
        "id": row["blueprint"]["trigger"],
        "binding": {"actor": actor, "target": target},
    }
