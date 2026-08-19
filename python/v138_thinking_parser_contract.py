from __future__ import annotations

import json
import re
from collections import Counter
from typing import Any


THINK_TAG = re.compile(r"</?think>")


def inspect_template_contract(template_text: str) -> dict[str, bool]:
    direct_fragment = "<think>\\n\\n</think>\\n\\n"
    thinking_fragment = "<think>\\n"
    false_branch = "enable_thinking is defined and enable_thinking is false"
    return {
        "has_enable_thinking_false_branch": false_branch in template_text,
        "direct_prompt_supplies_closed_empty_trace": direct_fragment in template_text,
        "thinking_prompt_supplies_open_trace": thinking_fragment in template_text,
        "generation_prompt_is_assistant_scoped": "<|im_start|>assistant\\n" in template_text,
    }


def split_prompt_opened_thinking_suffix(raw: str) -> dict[str, Any]:
    stripped = raw.strip()
    depth = 1
    closure_end: int | None = None
    saw_suffix_tag = False
    for match in THINK_TAG.finditer(stripped):
        saw_suffix_tag = True
        token = match.group(0)
        if closure_end is not None:
            return {
                "valid_trace_contract": False,
                "reason": "thinking_tag_after_prompt_trace_closed",
                "final_text": "",
                "thinking_trace_present": True,
                "thinking_trace_closed": True,
            }
        if token == "<think>":
            depth += 1
        else:
            depth -= 1
            if depth < 0:
                return {
                    "valid_trace_contract": False,
                    "reason": "extra_thinking_close",
                    "final_text": "",
                    "thinking_trace_present": True,
                    "thinking_trace_closed": True,
                }
            if depth == 0:
                closure_end = match.end()
    if depth != 0 or closure_end is None:
        return {
            "valid_trace_contract": False,
            "reason": "prompt_opened_thinking_trace_not_closed",
            "final_text": "",
            "thinking_trace_present": True,
            "thinking_trace_closed": False,
        }
    return {
        "valid_trace_contract": True,
        "reason": "trace_contract_valid",
        "final_text": stripped[closure_end:].strip(),
        "thinking_trace_present": True,
        "thinking_trace_closed": True,
        "suffix_contains_thinking_tag": saw_suffix_tag,
    }


def validate_final_answer_v138(
    raw: str,
    catalog: dict[str, Any],
    *,
    thinking_enabled: bool,
    prompt_think_opened: bool,
) -> dict[str, Any]:
    stripped = raw.strip()
    if thinking_enabled:
        if not prompt_think_opened:
            raise ValueError("V138 thinking parsing requires the frozen prompt-opened contract")
        trace = split_prompt_opened_thinking_suffix(stripped)
        if not trace["valid_trace_contract"]:
            return {
                "answer_choice_id": "A00",
                "response_valid": False,
                "validation_reason": trace["reason"],
                "thinking_trace_present": trace["thinking_trace_present"],
                "thinking_trace_closed": trace["thinking_trace_closed"],
                "final_text": "",
            }
        final_text = trace["final_text"]
        trace_present = True
        trace_closed = True
    else:
        if prompt_think_opened:
            raise ValueError("direct inference must use the template's closed-empty-trace branch")
        if THINK_TAG.search(stripped):
            return {
                "answer_choice_id": "A00",
                "response_valid": False,
                "validation_reason": "unexpected_thinking_trace",
                "thinking_trace_present": True,
                "thinking_trace_closed": False,
                "final_text": "",
            }
        final_text = stripped
        trace_present = False
        trace_closed = True
    try:
        value = json.loads(final_text)
    except json.JSONDecodeError:
        return {
            "answer_choice_id": "A00",
            "response_valid": False,
            "validation_reason": "invalid_final_json",
            "thinking_trace_present": trace_present,
            "thinking_trace_closed": trace_closed,
            "final_text": final_text,
        }
    valid_ids = {row["choice_id"] for row in catalog["choices"]}
    if not isinstance(value, dict) or set(value) != {"choice_id"}:
        return {
            "answer_choice_id": "A00",
            "response_valid": False,
            "validation_reason": "invalid_final_keys",
            "thinking_trace_present": trace_present,
            "thinking_trace_closed": trace_closed,
            "final_text": final_text,
        }
    choice = value.get("choice_id")
    if not isinstance(choice, str) or choice not in valid_ids:
        return {
            "answer_choice_id": "A00",
            "response_valid": False,
            "validation_reason": "unknown_choice_id",
            "thinking_trace_present": trace_present,
            "thinking_trace_closed": trace_closed,
            "final_text": final_text,
        }
    return {
        "answer_choice_id": choice,
        "response_valid": True,
        "validation_reason": "valid",
        "thinking_trace_present": trace_present,
        "thinking_trace_closed": trace_closed,
        "final_text": final_text,
    }


def summarize_frozen_v137_metadata(result: dict[str, Any]) -> dict[str, Any]:
    thinking = [row for row in result["fixtures"].values() if row["condition_id"] == "thinking"]
    reasons = Counter(row["validation_reason"] for row in thinking)
    maximum_token_count = max(row["generated_token_count"] for row in thinking)
    return {
        "thinkingFixtureCount": len(thinking),
        "unclosedThinkingTraceCount": reasons["unclosed_thinking_trace"],
        "invalidFinalJsonCount": reasons["invalid_final_json"],
        "thinkingTracePresenceCount": sum(row["thinking_trace_present"] for row in thinking),
        "maximumTokenCount": maximum_token_count,
        "maximumTokenHitCount": sum(row["generated_token_count"] == maximum_token_count for row in thinking),
        "containsRawResponseOrTrace": any(
            {"raw_response", "thinking_trace", "final_text"} & set(row) for row in thinking
        ),
    }


__all__ = [
    "inspect_template_contract",
    "split_prompt_opened_thinking_suffix",
    "summarize_frozen_v137_metadata",
    "validate_final_answer_v138",
]
