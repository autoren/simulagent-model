"""Streamlit UI for Phase 1 of the prospective language pilot.

Run from the repository root with:

    .venv/bin/streamlit run python/prospective_language_pilot_app.py

This app intentionally contains no assistant/model invocation.  It collects
and locks every initial participant request before the research proceeds.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import streamlit as st

from prospective_language_pilot import (
    PilotProtocolError,
    completed_count,
    initialize_or_load_session,
    load_export_bytes,
    load_study_config,
    lock_initial_response,
    next_incomplete_record_id,
    normalize_participant_code,
    scenario_index,
)
from prospective_language_phase3 import (
    eligible_clarification_records,
    initialize_or_load_phase3,
    load_controller_outputs,
    lock_clarification_response,
    next_clarification_record_id,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPOSITORY_ROOT / "configs" / "prospective-language-pilot-v1.json"
DEFAULT_STORAGE_ROOT = REPOSITORY_ROOT / "data" / "prospective-language-pilot"
STORAGE_ROOT = Path(os.environ.get("SIMULAGENT_PILOT_DATA_DIR", DEFAULT_STORAGE_ROOT))
PHASE3_CONFIG_PATH = REPOSITORY_ROOT / "configs" / "prospective-language-pilot-v1-phase3.json"
FINAL_CLOSURE_PATH = REPOSITORY_ROOT / "outputs" / "simulagent-final-closure" / "result.json"


st.set_page_config(
    page_title="Prospective Language Pilot",
    page_icon="🧭",
    layout="centered",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .block-container { max-width: 900px; padding-top: 2rem; padding-bottom: 4rem; }
      .pilot-kicker { color: #586174; font-size: 0.86rem; font-weight: 700;
                      letter-spacing: 0.08em; text-transform: uppercase; }
      .scenario-card { border: 1px solid rgba(120, 130, 150, 0.28); border-radius: 14px;
                       padding: 1.15rem 1.25rem; background: rgba(125, 135, 155, 0.06);
                       margin: 0.8rem 0 1.1rem 0; }
      .private-goal { border-left: 4px solid #7f6ee8; padding: 0.7rem 0.9rem;
                      background: rgba(127, 110, 232, 0.08); border-radius: 5px; }
      .locked-row { color: #3c7c59; }
      div[data-testid="stAlert"] { border-radius: 12px; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def get_config() -> dict:
    return load_study_config(CONFIG_PATH)


def begin_session(config: dict) -> None:
    try:
        participant_code = normalize_participant_code(st.session_state.participant_code_input)
        participant_dir, _ = initialize_or_load_session(config, participant_code, STORAGE_ROOT)
    except PilotProtocolError as exc:
        st.error(str(exc))
        return
    st.session_state.active_participant_code = participant_code
    st.session_state.participant_dir = str(participant_dir)


def switch_participant() -> None:
    st.session_state.pop("active_participant_code", None)
    st.session_state.pop("participant_dir", None)


def load_final_closure() -> dict | None:
    if not FINAL_CLOSURE_PATH.is_file():
        return None
    closure = json.loads(FINAL_CLOSURE_PATH.read_text(encoding="utf-8"))
    if closure.get("project_status") != "closed":
        return None
    return closure


def show_final_closure(closure: dict) -> None:
    pilot = closure["prospective_pilot"]
    st.markdown('<div class="pilot-kicker">Final research disposition</div>', unsafe_allow_html=True)
    st.title("The Simulagent study is closed")
    st.success("The existing records are frozen. No further input or assistant run is authorized.", icon="🔒")
    st.write(
        "Phase 1 preserved 16 participant-authored requests. Phase 2 completed as a negative development result. "
        "The exploratory clarification phase stopped after one of 11 eligible responses because the fictional "
        "scenario did not define the additional facts the assistant asked for."
    )
    col1, col2, col3 = st.columns(3)
    col1.metric("Initial requests", pilot["phase1"]["locked_record_count"])
    col2.metric("Valid Phase 2 records", f'{pilot["phase2"]["structurally_valid_count"]} / 16')
    col3.metric("Clarifications locked", f'{pilot["phase3"]["locked_response_count"]} / 11')
    st.info(
        "This was a protocol-design limitation, not a participant failure. The remaining clarification responses "
        "will not be collected, and no terminal plan or defer evaluation will be generated."
    )
    st.markdown(
        "See `docs/simulagent-final-closure.md` for the final conclusions and "
        "`docs/simulagent-unpursued-tracks.md` for work that remains open or belongs in a successor project."
    )


def show_landing(config: dict) -> None:
    st.markdown('<div class="pilot-kicker">Prospective language study · Phase 1</div>', unsafe_allow_html=True)
    st.title("Write all initial requests first")
    st.write(
        "You will receive 16 varied scenario cards. For each one, write the first message you would "
        "naturally send to an assistant. No assistant response or model generation occurs in this phase."
    )
    st.info(
        "Your wording is the research contribution. The scenarios are controlled, but the evaluated "
        "assistant will later see only your locked request and its limited assistant context—not your "
        "private goal or facts."
    )

    with st.expander("See a non-scored example", expanded=False):
        st.markdown("**Scenario:** You need to plan a picnic for several friends, but rain is possible.")
        st.markdown("**Private goal:** Make a workable plan with an indoor backup.")
        st.markdown(
            "**A natural first request might be:** “Can you help me sort out a picnic for Saturday? "
            "The weather may not cooperate.”"
        )
        st.caption("This example is not part of the 16 scored records and is never saved.")

    st.subheader("Before you begin")
    for instruction in config["participant_instructions"]:
        st.markdown(f"- {instruction}")

    st.text_input(
        "Participant code",
        value="P001",
        max_chars=32,
        key="participant_code_input",
        help="Use a pseudonymous code. Do not enter your name or email address.",
    )
    understood = st.checkbox(
        "I understand that submitted requests are immutable and that no assistant will respond until all 16 are locked."
    )
    st.button(
        "Start or resume Phase 1",
        type="primary",
        disabled=not understood,
        on_click=begin_session,
        args=(config,),
        use_container_width=True,
    )


def show_locked_review(config: dict, state: dict) -> None:
    index = scenario_index(config)
    with st.expander(f"Review {completed_count(state)} locked record(s)", expanded=False):
        if not state["initial_responses"]:
            st.caption("No requests have been locked yet.")
            return
        for record_id in state["scenario_order"]:
            response = state["initial_responses"].get(record_id)
            if response is None:
                continue
            status = "Unable to respond" if response["response_status"] == "unable_to_respond" else "Locked"
            st.markdown(f"**{record_id} · {index[record_id]['title']}** — {status}")
            if response["initial_request"]:
                st.caption(response["initial_request"])


def show_completion(config: dict, participant_dir: Path, state: dict) -> None:
    st.success("Phase 1 is complete. All 16 initial requests are locked.", icon="✅")
    st.markdown(
        "### Stop here\n"
        "No assistant has seen these records and no model has generated a response. The next research "
        "step is a separate frozen batch run."
    )
    st.metric("Assistant generations", "0")
    st.progress(1.0, text="16 of 16 initial requests locked")

    exports = load_export_bytes(participant_dir)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.download_button(
            "Download public requests",
            data=exports["public"],
            file_name=f"{state['participant_code']}-phase1-public.jsonl",
            mime="application/x-ndjson",
            use_container_width=True,
        )
    with col2:
        st.download_button(
            "Download private record",
            data=exports["private"],
            file_name=f"{state['participant_code']}-phase1-private.jsonl",
            mime="application/x-ndjson",
            use_container_width=True,
        )
    with col3:
        st.download_button(
            "Download audit manifest",
            data=exports["audit"],
            file_name=f"{state['participant_code']}-phase1-audit.json",
            mime="application/json",
            use_container_width=True,
        )

    st.caption(f"Local protected session: {participant_dir}")
    show_locked_review(config, state)


def phase3_paths(participant_dir: Path) -> tuple[Path, Path, Path]:
    result = participant_dir / "assistant" / "phase2_architecture" / "result.json"
    outputs = (
        participant_dir
        / "assistant"
        / "phase2_architecture"
        / "participant"
        / "phase2_controller_outputs.jsonl"
    )
    lock = participant_dir / "audit" / "phase3_clarification_lock.json"
    return result, outputs, lock


def show_phase3(config: dict, participant_dir: Path, phase1_state: dict) -> bool:
    result_path, outputs_path, lock_path = phase3_paths(participant_dir)
    if not (PHASE3_CONFIG_PATH.is_file() and result_path.is_file() and outputs_path.is_file() and lock_path.is_file()):
        return False
    phase3_config = json.loads(PHASE3_CONFIG_PATH.read_text(encoding="utf-8"))
    result = json.loads(result_path.read_text(encoding="utf-8"))
    phase3_lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if result.get("completed") is not True or phase3_lock.get("interaction_authorized") is not True:
        return False
    controller_rows = load_controller_outputs(outputs_path)
    eligible = eligible_clarification_records(phase3_config, controller_rows)
    phase3_state = initialize_or_load_phase3(
        phase3_config, config, participant_dir, controller_rows, phase3_lock
    )
    total = len(eligible)
    done = len(phase3_state["responses"])

    with st.sidebar:
        st.divider()
        st.markdown("### Clarification batch")
        st.metric("Clarifications locked", f"{done} / {total}")
        st.caption("No model runs while you answer these questions.")

    if phase3_state["phase"] == "phase_3_complete_waiting_for_terminal_run":
        st.success(f"All {total} clarification responses are locked.", icon="✅")
        st.markdown(
            "### Stop here\n"
            "No terminal continuation has been generated. The research runner must audit this complete batch first."
        )
        st.metric("Assistant generations during clarification collection", "0")
        st.progress(1.0, text=f"{total} of {total} clarification responses locked")
        st.caption(f"Local protected session: {participant_dir}")
        return True

    record_id = next_clarification_record_id(phase3_state)
    if record_id is None:
        return True
    row = next(item for item in eligible if item["record_id"] == record_id)
    scenario = scenario_index(config)[record_id]
    phase1_response = phase1_state["initial_responses"][record_id]
    card = scenario["participant_card"]
    position = phase3_state["clarification_order"].index(record_id) + 1

    st.markdown('<div class="pilot-kicker">Exploratory clarification batch</div>', unsafe_allow_html=True)
    st.title("Answer the assistant’s clarification")
    st.progress(done / total, text=f"{done} of {total} locked · Clarification {position} of {total}")
    st.caption(f"Opaque record ID: {record_id}")
    st.subheader(scenario["title"])
    st.write(card["setting"])
    st.markdown(
        f'<div class="private-goal"><strong>Your private goal</strong><br>{card["private_goal"]}</div>',
        unsafe_allow_html=True,
    )
    st.markdown("**Facts available to you**")
    for fact in card["known_facts"]:
        st.markdown(f"- {fact}")
    st.markdown("**Your locked initial request**")
    st.info(phase1_response["initial_request"])
    st.markdown("**The assistant asks**")
    for question in row["clarification_questions"]:
        st.warning(question)
    st.caption(
        "Answer naturally using what you know from the card. It is valid to say that you do not know. "
        "No model will respond until every clarification is locked."
    )

    unable_key = f"phase3_unable_{record_id}"
    answer_key = f"phase3_answer_{record_id}"
    unable = st.checkbox("I cannot answer these clarification questions", key=unable_key)
    answer = st.text_area(
        "Your response to the assistant",
        height=150,
        disabled=unable,
        key=answer_key,
        placeholder="Write what you would naturally reply…",
    )
    unable_reason = None
    unable_note = ""
    if unable:
        labels = {
            "do_not_know": "I do not know the requested information",
            "question_unclear": "The clarification is unclear",
            "prefer_not_to_answer": "I prefer not to answer",
            "other": "Other",
        }
        selected = st.selectbox("Why can’t you answer?", list(labels.values()), key=f"phase3_reason_{record_id}")
        unable_reason = next(key for key, label in labels.items() if label == selected)
        unable_note = st.text_input("Optional note", max_chars=300, key=f"phase3_note_{record_id}")
    attestation = st.checkbox(
        "This is my own response, and I understand that locking it is irreversible.",
        key=f"phase3_attest_{record_id}",
    )
    if st.button("Lock clarification and continue", type="primary", disabled=not attestation, use_container_width=True):
        try:
            lock_clarification_response(
                phase3_config,
                config,
                participant_dir,
                controller_rows,
                phase3_state,
                record_id=record_id,
                answer="" if unable else answer,
                unable_reason=unable_reason if unable else None,
                unable_note=unable_note,
                participant_attestation=attestation,
            )
        except PilotProtocolError as exc:
            st.error(str(exc))
        else:
            for key in list(st.session_state):
                if key.startswith(f"phase3_") and key.endswith(record_id):
                    st.session_state.pop(key, None)
            st.rerun()
    return True


def show_active_scenario(config: dict, participant_dir: Path, state: dict) -> None:
    index = scenario_index(config)
    total = len(config["scenarios"])
    done = completed_count(state)
    record_id = next_incomplete_record_id(state)
    if record_id is None:
        show_completion(config, participant_dir, state)
        return

    scenario = index[record_id]
    position = state["scenario_order"].index(record_id) + 1
    card = scenario["participant_card"]

    st.markdown('<div class="pilot-kicker">Phase 1 · Initial requests only</div>', unsafe_allow_html=True)
    st.title("Write your first message")
    st.progress(done / total, text=f"{done} of {total} locked · Scenario {position} of {total}")
    st.caption(f"Opaque record ID: {record_id}")

    st.markdown('<div class="scenario-card">', unsafe_allow_html=True)
    st.subheader(scenario["title"])
    st.write(card["setting"])
    st.markdown(f'<div class="private-goal"><strong>Your private goal</strong><br>{card["private_goal"]}</div>', unsafe_allow_html=True)
    st.markdown("**Facts available to you**")
    for fact in card["known_facts"]:
        st.markdown(f"- {fact}")
    st.markdown("</div>", unsafe_allow_html=True)

    st.warning(
        "Use your own natural wording. Do not copy the private-goal sentence, list research labels, or "
        "try to anticipate the assistant’s ideal question."
    )

    unable = st.checkbox(
        "I genuinely cannot provide a natural initial request for this scenario",
        key=f"unable_{record_id}",
    )
    request = st.text_area(
        "What would you initially say to the assistant?",
        height=150,
        placeholder="Write the message you would naturally send…",
        disabled=unable,
        key=f"request_{record_id}",
    )
    unable_reason = None
    unable_note = ""
    if unable:
        reason_labels = {
            "scenario_unclear": "The scenario is unclear to me",
            "would_not_ask_an_assistant": "I would not ask an assistant for this",
            "cannot_form_natural_request": "I cannot form a natural request",
            "other": "Other",
        }
        selected_label = st.selectbox(
            "Why can’t you provide a request?",
            list(reason_labels.values()),
            key=f"unable_reason_{record_id}",
        )
        unable_reason = next(key for key, label in reason_labels.items() if label == selected_label)
        unable_note = st.text_input(
            "Optional note",
            key=f"unable_note_{record_id}",
            max_chars=300,
        )

    attestation = st.checkbox(
        "This is my own response, and I understand that locking it is irreversible.",
        key=f"attest_{record_id}",
    )

    if st.button(
        "Lock request and continue",
        type="primary",
        disabled=not attestation,
        use_container_width=True,
    ):
        try:
            lock_initial_response(
                config,
                participant_dir,
                state,
                record_id=record_id,
                initial_request="" if unable else request,
                unable_reason=unable_reason if unable else None,
                unable_note=unable_note,
                participant_attestation=attestation,
            )
        except PilotProtocolError as exc:
            st.error(str(exc))
        else:
            for key in (
                f"unable_{record_id}",
                f"request_{record_id}",
                f"unable_reason_{record_id}",
                f"unable_note_{record_id}",
                f"attest_{record_id}",
            ):
                st.session_state.pop(key, None)
            st.rerun()

    st.caption(
        "Submitting stores separate public, private, and audit projections locally. It does not call a model."
    )
    show_locked_review(config, state)


def show_study(config: dict) -> None:
    participant_code = st.session_state["active_participant_code"]
    participant_dir, state = initialize_or_load_session(config, participant_code, STORAGE_ROOT)

    with st.sidebar:
        st.markdown("### Prospective pilot")
        st.write(f"Participant: `{participant_code}`")
        st.write(f"Phase: `{state['phase']}`")
        st.metric("Locked", f"{completed_count(state)} / {len(config['scenarios'])}")
        st.divider()
        st.caption("No assistant generation is enabled in Phase 1.")
        st.button("Switch participant code", on_click=switch_participant, use_container_width=True)

    if state["phase"] == "phase_1_complete_waiting_for_assistant_run" and show_phase3(
        config, participant_dir, state
    ):
        return
    if state["phase"] == "phase_1_complete_waiting_for_assistant_run":
        show_completion(config, participant_dir, state)
    else:
        show_active_scenario(config, participant_dir, state)


def main() -> None:
    try:
        config = get_config()
    except PilotProtocolError as exc:
        st.error(f"Study configuration error: {exc}")
        st.stop()

    closure = load_final_closure()
    if closure is not None:
        with st.sidebar:
            st.markdown("### Study status")
            st.write("`closed`")
            st.caption("Read-only evidence preservation; no further collection or generation.")
        show_final_closure(closure)
        return

    if "active_participant_code" not in st.session_state:
        with st.sidebar:
            st.markdown("### Phase boundary")
            st.caption("This app collects initial requests only. Assistant processing comes later.")
        show_landing(config)
    else:
        try:
            show_study(config)
        except PilotProtocolError as exc:
            st.error(f"Protected session error: {exc}")
            st.stop()


if __name__ == "__main__":
    main()
