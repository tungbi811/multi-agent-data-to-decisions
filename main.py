import streamlit as st

st.set_page_config(page_title="🤖 Multi-Agent for Data Science", layout="wide")

from multi_agents.runtime import AnalysisRuntime  # noqa: E402
from multi_agents.workspace import RunWorkspace  # noqa: E402
from utils.events import (  # noqa: E402
    coder_code_from_tool_call,
    next_or_none,
    safe_content,
    safe_event_type,
    safe_role,
)
from utils.sidebar import Sidebar  # noqa: E402
from utils.utils import display_group_chat  # noqa: E402


SESSION_DEFAULTS = {
    "messages": [],
    "events": None,
    "event": None,
    "runtime": None,
    "awaiting_response": False,
    "user_input": "",
    "terminated": False,
    "last_agent_name": None,
}
RUN_STATE_DEFAULTS = {
    key: SESSION_DEFAULTS[key]
    for key in (
        "events",
        "event",
        "awaiting_response",
        "user_input",
        "terminated",
        "last_agent_name",
    )
}


def initialize_state() -> None:
    for key, value in SESSION_DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = list(value) if isinstance(value, list) else value


def cleanup_runtime(status: str | None = None) -> bool:
    runtime = st.session_state.runtime
    if runtime is None:
        return True
    try:
        if status is None:
            runtime.close()
        else:
            runtime.finish(status)
    except Exception:
        return False
    st.session_state.runtime = None
    return True


def reset_run_state() -> None:
    for key, value in RUN_STATE_DEFAULTS.items():
        st.session_state[key] = value


def reset_state() -> bool:
    if not cleanup_runtime():
        return False
    for key in SESSION_DEFAULTS:
        st.session_state.pop(key, None)
    return True


def stop_with_message(message: str, diagnostic: str) -> None:
    runtime = st.session_state.runtime
    if runtime is not None:
        runtime.record_failure(diagnostic)
    cleaned = cleanup_runtime("failed")
    st.session_state.messages.append({"role": "System", "content": message})
    if not cleaned:
        st.session_state.messages.append(
            {"role": "System", "content": "Runtime cleanup is incomplete; retry Restart."}
        )
    st.session_state.terminated = True


def process_event(event, requirement: str) -> None:
    event_type = safe_event_type(event)
    if event_type == "unknown":
        st.session_state.messages.append(
            {"role": "System", "content": "Unsupported event received."}
        )
        return

    content = getattr(event, "content", None)
    if event_type == "text":
        sender = safe_role(getattr(content, "sender", "System"))
        message = safe_content(getattr(content, "content", None))
        if not (sender == "User" and isinstance(message, str) and requirement.strip() in message):
            st.session_state.messages.append({"role": sender, "content": message})
    elif event_type == "tool_call":
        raw_sender = getattr(content, "sender", None)
        sender = safe_role(raw_sender)
        if sender == "Coder":
            code = coder_code_from_tool_call(event)
            if code is None:
                st.session_state.messages.append(
                    {"role": "System", "content": "Malformed Coder tool call received."}
                )
            else:
                st.session_state.messages.append(
                    {"role": "Coder", "content": code, "in_expander": True}
                )
        elif isinstance(raw_sender, str) and sender != "System":
            st.session_state.last_agent_name = sender
        else:
            st.session_state.messages.append(
                {"role": "System", "content": "Malformed tool call received."}
            )
    elif event_type == "tool_response":
        sender = safe_role(st.session_state.last_agent_name)
        message = safe_content(getattr(content, "content", None))
        st.session_state.messages.append(
            {
                "role": sender,
                "content": message,
                "in_expander": sender != "BusinessAnalyst",
            }
        )
        st.session_state.last_agent_name = None
    elif event_type == "input_request":
        st.session_state.awaiting_response = True
    elif event_type == "run_completion":
        st.session_state.messages.append({"role": "System", "content": safe_content(content)})
        if not cleanup_runtime("completed"):
            st.session_state.messages.append(
                {"role": "System", "content": "Runtime cleanup is incomplete; retry Restart."}
            )
        st.session_state.terminated = True


initialize_state()
sidebar = Sidebar()

_, content_column, _ = st.columns([0.05, 0.9, 0.05])
with content_column:
    st.title("🤖 Multi-Agent for Data Science")
    st.write(
        "👋 Upload your dataset and describe your requirements in the sidebar, "
        "then click **Run Analysis** to start."
    )

    if st.sidebar.button("🚀 Run Analysis", use_container_width=True, key="run_analysis"):
        if not sidebar.api_key:
            st.warning("Please enter your API key to proceed.")
            st.stop()
        if not sidebar.uploaded_files:
            st.warning("Please upload at least one dataset to proceed.")
            st.stop()
        if not sidebar.user_requirements.strip():
            st.warning("Please describe your data analysis requirements to proceed.")
            st.stop()

        if not cleanup_runtime():
            st.error("Unable to replace the active analysis because cleanup is incomplete.")
            st.stop()
        reset_run_state()
        workspace = None
        try:
            workspace = RunWorkspace.create()
            for uploaded in sidebar.uploaded_files:
                workspace.save_upload(uploaded.name, bytes(uploaded.getbuffer()))
            runtime = AnalysisRuntime.start(sidebar.api_key, workspace)
            st.session_state.runtime = runtime
            st.session_state.messages.append({"role": "User", "content": sidebar.user_requirements})
            st.session_state.events = runtime.run(sidebar.user_requirements)
        except Exception:
            runtime = st.session_state.runtime
            if runtime is None:
                if workspace is not None:
                    try:
                        workspace.close()
                    except Exception:
                        pass
            else:
                runtime.record_failure("Analysis startup failed.")
                cleanup_runtime()
            st.error("Unable to start analysis. Check the upload and configuration, then retry.")
            st.stop()

    if st.sidebar.button("🔄 Restart", use_container_width=True, key="restart"):
        if reset_state():
            st.rerun()
        st.session_state.terminated = True
        st.session_state.awaiting_response = False
        st.session_state.user_input = ""
        st.session_state.event = None
        st.session_state.last_agent_name = None
        st.error("Unable to restart because runtime cleanup is incomplete.")
        st.stop()

    display_group_chat()
    if not st.session_state.terminated and not st.session_state.awaiting_response:
        if st.session_state.events is not None:
            if st.session_state.user_input:
                try:
                    st.session_state.event.content.respond(st.session_state.user_input)
                    st.session_state.user_input = ""
                except Exception:
                    stop_with_message(
                        "Unable to submit the response safely.",
                        "User response delivery failed.",
                    )
            else:
                try:
                    with st.spinner("Loading...", show_time=True):
                        event, exhausted = next_or_none(st.session_state.events)
                except Exception:
                    stop_with_message(
                        "Analysis stopped because its event stream failed.",
                        "Event iterator failed.",
                    )
                else:
                    if exhausted:
                        stop_with_message(
                            "Analysis ended before a completion event was received.",
                            "Event stream ended before completion.",
                        )
                    else:
                        st.session_state.event = event
                        try:
                            runtime = st.session_state.runtime
                            if runtime is None:
                                raise RuntimeError("Analysis runtime is unavailable.")
                            runtime.record_event(event)
                            process_event(event, sidebar.user_requirements)
                        except Exception:
                            stop_with_message(
                                "Analysis stopped while processing an event.",
                                "Event processing failed.",
                            )
            st.rerun()
    elif st.session_state.terminated:
        st.info(
            "The analysis has ended. You can restart the process by clicking "
            "the 'Restart' button in the sidebar."
        )

    if st.session_state.awaiting_response:
        user_input = st.text_area(
            "Replying as User. Type 'exit' to end the conversation:",
            key="user_input",
        )
        if st.button("Submit Response", key="submit_response"):
            if user_input.strip():
                st.session_state.awaiting_response = False
                st.rerun()
            else:
                st.warning("Please enter a response before submitting.")
