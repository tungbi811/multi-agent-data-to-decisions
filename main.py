import streamlit as st

st.set_page_config(page_title="🤖 Multi-Agent for Data Science", layout="wide")

from multi_agents.runtime import AnalysisRuntime  # noqa: E402
from multi_agents.workspace import RunWorkspace  # noqa: E402
from utils.events import coder_code_from_tool_call, next_or_none, safe_event_type  # noqa: E402
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


def initialize_state() -> None:
    for key, value in SESSION_DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = list(value) if isinstance(value, list) else value


def cleanup_runtime(status: str | None = None) -> None:
    runtime = st.session_state.runtime
    if runtime is None:
        return
    if status is None:
        runtime.close()
    else:
        runtime.finish(status)
    st.session_state.runtime = None


def reset_state() -> None:
    cleanup_runtime()
    for key in SESSION_DEFAULTS:
        st.session_state.pop(key, None)


def stop_with_message(message: str) -> None:
    cleanup_runtime("failed")
    st.session_state.messages.append({"role": "System", "content": message})
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
        sender = getattr(content, "sender", "System")
        message = getattr(content, "content", "")
        if not (sender == "User" and isinstance(message, str) and requirement.strip() in message):
            st.session_state.messages.append({"role": sender, "content": str(message)})
    elif event_type == "tool_call":
        sender = getattr(content, "sender", None)
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
        elif sender:
            st.session_state.last_agent_name = sender
        else:
            st.session_state.messages.append(
                {"role": "System", "content": "Malformed tool call received."}
            )
    elif event_type == "tool_response":
        sender = st.session_state.last_agent_name or "System"
        message = getattr(content, "content", "Malformed tool response received.")
        st.session_state.messages.append(
            {
                "role": sender,
                "content": str(message),
                "in_expander": sender != "BusinessAnalyst",
            }
        )
        st.session_state.last_agent_name = None
    elif event_type == "input_request":
        st.session_state.awaiting_response = True
    elif event_type == "run_completion":
        st.session_state.messages.append({"role": "System", "content": str(content)})
        cleanup_runtime("completed")
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

        cleanup_runtime()
        workspace = RunWorkspace.create()
        try:
            for uploaded in sidebar.uploaded_files:
                workspace.save_upload(uploaded.name, bytes(uploaded.getbuffer()))
            runtime = AnalysisRuntime.start(sidebar.api_key, workspace)
            st.session_state.runtime = runtime
            st.session_state.messages.append({"role": "User", "content": sidebar.user_requirements})
            st.session_state.events = runtime.run(sidebar.user_requirements)
        except Exception as exc:
            if st.session_state.runtime is None:
                workspace.close()
            else:
                cleanup_runtime()
            st.error(f"Unable to start analysis: {exc}")
            st.stop()

    if st.sidebar.button("🔄 Restart", use_container_width=True, key="restart"):
        reset_state()
        st.rerun()

    display_group_chat()
    if not st.session_state.terminated and not st.session_state.awaiting_response:
        if st.session_state.events is not None:
            if st.session_state.user_input:
                try:
                    st.session_state.event.content.respond(st.session_state.user_input)
                    st.session_state.user_input = ""
                except Exception as exc:
                    stop_with_message(f"Unable to submit response: {exc}")
            else:
                try:
                    with st.spinner("Loading...", show_time=True):
                        event, exhausted = next_or_none(st.session_state.events)
                    if exhausted:
                        stop_with_message("Analysis ended before a completion event was received.")
                    else:
                        st.session_state.event = event
                        runtime = st.session_state.runtime
                        if runtime is None:
                            raise RuntimeError("Analysis runtime is unavailable.")
                        runtime.record_event(event)
                        process_event(event, sidebar.user_requirements)
                except Exception as exc:
                    stop_with_message(f"Analysis stopped safely: {exc}")
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
