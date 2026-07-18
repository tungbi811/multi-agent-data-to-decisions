from html import escape

import streamlit as st

ROLE_EMOJI = {
    "User": "🧑‍💻",
    "BusinessAnalyst": "💼",
    "BusinessTranslator": "🗣️",
    "DataAnalyst": "🔎",
    "DataEngineer": "🛠️",
    "DataScientist": "📊",
    "Coder": "🧠",
    "Assistant": "🤖",
    "System": "⚙️",
    "CodeExecutor": "💻",
}


def safe_md(text: str) -> str:
    return escape(text, quote=False)


def display_group_chat():
    expander_buffer = []  # temporary buffer for consecutive expander messages

    for msg in st.session_state.messages:
        role = msg["role"]
        content = msg["content"]
        in_expander = msg.get("in_expander", False)

        if in_expander:
            expander_buffer.append(msg)
        else:
            # If we hit a normal message and there are buffered expander messages, render them first
            if expander_buffer:
                with st.expander("💡 Detailed Response", expanded=False):
                    for emsg in expander_buffer:
                        erole = emsg["role"]
                        econtent = emsg["content"]
                        with st.chat_message(erole, avatar=ROLE_EMOJI.get(erole, "")):
                            st.markdown(f"**{erole}**")
                            if erole in ["Coder", "System"]:
                                st.code(econtent)
                            else:
                                if "```markdown" in econtent:
                                    st.write(
                                        safe_md(
                                            econtent.split("```markdown")[1].split("```")[0].strip()
                                        )
                                    )
                                else:
                                    st.write(safe_md(econtent))
                expander_buffer = []  # reset buffer

            # Render this normal message
            with st.chat_message(role, avatar=ROLE_EMOJI.get(role, "")):
                st.markdown(f"**{role}**")
                if "```markdown" in content:
                    st.write(safe_md(content.split("```markdown")[1].split("```")[0].strip()))
                else:
                    st.write(safe_md(content))

    if expander_buffer:
        with st.expander("🧠 Thinking ...", expanded=False):
            for emsg in expander_buffer:
                erole = emsg["role"]
                econtent = emsg["content"]
                with st.chat_message(erole, avatar=ROLE_EMOJI.get(erole, "")):
                    st.markdown(f"**{erole}**")
                    if erole in ["Coder", "System"]:
                        st.code(econtent)
                    else:
                        if "```markdown" in econtent:
                            st.write(
                                safe_md(econtent.split("```markdown")[1].split("```")[0].strip())
                            )
                        else:
                            st.write(safe_md(econtent))
