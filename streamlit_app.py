"""Minimal Streamlit interface for the ChatGPT client."""

from __future__ import annotations

import streamlit as st

from chat_gpt_client import ChatGPTClient, DEFAULT_MODEL


def init_session() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "client_error" not in st.session_state:
        st.session_state.client_error = None


def build_sidebar() -> dict[str, float | str | None]:
    with st.sidebar:
        st.header("Settings")
        model = st.text_input("Model", value=DEFAULT_MODEL)
        temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.2)
        max_tokens = st.number_input(
            "Max Output Tokens (0 for default)", min_value=0, value=0, step=50
        )
    return {
        "model": model.strip() or DEFAULT_MODEL,
        "temperature": float(temperature),
        "max_output_tokens": max_tokens or None,
    }


def load_client(model: str) -> ChatGPTClient | None:
    try:
        return ChatGPTClient(model=model)
    except ValueError as exc:
        st.session_state.client_error = str(exc)
        return None


def main() -> None:
    st.set_page_config(page_title="ChatGPT Streamlit")
    st.title("ChatGPT Streamlit UI")
    init_session()

    sidebar_options = build_sidebar()
    client = load_client(sidebar_options["model"])

    if st.session_state.client_error:
        st.error(st.session_state.client_error)
        st.stop()

    prompt = st.text_area("Enter your message", height=150)

    col_send, col_clear = st.columns([3, 1])
    with col_send:
        send_clicked = st.button("Send", type="primary")
    with col_clear:
        clear_clicked = st.button("Clear Conversation")

    if clear_clicked:
        st.session_state.messages = []
        st.experimental_rerun()

    if send_clicked and prompt.strip():
        st.session_state.messages.append({"role": "user", "content": prompt.strip()})
        try:
            response = client.chat(
                st.session_state.messages,
                temperature=sidebar_options["temperature"],
                max_output_tokens=sidebar_options["max_output_tokens"],
            )
            st.session_state.messages.append({"role": "assistant", "content": response})
        except Exception as exc:  # pylint: disable=broad-except
            st.error(f"Request failed: {exc}")

    if not st.session_state.messages:
        st.info("Start the conversation by sending a message.")
    else:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])


if __name__ == "__main__":
    main()

