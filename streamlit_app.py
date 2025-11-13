import streamlit as st
from openai import OpenAI

st.title("ChatGPT-like clone")

# Set OpenAI API key from Streamlit secrets or environment variables
try:
    # Try Streamlit secrets first (for local development)
    api_key = st.secrets["OPENAI_API_KEY"]
except:
    # Fallback to environment variable (for deployment)
    import os
    api_key = os.getenv("OPENAI_API_KEY")

if api_key and api_key != "your-openai-api-key-here":
    client = OpenAI(api_key=api_key)
    api_key_available = True
else:
    client = None
    api_key_available = False
    st.warning("⚠️ OpenAI API key not found. Please add your API key to `.streamlit/secrets.toml` or set OPENAI_API_KEY environment variable")

# Set a default model
if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-3.5-turbo"

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("What is up?"):
    if not api_key_available:
        st.error("Please configure your OpenAI API key in `.streamlit/secrets.toml` to use this app.")
        st.stop()
        
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate assistant response
    with st.chat_message("assistant"):
        try:
            response = client.chat.completions.create(
                model=st.session_state["openai_model"],
                messages=[
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages
                ],
                stream=True,
            )
            
            # Stream the response
            response_placeholder = st.empty()
            full_response = ""
            
            for chunk in response:
                if chunk.choices[0].delta.content is not None:
                    full_response += chunk.choices[0].delta.content
                    response_placeholder.markdown(full_response + "▌")
            
            response_placeholder.markdown(full_response)
            
        except Exception as e:
            st.error(f"Error generating response: {str(e)}")
            full_response = "Sorry, I encountered an error. Please try again."
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": full_response})