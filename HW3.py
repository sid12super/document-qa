import streamlit as st
from openai import OpenAI
import tiktoken

def get_token_count(text, model="gpt-4o"):
    """Returns the number of tokens in a text string."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        print("Warning: model not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))

def main():
    """
    Main function to run the Streamlit chatbot application.
    This version includes an interactive follow-up question and a larger default buffer.
    """
    st.title("🤖 Interactive GPT-4o Chatbot")
    st.write("I'm a streaming chatbot with configurable memory and an interactive Q&A flow.")

    # --- Sidebar Configuration ---
    st.sidebar.header("Configuration")
    st.sidebar.info(
        "For this app to work, set your OpenAI API key in Streamlit's secrets: "
        "`.streamlit/secrets.toml`\n\n"
        "`OPENAI_API_KEY = \"sk-...\"`"
    )

    st.sidebar.header("Buffer Settings")
    buffer_type = st.sidebar.radio(
        "Choose buffer type:",
        ("Message Count (Default)", "Token Limit")
    )

    max_tokens = 0
    if buffer_type == "Token Limit":
        max_tokens = st.sidebar.number_input(
            "Max tokens for buffer:",
            min_value=100, max_value=4000, value=1000, step=100
        )

    # --- Initialize API Client ---
    try:
        api_key = st.secrets.get("OPENAI_API_KEY")
        if not api_key:
            st.error("OpenAI API key not found. Please follow instructions in the sidebar.")
            st.stop()
        client = OpenAI(api_key=api_key)
    except Exception as e:
        st.error(f"Failed to initialize OpenAI client: {e}")
        st.stop()

    # --- Initialize Session State ---
    if "messages" not in st.session_state:
        st.session_state.messages = []
    # 'last_question' will store the prompt for which we might ask for more info.
    if "last_question" not in st.session_state:
        st.session_state.last_question = None

    # --- Display Chat History ---
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # --- Handle New User Input ---
    if prompt := st.chat_input("What would you like to ask?"):
        # Append and display the user's message immediately.
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # --- Core Logic for Interactive Follow-up ---
        
        is_yes_response = st.session_state.last_question and prompt.lower().strip() in ['yes', 'yep', 'sure', 'ok', 'okay', 'please do']
        is_no_response = st.session_state.last_question and prompt.lower().strip() in ['no', 'nope', 'nah', 'no thanks']

        if is_no_response:
            # If "no", reset the state and provide a polite closing.
            st.session_state.last_question = None
            response_text = "Alright. What else can I help you with?"
            st.session_state.messages.append({"role": "assistant", "content": response_text})
            with st.chat_message("assistant"):
                st.markdown(response_text)
        else:
            # For a new question or a "yes" response, we call the LLM.
            try:
                # If user said "yes", formulate a new prompt asking for more detail.
                if is_yes_response:
                    api_prompt = f"Please provide more detailed information about: '{st.session_state.last_question}'"
                    # Keep track of the original question
                    question_to_remember = st.session_state.last_question
                else:
                    # Otherwise, it's a new question.
                    api_prompt = prompt
                    question_to_remember = prompt

                # --- Create the Conversation Buffer ---
                messages_to_send = []
                if buffer_type == "Token Limit":
                    current_tokens = 0
                    for msg in reversed(st.session_state.messages):
                        msg_tokens = get_token_count(msg["content"])
                        if current_tokens + msg_tokens <= max_tokens:
                            messages_to_send.insert(0, msg)
                            current_tokens += msg_tokens
                        else:
                            break
                else:  # Default to Message Count buffer (now larger)
                    messages_to_send = st.session_state.messages[-20:]

                messages_for_api = messages_to_send[:-1] + [{"role": "user", "content": api_prompt}]

                # --- API Call and Streaming Response ---
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        stream = client.chat.completions.create(
                            model="gpt-4o",
                            messages=messages_for_api,
                            stream=True,
                        )
                        response_stream = st.write_stream(stream)
                
                # After the stream is complete, formulate the final response and update state
                final_response = response_stream + "\n\n**DO YOU WANT MORE INFO?**"
                st.session_state.messages.append({"role": "assistant", "content": final_response})
                
                # Crucially, update the last_question state *after* the response
                st.session_state.last_question = question_to_remember

                # Rerun to display the final_response from the message history
                st.rerun()

            except Exception as e:
                st.error(f"An error occurred with the OpenAI API: {e}")

if __name__ == "__main__":
    main()