import streamlit as st
from openai import OpenAI
import google.generativeai as genai
import anthropic
import tiktoken
import requests
from bs4 import BeautifulSoup

# --- Helper Functions ---

@st.cache_data(show_spinner=False)
def fetch_url_content(url):
    """Fetches and extracts text content from a URL."""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()  # Raise an exception for bad status codes
        soup = BeautifulSoup(response.content, 'html.parser')
        # Remove script and style elements
        for script_or_style in soup(["script", "style"]):
            script_or_style.decompose()
        text = soup.get_text(separator='\n', strip=True)
        return text
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching URL {url}: {e}")
        return None

def get_token_count(text, model="gpt-5-nano"):
    """Returns the number of tokens in a text string."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        # Fallback for models not in tiktoken
        encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))

# --- Main Application ---

def main():
    st.title("🤖 URL-Based AI Chatbot")
    st.write("Enter URLs and ask questions about their content.")

    # --- Sidebar Configuration ---
    st.sidebar.header("Data Input")
    url1 = st.sidebar.text_input("URL 1", key="url1")
    url2 = st.sidebar.text_input("URL 2", key="url2")

    st.sidebar.header("Model Configuration")
    
    llm_provider = st.sidebar.selectbox(
        "Choose LLM Provider:",
        ("OpenAI", "Google Gemini", "Anthropic Claude")
    )
    
    use_advanced = st.sidebar.checkbox("Use advanced model")

    # --- CORRECTED: This entire block was moved inside main() ---
    
    # A dictionary to hold all provider-specific configurations
    LLM_CONFIG = {
        "OpenAI": {
            "models": ["gpt-5-mini", "gpt-5-nano"],
            "advanced_model": "gpt-5-chat-latest",
            "secret_key": "OPENAI_API_KEY"
        },
        "Google Gemini": {
            "models": ["gemini-2.5-flash-lite", "gemini-2.5-flash"],
            "advanced_model": "gemini-2.5-pro",
            "secret_key": "GOOGLE_API_KEY"
        },
        "Anthropic Claude": {
            "models": ["claude-3-5-haiku-20241022", "claude-sonnet-4-20250514"],
            "advanced_model": "claude-opus-4-20250514",
            "secret_key": "ANTHROPIC_API_KEY"
        }
    }

    # Get configuration for the selected provider
    provider_config = LLM_CONFIG.get(llm_provider)
    models_available = provider_config["models"]
    advanced_model = provider_config["advanced_model"]

    # Get and validate the API key dynamically
    api_key_name = provider_config["secret_key"]
    api_key = st.secrets.get(api_key_name)

    if not api_key:
        st.warning(f"{llm_provider} API key not found. Please set `{api_key_name}` in your secrets file.")
        st.stop()
    
    # CORRECTED: Added missing configuration step for Google Gemini
    if llm_provider == "Google Gemini":
        genai.configure(api_key=api_key)

    if use_advanced:
        selected_model = advanced_model
        st.sidebar.info(f"Using advanced model: **{selected_model}**")
    else:
        selected_model = models_available[0]
        st.sidebar.info(f"Using standard model: **{selected_model}**")

    st.sidebar.header("Memory Settings")
    memory_type = st.sidebar.radio(
        "Choose conversation memory type:",
        ("Buffer of 6 messages", "Conversation Summary", "Buffer of 2,000 tokens")
    )

    # --- Initialize Session State ---
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "conversation_summary" not in st.session_state:
        st.session_state.conversation_summary = ""

    # --- Display Chat History ---
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # --- Handle New User Input ---
    if prompt := st.chat_input("Ask a question about the content of the URLs..."):
        if not url1 and not url2:
            st.error("Please provide at least one URL in the sidebar.")
            st.stop()

        # Append and display user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Fetch and process URL content
        with st.spinner("Fetching and processing content from URLs..."):
            content1 = fetch_url_content(url1) if url1 else ""
            content2 = fetch_url_content(url2) if url2 else ""
            url_context = f"CONTENT FROM URL 1:\n{content1}\n\nCONTENT FROM URL 2:\n{content2}"


        # --- Create Conversation Buffer based on Memory Type ---
        messages_to_send = []
        if memory_type == "Buffer of 6 messages":
            messages_to_send = st.session_state.messages[-6:]
        elif memory_type == "Buffer of 2,000 tokens":
            current_tokens = 0
            for msg in reversed(st.session_state.messages):
                msg_tokens = get_token_count(msg["content"], selected_model)
                if current_tokens + msg_tokens <= 2000:
                    messages_to_send.insert(0, msg)
                    current_tokens += msg_tokens
                else:
                    break
        elif memory_type == "Conversation Summary":
            if st.session_state.conversation_summary:
                messages_to_send.append({"role": "system", "content": f"Here is a summary of the conversation so far: {st.session_state.conversation_summary}"})
            messages_to_send.append(st.session_state.messages[-1]) # Add the latest user prompt

        # --- Construct the final prompt for the API ---
        final_messages_for_api = [
            {"role": "system", "content": "You are a helpful assistant. Answer the user's question based on the provided URL content. If the answer is not in the content, say so."},
            {"role": "user", "content": f"CONTEXT:\n{url_context}\n\nQUESTION:\n{prompt}"}
        ]
        
        # --- API Call and Streaming Response ---
        try:
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                full_response_content = ""

                # Provider-specific API calls
                if llm_provider == "OpenAI":
                    client = OpenAI(api_key=api_key)
                    stream = client.chat.completions.create(
                        model=selected_model,
                        messages=final_messages_for_api,
                        stream=True,
                    )
                    for chunk in stream:
                        if chunk.choices[0].delta.content is not in [None, ""]:
                            full_response_content += chunk.choices[0].delta.content
                            message_placeholder.markdown(full_response_content + "▌")
                
                elif llm_provider == "Google Gemini":
                    model = genai.GenerativeModel(selected_model)
                    gemini_prompt = f"CONTEXT:\n{url_context}\n\nQUESTION:\n{prompt}"
                    stream = model.generate_content(gemini_prompt, stream=True)
                    for chunk in stream:
                        full_response_content += chunk.text
                        message_placeholder.markdown(full_response_content + "▌")

                # CORRECTED: Re-applied the fix for the Anthropic Claude API call
                elif llm_provider == "Anthropic Claude":
                    client = anthropic.Anthropic(api_key=api_key)
                    system_prompt = final_messages_for_api[0]['content']
                    claude_messages = final_messages_for_api[1:]

                    stream = client.messages.create(
                        model=selected_model,
                        max_tokens=2048,
                        system=system_prompt,
                        messages=claude_messages,
                        stream=True
                    )
                    for chunk in stream:
                        if chunk.type == "content_block_delta":
                            full_response_content += chunk.delta.text
                            message_placeholder.markdown(full_response_content + "▌")

                message_placeholder.markdown(full_response_content)
            
            st.session_state.messages.append({"role": "assistant", "content": full_response_content})

            # CORRECTED: Added a check for the OpenAI key before trying to summarize
            if memory_type == "Conversation Summary" and st.secrets.get("OPENAI_API_KEY"):
                with st.spinner("Creating conversation summary..."):
                    summary_prompt = "Please create a concise summary of the following conversation for your own memory."
                    conversation_for_summary = st.session_state.messages
                    
                    client = OpenAI(api_key=st.secrets.get("OPENAI_API_KEY")) 
                    response = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[
                            {"role": "system", "content": summary_prompt},
                            *conversation_for_summary
                        ],
                    )
                    st.session_state.conversation_summary = response.choices[0].message.content

        except Exception as e:
            st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()