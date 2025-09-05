import streamlit as st
import os
import requests
from bs4 import BeautifulSoup

# --- Import SDKs for the different LLMs ---
from openai import OpenAI
import google.generativeai as genai
import anthropic

def read_url_content(url):
    """Fetches and returns the text content of a given URL."""
    try:
        response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        for script_or_style in soup(["script", "style"]):
            script_or_style.decompose()
        return soup.get_text(separator='\n', strip=True)
    except requests.RequestException as e:
        st.error(f"Error fetching URL: {e}")
        return None

def main():
    st.title("🌐 Multi-LLM URL Summarizer")
    st.write("Enter a URL to compare summaries from different AI models!")

    # --- Sidebar for all user options ---
    st.sidebar.header("Configuration")

    llm_provider = st.sidebar.selectbox(
        "Choose LLM Provider:",
        ("OpenAI", "Google Gemini", "Anthropic Claude")
    )

    api_key = None
    models_available = []

    if llm_provider == "OpenAI":
        models_available = ["gpt-4o", "gpt-4-turbo"]
        api_key = st.secrets.get("OPENAI_API_KEY")
        if not api_key:
            st.error("OpenAI API key not found. Please set it in .streamlit/secrets.toml")
            st.stop()

    elif llm_provider == "Google Gemini":
        models_available = ["gemini-1.5-pro-latest", "gemini-1.5-flash-latest"]
        api_key = st.secrets.get("GOOGLE_API_KEY")
        if not api_key:
            st.error("Google API key not found. Please set it in .streamlit/secrets.toml")
            st.stop()
        genai.configure(api_key=api_key)

    elif llm_provider == "Anthropic Claude":
        # --- FIXED THE ERROR HERE ---
        # Replaced the premium 'claude-3-opus' model with the more accessible 'claude-3-haiku'
        # model to resolve the '404 model not found' error, which is common for keys
        # that don't have access to the most expensive model.
        models_available = ["claude-3-haiku-20240307", "claude-3-sonnet-20240229"]
        api_key = st.secrets.get("ANTHROPIC_API_KEY")
        if not api_key:
            st.error("Anthropic API key not found. Please set it in .streamlit/secrets.toml")
            st.stop()

    model = st.sidebar.selectbox("Choose the model:", options=models_available)
    summary_type = st.sidebar.radio(
        "Choose summary style:",
        ("100 words", "2 paragraphs", "5 bullet points")
    )
    language = st.sidebar.selectbox(
        "Choose the output language:",
        options=["English", "French", "Spanish", "German"]
    )

    url = st.text_input("Enter the URL to summarize:", placeholder="https://example.com")

    if st.button(f"Generate Summary with {llm_provider}"):
        if not url:
            st.warning("Please enter a URL to generate a summary.")
            st.stop()

        with st.spinner("Fetching content from URL..."):
            document = read_url_content(url)

        if not document:
            st.error("Could not retrieve content from the URL.")
            st.stop()

        if summary_type == "100 words":
            instruction = "Summarize the document in about 100 words."
        elif summary_type == "2 paragraphs":
            instruction = "Summarize the document in exactly 2 connecting paragraphs."
        else:
            instruction = "Summarize the document in 5 concise bullet points."

        prompt = f"Here’s content from a URL: {document}\n\n---\n\n{instruction}. Please provide the summary in {language}."

        with st.spinner(f"Generating summary with {model}..."):
            try:
                summary = ""
                if llm_provider == "OpenAI":
                    client = OpenAI(api_key=api_key)
                    response = client.chat.completions.create(
                        model=model,
                        messages=[{"role": "user", "content": prompt}]
                    )
                    summary = response.choices[0].message.content

                elif llm_provider == "Google Gemini":
                    model_instance = genai.GenerativeModel(model)
                    response = model_instance.generate_content(prompt)
                    summary = response.text

                elif llm_provider == "Anthropic Claude":
                    client = anthropic.Anthropic(api_key=api_key)
                    response = client.messages.create(
                        model=model,
                        max_tokens=1024,
                        messages=[{"role": "user", "content": prompt}]
                    )
                    summary = response.content[0].text

                st.subheader(f"Summary from {llm_provider} ({model})")
                st.write(summary)

            except Exception as e:
                st.error(f"An error occurred with the {llm_provider} API: {e}")

if __name__ == "__main__":
    main()

