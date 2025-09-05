import streamlit as st
from openai import OpenAI
import os
import requests
from bs4 import BeautifulSoup

def read_url_content(url):
    """Fetches and returns the text content of a given URL."""
    try:
        response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
        response.raise_for_status()  # Raise an exception for HTTP errors
        soup = BeautifulSoup(response.content, 'html.parser')
        # Remove script and style elements
        for script_or_style in soup(["script", "style"]):
            script_or_style.decompose()
        return soup.get_text(separator='\n', strip=True)
    except requests.RequestException as e:
        st.error(f"Error fetching URL: {e}")
        return None

def main():
    """Main function for the URL Summarizer app page."""
    # Show title and description.
    st.title("🌐 Sid's URL Summarizer")
    st.write("Enter a URL and choose how you’d like it summarized!")

    # Retrieve the OpenAI API key from Streamlit Secrets
    openai_api_key = st.secrets.get("OPENAI_API_KEY")

    if not openai_api_key:
        st.error("No API key found. Please set it in .streamlit/secrets.toml.")
        return

    client = OpenAI(api_key=openai_api_key)

    # --- Sidebar Options ---
    st.sidebar.header("Summary Options")

    # Sidebar: summary options
    summary_type = st.sidebar.radio(
        "Choose summary style:",
        ("100 words", "2 paragraphs", "5 bullet points")
    )

    # Sidebar: language selection
    language = st.sidebar.selectbox(
        "Choose the output language:",
        options=["English", "French", "Spanish", "German"],
        index=0
    )

    # Sidebar: model selection
    model = st.sidebar.selectbox(
        "Choose the model:",
        options=["gpt-4o", "gpt-4o-mini", "gpt-4-turbo"],
        index=1  # default = gpt-4o-mini
    )

    # --- Main Page ---
    # Input for the URL
    url = st.text_input("Enter the URL you want to summarize:", placeholder="https://example.com")

    if st.button("Generate Summary"):
        if url:
            # Read and process the URL content
            with st.spinner("Fetching content from URL..."):
                document = read_url_content(url)

            if document:
                # Prompt instruction based on summary type
                if summary_type == "100 words":
                    instruction = "Summarize the document in about 100 words."
                elif summary_type == "2 paragraphs":
                    instruction = "Summarize the document in exactly 2 connecting paragraphs."
                else:  # 5 bullet points
                    instruction = "Summarize the document in 5 concise bullet points."

                # Prepare the prompt for the OpenAI API, including language instruction
                messages = [
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that summarizes web content in the user's specified language."
                    },
                    {
                        "role": "user",
                        "content": f"Here’s content from a URL: {document}\n\n---\n\n{instruction}. Please provide the summary in {language}."
                    }
                ]

                # Generate the summary
                with st.spinner(f"Generating summary in {language} using {model}..."):
                    try:
                        response = client.chat.completions.create(
                            model=model,
                            messages=messages,
                            stream=False,
                        )
                        st.subheader("Generated Summary")
                        st.write(response.choices[0].message.content)
                    except Exception as e:
                        st.error(f"Error generating summary: {e}")
        else:
            st.warning("Please enter a URL to generate a summary.")

if __name__ == "__main__":
    main()
